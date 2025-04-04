from tqdm import tqdm

import numpy as np
import gymnasium as gym

from .train_experiment import TrainExperiment
from SCOPIL.utils.exp_utils import print_latest_metrics_from_dict


class ICRLExperiment(TrainExperiment):

    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):
        super().__init__(environment, agent, config, file_results_dir, seed)

        # Retrieve information from the config file for training
        self.n_iters = config['ICRL']['n_iters']
        self.forward_steps = config['ICRL']['forward_steps']
        self.n_steps = config['ICRL']['n_steps']
        self.backward_n_rollouts = config['ICRL']['backward_n_rollouts']
        self.backward_iterations = config['ICRL']['backward_iterations']

        # Placeholders
        self.current_progress_remaining = 1
        self.pred_cost = None
        self.cur_iter_steps = None
        self.episode_step_counter = None
        self.episode_reward = None
        self.episode_game_counter = None
        self._iter = None

        # Initialize variables
        self.i_episode = 0

        # Initialize extra dict to keep track of "Constraint Net" information during training
        self.constraint_net_train_logs_dict = {}

    def train(self):
        """
        Code based on: https://github.com/shehryar-malik/icrl/blob/master/icrl/icrl.py
        """

        # In this experiment, we train an agent using ICRL.
        # The loop iterating over "n_iters" is used for alternating
        # between PPO updates and constraint-network updates.
        # The inner-loop represents the episodes, each of which consists of
        # (around) "forward_steps" environment steps.
        # The total number of the experiment is given by "n_iters*forward_steps".

        for self._iter in range(self.n_iters):

            # Initialize the variables of the current iteration
            self.initialize_iter_var_train()

            # Loop of episodes where samples are collected and used for PPO updates
            while self.cur_iter_steps < self.forward_steps:

                # Initialize the variables of the current episode and the buffer
                self.initialize_episode_var_train()

                print("\nEpisode: " + str(self.i_episode))

                # Loop of steps for each episode.
                # At the end of each episode, we perform "epochs" number of PPO updates
                while self.episode_step_counter < self.n_steps:

                    # At the beginning of each game, initialize the environment and the variables
                    self.initialize_game_var_train()

                    # Loop of the game
                    while not self.done:

                        # Get actions
                        action, reward_value, cost_value, logprob = self.agent.ppo.policy.forward(
                            self.observation, deterministic=False
                        )
                        clipped_action = self.agent.clip_action(action)

                        # Get predicted cost from the constraint-net
                        self.pred_cost = self.agent.constraint_net.cost_function(self.observation, action)

                        # Environment step
                        self.next_observation, self.reward, self.cost, self.done, self.truncated, self.info = \
                            self.env_step(clipped_action)

                        # Update variables after step
                        self.train_update_per_step_vars()

                        # add experiences to buffers
                        self.save_experience(
                            [
                                self.observation,
                                action,
                                # Normalize it here to get the logs with the unnormalized rewards
                                self.normalize_rewards_func(self.reward),
                                self.pred_cost,
                                self.done,
                                reward_value,
                                cost_value,
                                logprob
                            ]
                        )

                        # set the observation for the next step
                        self.observation = self.next_observation.copy()

                    #### End of game ####

                    # Update and store info for the current game, print avg logs
                    self.train_game_logging()

                    # Testing
                    self.test_during_training()

                ##### End of the episode #####

                # At the end of each episode's rollouts compute discounted returns and advantages
                _, reward_value, cost_value, _ = self.agent.ppo.policy.forward(self.observation, deterministic=False)
                self.agent.ppo.rollout_buffer.compute_returns_and_advantage(reward_value, cost_value)
                # PPO update networks, store and print useful info
                self.train_store_and_print_info()

            ###### End of PPO episodes for the current iteration ######

            observations_bw, actions_bw, lengths_bw = [], [], []

            # Loop of backward episodes to collect samples for training the constraint-net
            for bw_games in tqdm(range(self.backward_n_rollouts), desc='Backward game: '):

                self.eval_mode()
                done_bw, step_counter_bw = False, 0
                observation_bw, info_bw = self.env_reset()

                # Loop of a game of backward iterations
                while not done_bw:

                    step_counter_bw += 1

                    # Get action
                    action_bw = self.agent.ppo.policy.predict(observation_bw, deterministic=False)
                    clipped_action_bw = self.agent.clip_action(action_bw)

                    # Store samples
                    observations_bw.append(observation_bw.copy())
                    actions_bw.append(action_bw.copy())

                    # Environment step
                    next_observation_bw, reward_bw, cost_bw, done_bw, truncated_bw, info_bw = \
                        self.env_step(clipped_action_bw)

                    # Make done to be True when hitting the time horizon
                    if step_counter_bw == self.max_timesteps_per_game:
                        assert truncated_bw is True, f"'truncated': {truncated_bw}"
                        done_bw = True

                    # set the observation for the next step
                    observation_bw = next_observation_bw.copy()

                lengths_bw.append(step_counter_bw)

            ##### End of backwards rollouts #####
            self.train_constraint_net(observations_bw, actions_bw, lengths_bw)

        ####### End of experiment #######

        # Close the environment
        self.env.close()

    def initialize_episode_var_train(self):
        # Keep track of the total episodes
        self.i_episode += 1

        # Initialize the variables for the current episode
        self.episode_step_counter = 0
        self.episode_game_counter = 0
        self.episode_reward = 0

        # Initialize the buffer
        self.agent.ppo.rollout_buffer.reset()

    def initialize_iter_var_train(self):
        # Used to update the learning rate of the constraint-network.
        self.current_progress_remaining = 1 - float(self._iter) / float(self.n_iters)

        # Keep track of the current iteration's total step
        self.cur_iter_steps = 0

    def initialize_game_var_train(self):
        super().initialize_game_var_train()

        # Keep track of the current episode total games
        self.episode_game_counter += 1

    def train_update_per_step_vars(self):
        """
        Update variables after each step
        """

        super().train_update_per_step_vars()

        # Keep track of the current iteration's total step
        self.cur_iter_steps += 1

        # Keep track of the current episode total games
        self.episode_step_counter += 1

    def train_store_and_print_info(self):
        """
        Perform network updates
        """

        # Update networks' parameters
        training_returns = self.train_networks()

        # Store train metrics
        for key, value in training_returns.items():
            if key not in list(self.train_logs_dict.keys()):
                self.train_logs_dict[key] = []
            if value is not np.nan:
                self.train_logs_dict[key].append(value)

        # Print train metrics
        print_latest_metrics_from_dict(self.train_logs_dict)

    def train_networks(self) -> dict:
        # Set train mode
        self.train_mode()

        # Train
        training_returns = self.agent.ppo.train()

        # Go back to eval mode
        self.eval_mode()

        return training_returns

    def save_experience(self, data):
        """
        Saves an interaction to the replay buffer of the agent.
        :param data: List with data of the interaction to be stored in the Replay Buffer.
                     The list has the following elements:
                     1) observation
                     2) action
                     3) reward
                     4) cost
                     5) done
                     6) value
                     7) cost value
                     8) logprob
        """
        self.agent.ppo.rollout_buffer.add(*data)

    def get_agent_action_test(self, obs):
        """
        Get the agent action for testing
        :param obs: np.array, observation of the agent
        :return: np.array, action
        """

        action = self.agent.predict(obs, deterministic=True)

        return action

    def eval_mode(self):
        self.agent.ppo.policy.eval()
        self.agent.constraint_net.eval()

    def train_mode(self):
        self.agent.ppo.policy.train()
        self.agent.constraint_net.train()

    def train_constraint_net(self, obs, acts, lens):
        # Set train mode
        self.train_mode()

        training_returns = self.agent.constraint_net.train_network(
            self.backward_iterations,
            np.array(obs),
            np.array(acts),
            np.array(lens),
            self.current_progress_remaining
        )

        # Go back to eval mode
        self.eval_mode()

        # Store train metrics
        for key, value in training_returns.items():
            if key not in list(self.constraint_net_train_logs_dict.keys()):
                self.constraint_net_train_logs_dict[key] = []
            if value is not np.nan:
                self.constraint_net_train_logs_dict[key].append(value)

        # Print train metrics
        print_latest_metrics_from_dict(self.constraint_net_train_logs_dict)

