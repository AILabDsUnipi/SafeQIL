import itertools

import numpy as np

from .train_experiment import TrainExperiment


class SACExperiment(TrainExperiment):
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
        self.start_steps = config['SAC']['start_steps']
        self.num_steps = config['SAC']['num_steps']
        self.batch_size = config['SAC']['batch_size']
        self.update_every_steps = config['SAC']['update_every_steps']
        self.gradient_steps = config['SAC']['gradient_steps']
        self.use_sde = config['SAC']['use_sde']
        self.sde_sample_freq = config['SAC']['sde_sample_freq']
        self.w_constraint_optimization = config['SAC']['w_constraint_optimization']
        self.clip_grad_norm = config['SAC']['clip_grad_norm']
        self.pretrain = config['SAC']['pretrain']
        self.only_pretrain = config['SAC']['only_pretrain']
        self.w_mse = config['SAC']['w_mse']

        # Initialize lists to keep track of information and variables during training
        self.i_train_episode = 0
        self.train_logs_cur_game_dict = {}
        if self.pretrain is True:
            self.pretrain_logs_dict = {}

    def train(self):
        """
        Trains a SAC agent with or without constraints' optimization
        """

        # Pretrain the agent with the given demonstrations
        if self.pretrain is True:
            self.pretrain_func()
            if self.only_pretrain is True:
                # Close the environment
                self.env.close()
                # Dummy return just to avoid RL train
                return

        # RL training Loop
        while True:

            # At the beginning of each episode, initialize the environment and the variables
            self.initialize_game_var_train()

            print("\nEpisode: " + str(self.total_games))

            # Start the episode
            while not self.done:

                # Get action
                action, buffer_action = self.agent.sample_action(self.observation, self.total_steps)

                # Environment step
                self.next_observation, self.reward, self.cost, self.done, self.truncated, self.info = \
                    self.env_step(action)

                # Update variables after step
                self.train_update_per_step_vars()

                # add experiences to buffers
                self.save_experience(
                    [
                        self.observation,
                        self.next_observation,
                        buffer_action,
                        # Normalize it here to get the logs with the unnormalized rewards
                        self.normalize_rewards_func(self.reward),
                        self.fixed_done,
                        self.done,
                        self.truncated,
                        self.info
                    ]
                )

                # set the observation for the next step
                self.observation = self.next_observation.copy()

                # Train the networks and store the corresponding info
                self.train_store_and_print_info()

            ## End of game

            # Update, store and print info for the current game. Also, print avg logs
            self.train_game_logging()

            # Testing
            self.test_during_training()

            # Stop training
            if self.total_steps >= self.num_steps:
                break

        # Close the environment
        self.env.close()

    def pretrain_func(self):
        # Train
        self.pretrain_logs_dict = self.agent.pretrain_func()

        # Save the models
        if self.only_pretrain is True and self.save_model is True:
            print('Saving last models...')
            self.save_agent_models('last')

    def save_experience(self, data):
        """
        Saves an interaction to the replay buffer of the agent.
        :param data: List with data of the interaction to be stored in the Replay Buffer.
            The list has the following elements:
            1) observation
            2) next_observation
            3) action
            4) reward
            5) fixed done
            6) done
            7) truncated
            8) info
        """
        self.agent.replay_buffer.add(*data)

    def initialize_game_var_train(self):

        super().initialize_game_var_train()

        # Reset the noise for the SDE
        if self.use_sde:
            self.agent.actor.reset_noise(1)

        # Initialize lists for training details
        for key in self.train_logs_cur_game_dict.keys():
            self.train_logs_cur_game_dict[key] = []

    def train_update_per_step_vars(self):
        """
        Update variables after each step
        """

        super().train_update_per_step_vars()

        # Sample a new noise matrix if needed
        if self.use_sde and self.sde_sample_freq > 0 and self.train_step_counter % self.sde_sample_freq == 0:
            self.agent.actor.reset_noise(1)

    def train_store_and_print_info(self):
        """
        Perform network updates
        """

        if self.total_steps >= self.start_steps:

            # Increase 'i_train_episode' var
            if self.done:
                self.i_train_episode += 1

            # Update networks' parameters
            training_returns = self.train_networks()

            # Store train metrics
            for key, value in training_returns.items():
                if key not in list(self.train_logs_cur_game_dict.keys()):
                    self.train_logs_cur_game_dict[key] = []
                if value is not np.nan:
                    self.train_logs_cur_game_dict[key].append(value)

    def train_mode(self):
        """
        Set the agent in training mode
        """
        self.agent.policy.set_training_mode(True)

    def eval_mode(self):
        """
        Set the agent in evaluation mode
        """
        self.agent.policy.set_training_mode(False)

    def train_game_logging(self):

        super().train_game_logging()

        # Store all game train metrics
        if self.total_steps >= self.start_steps:
            for key, value in self.train_logs_cur_game_dict.items():
                if key not in list(self.train_logs_dict.keys()):
                    self.train_logs_dict[key] = []
                if len(value) > 0:
                    self.train_logs_dict[key].append(np.mean(value))

        ## logging per interval
        if (
                self.total_steps >= self.start_steps and
                (self.total_games % self.log_interval == 0 or self.total_games == 1)
        ):

            # Calculate and store per_log_interval values
            self.train_logging_per_interval()

        if self.debug_:
            # TODO: Print useful information
            pass

    def train_networks(self):
        # Set train mode
        self.train_mode()

        # Train
        training_returns = self.agent.train()

        # Go back to eval mode
        self.eval_mode()

        return training_returns

    def get_agent_action_test(self, obs):
        """
        Get the agent action for testing
        :param obs: np.array, observation of the agent
        :return: np.array, action
        """

        return self.agent.predict(obs, deterministic=True)
