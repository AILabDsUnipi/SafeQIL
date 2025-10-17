from abc import abstractmethod
from statistics import mean
from typing import Tuple

import numpy as np
import torch as th

from .base_experiment import BaseExperiment
from SCOPIL.utils.env_utils import get_cost_sum_from_info_dict, render
from SCOPIL.utils.exp_utils import test_print_logs, print_latest_metrics_from_dict
from SCOPIL.utils.demonstration_utils import normalize_features_func, normalize_reward_func
from SCOPIL.utils.file_utils import write_info_file


class TrainExperiment(BaseExperiment):
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
        self.normalize_features = config['Experiment']['normalize_features']
        self.normalize_rewards = config['Experiment']['normalize_rewards']
        self.max_timesteps_per_game = config['Experiment']['max_timesteps_per_game']
        self.log_interval = self.config['Experiment']['log_interval']
        self.test_every_episodes = config['Experiment']['test_every_episodes']
        self.save_model = self.config['Experiment']['save_model']
        self.use_image_obs = self.config['game']['use_image_obs']
        self.render = self.config['Experiment']['render']

        # Initialize lists/dicts/variables to keep track of information during training
        self.episodes_model_saved = {'last': [], 'best_reward': [], 'lowest_constr': []}
        self.test_avg_best_score = -np.inf
        self.test_avg_lowest_num_constr = np.inf
        self.total_games = 0
        self.game_reward = 0
        self.train_step_counter = 0
        self.done = False
        self.fixed_done = 0.
        self.observation = None
        self.next_observation = None
        self.reward = None
        self.cost = None
        self.truncated = None
        self.info = None
        self.reward_list = []
        self.step_list = []
        self.constraint_violation = None
        self.num_constraint_violation_list = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.freq_constraint_violation_list = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.step_list_avg_per_log_interval = []
        self.reward_list_avg_per_log_interval = []
        self.num_constraint_violation_per_log_interval = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.freq_constraint_violation_per_log_interval = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.train_logs_dict = {}
        self.train_logs_avg_per_log_interval_dict = {}

        # Initialize lists/dicts to keep track of information during testing
        self.test_step_list_avg_per_log_interval = []
        self.test_reward_list_avg_per_log_interval = []
        self.test_num_constraint_violation_per_log_interval = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.test_freq_constraint_violation_per_log_interval = {
            constraint_type: [] for constraint_type in self.constraint_types
        }

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def test_during_training(self):

        if self.total_games % self.test_every_episodes == 0 or self.total_games == 1:

            self.test_game_number += 1  # keep track of the testing session number
            print('\nTest {}'.format(self.test_game_number) + '\n')

            ## Run test
            # Evaluation mode
            self.eval_mode()
            with th.no_grad():
                # Test
                self.test_process()
            # Return to train mode
            self.train_mode()

            # Update per test metrics
            test_avg_reward, test_avg_num_constr = self.test_logging()

            # Save the models
            if self.save_model is True:
                if test_avg_reward > self.test_avg_best_score:
                    self.test_avg_best_score = test_avg_reward
                    print(
                        'Saving model... \nHighest reward achieved: ' + str(round(test_avg_reward, 2))
                    )
                    self.save_agent_models('best_reward')
                    self.episodes_model_saved['best_reward'].append(self.total_games)
                if test_avg_num_constr < self.test_avg_lowest_num_constr:
                    self.test_avg_lowest_num_constr = test_avg_num_constr
                    print(
                        'Saving model... \nLowest number of constraints achieved: ' + str(round(test_avg_num_constr, 2))
                    )
                    self.save_agent_models('lowest_constr')
                    self.episodes_model_saved['lowest_constr'].append(self.total_games)
                if self.total_games == 1 or self.total_games % self.test_every_episodes == 0:
                    print('Saving last models...')
                    self.save_agent_models('last')
                    self.episodes_model_saved['last'].append(self.total_games)

            # Print the average results of the test
            test_print_logs(
                test_avg_reward,
                self.test_step_list_avg_per_log_interval[-1],
                {
                    constraint_type: self.test_num_constraint_violation_per_log_interval[constraint_type][-1]
                    for constraint_type in self.constraint_types
                },
                {
                    constraint_type: self.test_freq_constraint_violation_per_log_interval[constraint_type][-1]
                    for constraint_type in self.constraint_types
                }
            )

    def test_logging(self) -> Tuple[float, float]:
        test_avg_reward = mean(self.test_reward_list[-self.test_max_games:])
        self.test_reward_list_avg_per_log_interval.append(test_avg_reward)
        self.test_step_list_avg_per_log_interval.append(
            mean(self.test_step_list[-self.test_max_games:])
        )
        for constraint_type in self.constraint_types:
            self.test_num_constraint_violation_per_log_interval[constraint_type].append(
                mean(self.test_num_constraint_violation_list[constraint_type][-self.test_max_games:])
            )
            self.test_freq_constraint_violation_per_log_interval[constraint_type].append(
                mean(self.test_freq_constraint_violation_list[constraint_type][-self.test_max_games:])
            )
        test_avg_num_constr = self.test_num_constraint_violation_per_log_interval['cost_sum'][-1]

        return test_avg_reward, test_avg_num_constr

    def test_process(self):

        for test_game_i in range(self.test_max_games):

            if self.debug_ is True:
                print('Test game: ' + str(test_game_i) + '\n')

            # Initialize test game variables
            test_step_counter = 0
            test_game_reward = 0
            test_game_num_constraint_violation = {constraint_type: 0 for constraint_type in self.constraint_types}
            test_done = False
            test_fixed_done = 0.

            # Reset environment. The cost is always 0 after resetting.
            test_obs, test_info = self.env_reset()
            assert len(test_info) == 0, f"'test_info': {test_info}"

            # Start to play a game
            while not test_done:

                # Get the action of the agent
                test_action = self.get_agent_action_test(test_obs)

                # Apply an environment step and get the results
                test_next_obs, test_reward, test_cost, test_done, test_truncated, test_info = \
                    self.env_step(test_action)

                # Print a message for the user if at least a constraint is violated
                if self.debug_ is True:
                    print('\n#########################')
                    print('Constraint violation!')
                    print('#########################\n')

                # Keep track of whether the game ended due to success or due to hitting the time horizon.
                test_fixed_done = float(test_done)
                if test_step_counter == self.test_max_timesteps_per_game - 1:
                    test_fixed_done = 0.
                    test_done = True

                if self.debug_ is True:
                    # TODO: Store the transition data
                    pass

                # Update the total reward
                test_game_reward += test_reward

                # Update the number of constraint violations
                test_game_num_constraint_violation.update({
                    constraint_type: test_game_num_constraint_violation[constraint_type] +
                                     (
                                         test_info.get(constraint_type, 0) if constraint_type != 'cost_sum'
                                         else
                                         get_cost_sum_from_info_dict(test_info)
                                     )
                    for constraint_type in self.constraint_types
                })

                # Set next as the current
                test_obs = test_next_obs.copy()

                # Update the step number of the game and the total steps of all games
                test_step_counter += 1

                if test_done is True and self.debug_ is True:
                    # TODO: Store the last observations and actions
                    pass

            ## End of test game

            # Update the testing metrics
            self.update_test_metrics(
                test_game_reward,
                test_step_counter,
                test_game_num_constraint_violation
            )

    @abstractmethod
    def get_agent_action_test(self, obs):
        """
        Get the action of the agent in test mode.
        Args:
            obs: np.array, observation of the agent.

        Returns:
            action: np.array, action of the agent.

        """
        raise NotImplementedError

    def initialize_game_var_train(self):

        # Evaluation mode for the agent networks
        self.eval_mode()

        # Keep track of the total games
        self.total_games += 1

        self.game_reward = 0
        self.train_step_counter = 0
        self.done = False

        # Keep track of constraint violations
        self.constraint_violation = {
            constraint_type: 0 for constraint_type in self.test_num_constraint_violation_list
        }

        # Reset environment
        self.observation, self.info = self.env_reset()
        assert len(self.info) == 0, f"'self.info': {self.info}"

        if self.debug_:
            # TODO: Store transitions for debugging
            pass

    def train_update_per_step_vars(self):
        """
        Updates variables after each step
        """

        # Episodic reward
        self.game_reward += self.reward

        # Make done to be True when hitting the time horizon
        if self.train_step_counter == self.max_timesteps_per_game - 1:
            assert self.truncated is True, f"'truncated': {self.truncated}"
            self.done = True

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)
        self.fixed_done = 0. if self.train_step_counter == self.max_timesteps_per_game - 1 else float(self.done)

        # keep track of the overall step number
        self.total_steps += 1

        # keep track of the step number for each game
        self.train_step_counter += 1

        # Update number of constraint violations
        self.constraint_violation.update({
            constraint_type: self.constraint_violation[constraint_type] +
                             (
                                 self.info.get(constraint_type, 0) if constraint_type != 'cost_sum'
                                 else
                                 get_cost_sum_from_info_dict(self.info)
                             )
            for constraint_type in self.constraint_types
        })

        if self.debug_ is True:
            # TODO: Store transitions for debugging
            pass

    @abstractmethod
    def train_mode(self):
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self):
        raise NotImplementedError

    @abstractmethod
    def save_experience(self, data):
        """
        Saves an interaction to the replay buffer of the agent.
        :param data: List with data of the interaction to be stored in the Replay Buffer.
        """
        raise NotImplementedError

    def train_game_logging(self):

        # Update training metrics about the experiment
        self.update_train_metrics()

        if self.total_games % self.log_interval == 0 or self.total_games == 1:

            # Calculate per_log_interval values
            reward_avg_per_log_interval = mean(self.reward_list[-self.log_interval:])
            steps_avg_per_log_interval = mean(self.step_list[-self.log_interval:])
            num_constraint_violation_list = {}
            freq_constraint_violation_list = {}
            for constraint_type in self.constraint_types:
                num_constraint_violation_list[constraint_type] = mean(
                    self.num_constraint_violation_list[constraint_type][-self.log_interval:]
                )
                freq_constraint_violation_list[constraint_type] = mean(
                    self.freq_constraint_violation_list[constraint_type][-self.log_interval:]
                )

            # Store per_log_interval values
            self.reward_list_avg_per_log_interval.append(reward_avg_per_log_interval)
            self.step_list_avg_per_log_interval.append(steps_avg_per_log_interval)
            for constraint_type in self.constraint_types:
                self.num_constraint_violation_per_log_interval[constraint_type].append(
                    num_constraint_violation_list[constraint_type]
                )
                self.freq_constraint_violation_per_log_interval[constraint_type].append(
                    freq_constraint_violation_list[constraint_type]
                )

            # Print per_log_interval values
            print('\n##########Average stats for training##########')
            print(
                'Total episodes until now: {}\n'
                'Total timesteps: {}\n'
                '##Avg over the last {} games##\n'
                'Avg steps: {}\n'
                'Avg per game reward: {}'.format(
                    self.total_games,
                    self.total_steps,
                    self.log_interval,
                    round(steps_avg_per_log_interval, 2),
                    round(reward_avg_per_log_interval, 2)
                )
            )
            for constraint_type in self.constraint_types:
                constraint_type_to_print = constraint_type.replace("cost_", "")
                print(
                    "\nAvg number of '{}' constraint violations: {}\n"
                    "Avg frequency of '{}' constraint violations: {}".format(
                        constraint_type_to_print,
                        round(num_constraint_violation_list[constraint_type], 2),
                        constraint_type_to_print,
                        round(freq_constraint_violation_list[constraint_type], 2)
                    )
                )

        if self.debug_:
            # TODO: Print useful information about game metrics
            pass

    def update_train_metrics(self):
        """
        Updates train metrics:
            reward_list,
            step_list,
            num_constraint_violation_list,
            freq_constraint_violation_list
        """

        # keep track of the game reward history
        self.reward_list.append(self.game_reward)

        # keep track of the game length in steps
        self.step_list.append(self.train_step_counter)

        # keep track of constraint violations
        for constraint_type in self.constraint_types:
            self.num_constraint_violation_list[constraint_type].append(
                self.constraint_violation[constraint_type]
            )
            self.freq_constraint_violation_list[constraint_type].append(
                self.constraint_violation[constraint_type] / self.train_step_counter
            )

    def update_test_metrics(
            self,
            game_reward,
            step_counter,
            test_constraint_violation_list,
    ):
        """
        Updates test metrics
        """

        # Keep track of the test game reward history
        self.test_reward_list.append(game_reward)

        # Keep track of the test game steps
        self.test_step_list.append(step_counter)

        # Keep track of constraint violations
        if self.debug_ is True:
            print()  # just an empty line
        for constraint_type in self.constraint_types:
            self.test_num_constraint_violation_list[constraint_type].append(
                test_constraint_violation_list[constraint_type]
            )
            self.test_freq_constraint_violation_list[constraint_type].append(
                test_constraint_violation_list[constraint_type] / step_counter
            )
            if self.debug_ is True:
                print(
                    "{} violations of constraint '{}'\n".format(
                        test_constraint_violation_list[constraint_type],
                        constraint_type
                    )
                )

    def save_agent_models(self, prefix_model_name):
        """
        Save the agent models
        :param prefix_model_name: str, prefix name for the models
        """
        self.agent.save(prefix_model_name, self.file_results_dir)

    @abstractmethod
    def train_store_and_print_info(self):
        raise NotImplementedError

    def train_logging_per_interval(self):
        """
        Calculate and store per_log_interval values
        """
        for key, value in self.train_logs_dict.items():
            if key not in list(self.train_logs_avg_per_log_interval_dict.keys()):
                self.train_logs_avg_per_log_interval_dict[key] = []
            if len(value) > 0:
                self.train_logs_avg_per_log_interval_dict[key].append(np.mean(value[-self.log_interval:]))

        # Print per_log_interval values
        print_latest_metrics_from_dict(self.train_logs_avg_per_log_interval_dict)

    @abstractmethod
    def train_networks(self):
        """
        Train the networks of agent(s)
        :return: training results
        """
        raise NotImplementedError

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, dict]:
        """
        Perform an environment step, normalizing obs and rewards before return.
        :param action: Action to perform
        :return: Next observation, reward, cost, done, truncated, info
        """

        # Step
        next_obs, reward, cost, done, truncated, info = self.env.step(action)

        # Rendering
        if self.render is True:
            vision_obs = self.env.render()
            render(vision_obs)

        return self.normalize_features_func(next_obs), reward, cost, done, truncated, info

    def env_reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset environment

        :return: observation, info

        """
        observation, info = self.env.reset()
        observation = self.normalize_features_func(observation)

        return observation, info

    def normalize_features_func(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the features of the observation.
        :param obs: Observation to normalize
        :return: Normalized observation
        """
        
        if self.normalize_features is True:
            normalized_obs = normalize_features_func(obs, self.env.task_id)

            return normalized_obs
            
        return obs

    def normalize_rewards_func(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize the reward.
        :param reward: Reward to normalize
        :return: Normalized reward
        """
        if self.normalize_rewards is True:
            normalized_reward = normalize_reward_func(reward, self.env.task_id)

            return normalized_reward

        return reward

    def save_info(self, chkpt_dir):
        """
        Saves experiment additional information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        """

        info = {
            'total_games': self.total_games,
            'total_steps': self.total_steps
        }
        write_info_file(info, 'rest_info', chkpt_dir)
