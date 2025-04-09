from statistics import mean
import csv

import numpy as np

from SCOPIL.utils.env_utils import get_cost_sum_from_info_dict, render
from .base_experiment import BaseExperiment
from SCOPIL.utils.demonstration_utils import min_max_obs_values, min_max_rew_values
from SCOPIL.utils.exp_utils import test_print_logs, save_demonstrations
from SCOPIL.utils.file_utils import write_info_file


class TestExperiment(BaseExperiment):
    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):

        super().__init__(environment, agent, config, file_results_dir, seed)

        # Define the type of algorithm
        if 'SAC' in list(config.items())[1]:
            self.algo = 'SAC'
        elif 'ICRL' in list(config.items())[1]:
            self.algo = 'ICRL'
        else:
            raise ValueError("There is no valid algorithm provided!")

        # Retrieve information from the config file for testing
        self.normalize_features = config['Experiment']['normalize_features']
        self.normalize_rewards = config['Experiment']['normalize_rewards']
        self.render = config['Experiment']['render']

        # Load the trained agent
        self.load_agent_models()

    def agent_test(self):

        # Initialize test variables
        test_game_i = 0

        while test_game_i < self.test_max_games:

            print('Test game: ' + str(test_game_i) + '\n')

            # Initialize test game variables
            test_step_counter = 0
            test_game_reward = 0
            test_game_num_constraint_violation = {constraint_type: 0 for constraint_type in self.constraint_types}
            test_done = False
            test_fixed_done = 0.
            for key in self.test_history:
                self.test_history[key].update({f'episode_{test_game_i}': {}})

            # Reset environment. The cost is always 0 after resetting.
            test_obs, test_info = self.env.reset()
            assert len(test_info) == 0, f"'test_info': {test_info}"
            test_vision_obs = self.env.render()

            # Start to play a game.
            while not test_done:

                # Render the environment
                if self.render is True:
                    render(test_vision_obs)

                # Get the agent action
                test_action = self.get_agent_action(test_obs)

                # Apply an environment step and get the results.
                test_next_obs, test_reward, test_cost, test_done, test_truncated, test_info = \
                    self.env.step(test_action)
                test_next_vision_obs = self.env.render()

                # Print a message for the user if at least a constraint is violated
                if self.debug_ is True and test_cost > 0:
                    print('\n#########################')
                    print('Constraint violation!')
                    print('#########################\n')

                # Keep track of whether the game ended due to success or due to hitting the time horizon.
                test_fixed_done = float(test_done)
                if test_step_counter == self.test_max_timesteps_per_game - 1:
                    test_fixed_done = 0.
                    test_done = True

                # Store the transition data
                if self.debug_ is True:
                    self.test_history['actions'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_action.copy()
                    })
                    self.test_history['vector_obs'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_obs.copy()
                    })
                    self.test_history['reward'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_reward
                    })
                    self.test_history['cost'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_cost
                    })
                    self.test_history['done'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_done
                    })
                    self.test_history['fixed_done'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_fixed_done
                    })
                    self.test_history['truncated'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_truncated
                    })
                    self.test_history['info'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_info.copy()
                    })
                self.test_history['vision_obs'][f'episode_{test_game_i}'].update({
                    f'step_{test_step_counter}': test_vision_obs.copy()
                })

                # Update the total reward
                test_game_reward += self.normalize_rewards_func(test_reward)

                # Update the number of constraint violations
                test_game_num_constraint_violation.update({
                    constraint_type: test_game_num_constraint_violation[constraint_type] +
                                     (
                                         test_info[constraint_type] if constraint_type != 'cost_sum'
                                                                    else
                                         get_cost_sum_from_info_dict(test_info)
                                     )
                    for constraint_type in self.constraint_types
                })

                # Set next as the current
                test_obs = test_next_obs.copy()
                test_vision_obs = test_next_vision_obs.copy()

                # Update the step number of the game and the total steps of all games
                test_step_counter += 1
                self.total_steps += 1

                if self.debug_ is True and test_done is True:
                    # Store the last observations
                    self.test_history['vector_obs'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_next_obs.copy()
                    })
                self.test_history['vision_obs'][f'episode_{test_game_i}'].update({
                    f'step_{test_step_counter}': test_next_vision_obs.copy()
                })

            ## End of test game

            # Save the demonstrations to a pickle file and delete them to save RAM
            save_demonstrations(self.test_history, self.file_results_dir, save_pickle_file=False, delete_episode=True)
            # Update the testing metrics
            self.update_test_metrics(
                test_game_reward,
                test_step_counter,
                test_game_num_constraint_violation
            )
            # Increase test game counter
            test_game_i += 1

        # Terminate the environment
        self.env.close()

        # Print the average results of the test
        test_print_logs(
            mean(self.test_reward_list),
            mean(self.test_step_list),
            {
                constraint_type: mean(self.test_num_constraint_violation_list[constraint_type])
                for constraint_type in self.constraint_types
            },
            {
                constraint_type: mean(self.test_freq_constraint_violation_list[constraint_type])
                for constraint_type in self.constraint_types
            }
        )

    def update_test_metrics(
            self,
            game_reward,
            game_step_counter,
            game_num_constraint_violation
    ):

        self.test_reward_list.append(game_reward)
        self.test_step_list.append(game_step_counter)
        for constraint_type in self.constraint_types:
            self.test_num_constraint_violation_list[constraint_type].append(
                game_num_constraint_violation[constraint_type]
            )
            self.test_freq_constraint_violation_list[constraint_type].append(
                game_num_constraint_violation[constraint_type] / game_step_counter
            )

    def load_agent_models(self):
        print()  # Just for letting an empty line before the message of loading
        self.agent.load(prefix=self.checkpoint_prefix, path=self.checkpoint_path)
        print()  # Just for letting an empty line after the message of loading

    def normalize_features_func(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the features of the observation.
        :param obs: Observation to normalize
        :return: Normalized observation
        """

        if self.normalize_features is True:
            max_obs, min_obs = min_max_obs_values(self.env.task_id)
            normalized_obs = (obs - min_obs) / (max_obs - min_obs + 1e-8)  # avoid division by zero
            return normalized_obs

        return obs

    def normalize_rewards_func(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize the reward.
        :param reward: Reward to normalize
        :return: Normalized reward
        """
        if self.normalize_rewards is True:
            max_reward, min_reward = min_max_rew_values(self.env.task_id)
            normalized_reward = (reward - min_reward) / (max_reward - min_reward + 1e-8)  # avoid division by zero
            return normalized_reward

        return reward

    def get_agent_action(self, obs):
        """
        Get the agent action
        :param obs: np.array, observation of the agent
        :return: np.array, action
        """
        return self.agent.predict(self.normalize_features_func(obs), deterministic=True)

    def save_info(self, chkpt_dir):
        """
        Saves experiment additional information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        """

        info = {
            'total_games': self.test_max_games,
            'total_steps': self.total_steps
        }
        write_info_file(info, 'rest_info', chkpt_dir)
