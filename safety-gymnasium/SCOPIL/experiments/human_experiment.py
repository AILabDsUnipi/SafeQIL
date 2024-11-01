from statistics import mean
import csv

from SCOPIL.utils.env_utils import get_cost_sum_from_info_dict, close_env_debug_mode
from SCOPIL.utils.exp_utils import test_print_logs, save_demonstrations
from .base_experiment import BaseExperiment


class HumanExperiment(BaseExperiment):
    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):

        super().__init__(environment, agent, config, file_results_dir, seed)

        # Check config
        assert 'NO_ALGO' in list(config.items())[1], "'NO_ALGO' is not listed in the config file!"

        # Freeze motion
        self.freeze_motion = self.config['Experiment']['freeze_motion']
        assert not self.freeze_motion or (self.freeze_motion and self.algo is None)

        print('\nPlay the game without an algorithm !')

    def human_alone_test(self):

        # Import here since it is only necessary for this function
        import mujoco
        import glfw
        from SCOPIL.utils.env_utils import capture_img_from_env_rendering

        # Initialize test variables
        replay_game = False
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

            ## Prepare the environment rendering
            # Change camera view to agent view
            self.env.render()
            self.env.task.viewer.cam.fixedcamid = 3
            self.env.task.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            # Hide menu from rendering
            self.env.task.viewer._hide_menu = True
            self.env.render()

            # Get the first image observations and actions after resetting.
            test_vision_obs = capture_img_from_env_rendering(self.env)
            test_actions = self.env.task.agent.engine.data.ctrl

            # Start to play a game.
            while not test_done:

                # Apply an environment step and get the results.
                # No actions needed since the agent is controlled through the keyboard.
                # NOTE: When 'test_truncated' is True, 'test_done' is False.
                test_next_obs, test_reward, test_cost, test_done, test_truncated, test_info = \
                    self.env.step([None, None])

                # Print a message for the user if at least a constraint is violated
                if test_cost > 0:
                    print('\n#########################')
                    print('Constraint violation!')
                    print('#########################\n')

                # Keep track of whether the game ended due to success or due to hitting the time horizon.
                test_fixed_done = float(test_done)
                if test_step_counter == self.test_max_timesteps_per_game - 1:
                    test_fixed_done = 0.
                    test_done = True

                # Store the transition data
                self.test_history['actions'][f'episode_{test_game_i}'].update({
                    f'step_{test_step_counter}': test_actions.copy()
                })
                self.test_history['vector_obs'][f'episode_{test_game_i}'].update({
                    f'step_{test_step_counter}': test_obs.copy()
                })
                self.test_history['vision_obs'][f'episode_{test_game_i}'].update({
                    f'step_{test_step_counter}': test_vision_obs.copy()
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

                # Update the total reward
                test_game_reward += test_reward

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

                # Get the next image observation
                self.env.render()
                test_next_vision_obs = capture_img_from_env_rendering(self.env)

                # Get the next action
                if self.freeze_motion:
                    print('Waiting for you action ...')
                    glfw.wait_events()
                test_next_actions = self.env.task.agent.engine.data.ctrl
                print('actions: ', test_actions)

                # Set next as the current
                test_obs = test_next_obs.copy()
                test_vision_obs = test_next_vision_obs.copy()
                test_actions = test_next_actions.copy()

                # Update the step number of the game and the total steps of all games
                test_step_counter += 1
                self.total_steps += 1

                if test_done is True:
                    # Store the last observations and actions
                    self.test_history['actions'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_next_actions.copy()
                    })
                    self.test_history['vector_obs'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_next_obs.copy()
                    })
                    self.test_history['vision_obs'][f'episode_{test_game_i}'].update({
                        f'step_{test_step_counter}': test_next_vision_obs.copy()
                    })

            ## End of test game

            if self.freeze_motion:
                print(f"Total cost is: {test_game_num_constraint_violation['cost_sum']}. Replay game?")
                answer = input('Y/N: ')
                print()  # Just for leaving an empty line
                while answer != 'Y' and answer != 'N':
                    answer = input('Invalid answer! Please answer with Y or N: ')
                replay_game = False if answer == 'N' else (True if answer == 'Y' else None)
                if replay_game:
                    # Delete the recorded data of the last episode
                    for key in self.test_history.keys():
                        del self.test_history[key][f'episode_{test_game_i}']

            if not replay_game:
                # Save the demonstrations to a pickle file and delete them to save RAM
                save_demonstrations(self.test_history, self.file_results_dir, delete_episode=True)
                # Update the testing metrics
                self.update_test_metrics(
                    test_game_reward,
                    test_step_counter,
                    test_game_num_constraint_violation
                )
                # Increase test game counter
                test_game_i += 1

        # Terminate the environment
        close_env_debug_mode(self.env)

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

    def save_info(self, chkpt_dir):
        """
        Saves experiment additional information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        """

        info = {
            'total_games': self.test_max_games,
            'total_steps': self.total_steps
        }

        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

