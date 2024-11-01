import sys
sys.path.append('./')

import pandas as pd
import os
from itertools import chain

from plot_utils.plot_utils import get_config
from maze3D_new.utils import goals
from maze3D_new.utils import save_test_logs_and_plot
from maze3D_new.gameObjects import box_size
from maze3D_new.Maze3DEnv import layouts
num_of_boxes_x = len(layouts[0][0])
num_of_boxes_y = len(layouts[0][0][0])


class Experiment(object):

    def __init__(self, data_dir, games_to_keep):
        # Directory in which the data are stored
        self.data_dir = data_dir
        # Keep only the specified games
        self.games_to_keep = games_to_keep

        # Find the required folder inside 'self.data_dir'
        self.dir_list = list(os.walk(os.path.join(self.data_dir, 'tmp')))[0][1]
        assert len(self.dir_list) == 1

        # Read the data and transform them to flat lists (when the column is just one)
        test_actions_csv_path = os.path.join(
            self.data_dir,
            'tmp',
            self.dir_list[0],
            'test_actions.csv'
        )
        self.test_action_history = (
            pd.read_csv(test_actions_csv_path)
        ).values.tolist()
        test_episode_duration_csv_path = os.path.join(
            self.data_dir,
            'tmp',
            self.dir_list[0],
            'test_episode_duration.csv'
        )
        self.test_game_duration_list = list(
            chain.from_iterable((pd.read_csv(test_episode_duration_csv_path)).values.tolist())
        )
        test_length_csv_path = os.path.join(
            self.data_dir,
            'tmp',
            self.dir_list[0],
            'test_length.csv')
        self.test_length_list = list(
            chain.from_iterable((pd.read_csv(test_length_csv_path)).values.tolist())
        )
        distance_travel_test_csv_path = os.path.join(
            self.data_dir,
            'tmp',
            self.dir_list[0],
            'distance_travel_test.csv'
        )
        self.test_distance_travel_list = list(
            chain.from_iterable((pd.read_csv(distance_travel_test_csv_path)).values.tolist())
        )
        pure_rewards_test_csv_path = os.path.join(
            self.data_dir,
            'tmp',
            self.dir_list[0],
            'pure_rewards_test.csv'
        )
        self.test_reward_list = list(
            chain.from_iterable((pd.read_csv(pure_rewards_test_csv_path)).values.tolist())
        )

        # Find the yaml file
        yaml_files_list = [
            elem for elem in os.listdir(os.path.join(self.data_dir, 'tmp', self.dir_list[0])) if elem.endswith('.yaml')
        ]
        assert len(yaml_files_list) == 1
        config_file_name = yaml_files_list[0]
        self.config = get_config(os.path.join(self.data_dir, 'tmp', self.dir_list[0], config_file_name))

        # Get info from config file
        self.constr_ball_only_at_the_right_side_wrt_hole = \
            self.config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']
        self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef = \
            self.config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole_x_coef']
        self.constr_ball_only_at_the_up_side_wrt_hole = \
            self.config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']
        self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef = \
            self.config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole_y_coef']
        self.constr_ball_not_in_circle = \
            self.config['Experiment']['constraint_ball_not_in_circle']
        self.constr_ball_not_in_circle_circle_position = \
            self.config['Experiment']['constraint_ball_not_in_circle_circle_position']
        self.constr_ball_not_in_circle_circle_radius = \
            self.config['Experiment']['constraint_ball_not_in_circle_circle_radius']
        self.goal = self.config['game']['goal']

        # Set the required attributes in dummy classes
        self.env = lambda: None
        setattr(self.env, 'board', lambda: None)
        # Circle constraints
        setattr(
            self.env.board,
            'red_torus',
            [lambda: None for _ in range(len(self.constr_ball_not_in_circle_circle_position))]
        )
        setattr(
            self.env.board,
            'green_torus',
            [lambda: None for _ in range(len(self.constr_ball_not_in_circle_circle_position))]
        )
        for torus_idx in range(len(self.constr_ball_not_in_circle_circle_position)):
            setattr(
                self.env.board.red_torus[torus_idx],
                'x',
                self.constr_ball_not_in_circle_circle_position[torus_idx][0]
            )
            setattr(
                self.env.board.red_torus[torus_idx],
                'y',
                self.constr_ball_not_in_circle_circle_position[torus_idx][1]
            )
            setattr(
                self.env.board.red_torus[torus_idx],
                'radius_outer_circle',
                self.constr_ball_not_in_circle_circle_radius[torus_idx]
            )
            setattr(
                self.env.board.green_torus[torus_idx],
                'x',
                self.constr_ball_not_in_circle_circle_position[torus_idx][0]
            )
            setattr(
                self.env.board.green_torus[torus_idx],
                'y',
                self.constr_ball_not_in_circle_circle_position[torus_idx][1]
            )
            setattr(
                self.env.board.green_torus[torus_idx],
                'radius_outer_circle',
                self.constr_ball_not_in_circle_circle_radius[torus_idx]
            )
        # Horizontal constraint
        setattr(self.env.board, 'horizontal_red_line', lambda: None)
        setattr(
            self.env.board.horizontal_red_line,
            'x',
            box_size * num_of_boxes_y / 3 - num_of_boxes_y * box_size / 2
        )
        setattr(
            self.env.board.horizontal_red_line,
            'y',
            box_size * self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef - num_of_boxes_x * box_size / 2
        )
        setattr(self.env.board, 'horizontal_green_line', lambda: None)
        setattr(
            self.env.board.horizontal_green_line,
            'x',
            box_size * num_of_boxes_y / 3 - num_of_boxes_y * box_size / 2
        )
        setattr(
            self.env.board.horizontal_green_line,
            'y',
            box_size * self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef - num_of_boxes_x * box_size / 2
        )
        # Vertical constraint
        setattr(self.env.board, 'vertical_red_line', lambda: None)
        setattr(
            self.env.board.vertical_red_line,
            'x',
            box_size * self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef - num_of_boxes_y * box_size / 2
        )
        setattr(
            self.env.board.vertical_red_line,
            'y',
            box_size * num_of_boxes_x / 3 - num_of_boxes_x * box_size / 2
        )
        setattr(self.env.board, 'vertical_green_line', lambda: None)
        setattr(
            self.env.board.vertical_green_line,
            'x',
            box_size * self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef - num_of_boxes_y * box_size / 2
        )
        setattr(
            self.env.board.vertical_green_line,
            'y',
            box_size * num_of_boxes_x / 3 - num_of_boxes_x * box_size / 2
        )

        ## Iterate through observations and track the states where any constraint is violated
        indices_to_keep = []
        if (self.constr_ball_only_at_the_up_side_wrt_hole is True or
             self.constr_ball_only_at_the_right_side_wrt_hole is True or
             self.constr_ball_not_in_circle is True):

            # Read the file with observations
            self.df_test = pd.read_pickle(os.path.join(self.data_dir, 'tmp', self.dir_list[0], 'test_logs.pkl'))

            # Initialize variables
            self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list = []
            self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list = []
            self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list = []
            self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list = []
            self.test_ball_not_in_circle_num_constraint_violated_list = []
            self.test_ball_not_in_circle_freq_constraint_violated_list = []
            new_game = True
            ball_only_at_the_right_side_wrt_hole_constraint_violation_list = []
            ball_only_at_the_up_side_wrt_hole_constraint_violation_list = []
            ball_not_in_circle_constraint_violation_list = []
            game_index = -1

            for row_ind, obs in enumerate(self.df_test.iterrows()):
                if new_game:
                    new_game = False
                    ball_only_at_the_right_side_wrt_hole_constraint_violation_list = []
                    ball_only_at_the_up_side_wrt_hole_constraint_violation_list = []
                    ball_not_in_circle_constraint_violation_list = []
                    game_index += 1
                else:
                    ball_only_at_the_right_side_wrt_hole_constraint_violation = 0
                    ball_only_at_the_up_side_wrt_hole_constraint_violation = 0
                    ball_not_in_circle_constraint_violation = 0
                    if (
                            self.constr_ball_only_at_the_up_side_wrt_hole is True and
                            obs[1]['ball_pos_y'] <= self.env.board.horizontal_red_line.y
                    ):
                        ball_only_at_the_up_side_wrt_hole_constraint_violation = 1
                    ball_only_at_the_up_side_wrt_hole_constraint_violation_list.append(
                        ball_only_at_the_up_side_wrt_hole_constraint_violation
                    )
                    if (
                            self.constr_ball_only_at_the_right_side_wrt_hole is True and
                            obs[1]['ball_pos_x'] <= self.env.board.vertical_red_line.x
                    ):
                        ball_only_at_the_right_side_wrt_hole_constraint_violation = 1
                    ball_only_at_the_right_side_wrt_hole_constraint_violation_list.append(
                        ball_only_at_the_right_side_wrt_hole_constraint_violation
                    )
                    if self.constr_ball_not_in_circle is True:
                        for torus_idx in range(len(self.env.board.red_torus)):
                            if (
                                    (
                                            self.constr_ball_not_in_circle_circle_position[torus_idx][0] -
                                            self.constr_ball_not_in_circle_circle_radius[torus_idx] <=
                                            obs[1]['ball_pos_x'] <=
                                            self.constr_ball_not_in_circle_circle_position[torus_idx][0] +
                                            self.constr_ball_not_in_circle_circle_radius[torus_idx]
                                    )
                                    and
                                    (
                                            self.constr_ball_not_in_circle_circle_position[torus_idx][1] -
                                            self.constr_ball_not_in_circle_circle_radius[torus_idx] <=
                                            obs[1]['ball_pos_y'] <=
                                            self.constr_ball_not_in_circle_circle_position[torus_idx][1] +
                                            self.constr_ball_not_in_circle_circle_radius[torus_idx]
                                    )
                            ):
                                ball_not_in_circle_constraint_violation = 1
                                break  # At least one violation of a circle constraint is enough
                    ball_not_in_circle_constraint_violation_list.append(ball_not_in_circle_constraint_violation)

                    if obs[1]['done'] == 1:
                        self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list.append(
                            sum(ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
                        )
                        self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list.append(
                            sum(ball_only_at_the_right_side_wrt_hole_constraint_violation_list) /
                            len(ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
                        )
                        self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list.append(
                            sum(ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
                        )
                        self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list.append(
                            sum(ball_only_at_the_up_side_wrt_hole_constraint_violation_list) /
                            len(ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
                        )
                        self.test_ball_not_in_circle_num_constraint_violated_list.append(
                            sum(ball_not_in_circle_constraint_violation_list)
                        )
                        self.test_ball_not_in_circle_freq_constraint_violated_list.append(
                            sum(ball_not_in_circle_constraint_violation_list) /
                            len(ball_not_in_circle_constraint_violation_list)
                        )
                        new_game = True

                # Store indices to keep the current game if in the specified games
                if games_to_keep != 'all' and game_index in games_to_keep:
                    indices_to_keep.append(row_ind)

            # In case that there are the corresponding files of constraint violations,
            # check if they match with those computed here.
            self.check_constraints_consistency(
                'test_ball_only_at_the_up_side_wrt_hole_num_constraint_violations.csv',
                'test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violations.csv',
                self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list,
                self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list
            )
            self.check_constraints_consistency(
                'test_ball_only_at_the_right_side_wrt_hole_num_constraint_violations.csv',
                'test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violations.csv',
                self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list,
                self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list
            )
            self.check_constraints_consistency(
                'test_ball_not_in_circle_num_constraint_violations.csv',
                'test_ball_not_in_circle_freq_constraint_violations.csv',
                self.test_ball_not_in_circle_num_constraint_violated_list,
                self.test_ball_not_in_circle_freq_constraint_violated_list
            )

        self.test_window_size_moving_avg = self.config['Experiment']['test_window_size_moving_avg']

        # Keep only the info for the specified games
        if keep_games != 'all':
            assert len(indices_to_keep) > 0 and len(keep_games) > 0, \
                'len(indices_to_keep): {}, len(keep_games): {}'.format(len(indices_to_keep), len(keep_games))
            self.test_game_duration_list = [
                elem for elem_id, elem in enumerate(self.test_game_duration_list) if elem_id in keep_games
            ]
            self.test_length_list = [
                elem for elem_id, elem in enumerate(self.test_length_list) if elem_id in keep_games
            ]
            self.test_distance_travel_list = [
                elem for elem_id, elem in enumerate(self.test_distance_travel_list) if elem_id in keep_games
            ]
            self.test_reward_list = [
                elem for elem_id, elem in enumerate(self.test_reward_list) if elem_id in keep_games
            ]
            self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.test_ball_not_in_circle_num_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_not_in_circle_num_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.test_ball_not_in_circle_freq_constraint_violated_list = [
                elem for elem_id, elem in enumerate(
                    self.test_ball_not_in_circle_freq_constraint_violated_list
                ) if elem_id in keep_games
            ]
            self.df_test = self.df_test.iloc[indices_to_keep]

    def check_constraints_consistency(
            self,
            num_constraint_violations_csv_file_name,
            freq_constraint_violations_csv_file_name,
            num_constraint_violated_list,
            freq_constraint_violated_list
    ):

        if (
                os.path.exists(
                    os.path.join(self.data_dir, 'tmp', self.dir_list[0], num_constraint_violations_csv_file_name)
                ) and
                os.path.exists(
                    os.path.join(self.data_dir, 'tmp', self.dir_list[0], freq_constraint_violations_csv_file_name)
                )
        ):

            print(
                'files {} and {} found!'.format(
                    num_constraint_violations_csv_file_name, freq_constraint_violations_csv_file_name
                )
            )

            # Read the files and flat the lists
            num_constraint_violations_csv_file_path = os.path.join(
                self.data_dir,
                'tmp',
                self.dir_list[0],
                num_constraint_violations_csv_file_name
            )
            tmp_test_num_constraint_violated_list = list(
                chain.from_iterable((pd.read_csv(num_constraint_violations_csv_file_path)).values.tolist())
            )
            freq_constraint_violations_csv_file_path = os.path.join(
                self.data_dir,
                'tmp',
                self.dir_list[0],
                freq_constraint_violations_csv_file_name
            )
            tmp_test_freq_constraint_violated_list = list(
                chain.from_iterable((pd.read_csv(freq_constraint_violations_csv_file_path)).values.tolist())
            )

            if num_constraint_violated_list != tmp_test_num_constraint_violated_list:
                print(
                    '\n\n#######################################################################################\n'
                    '#####################################ALERT#############################################\n'
                    'num_constraint_violated_list={} while tmp_test_num_constraint_violated_list={}\n\n'.format(
                        num_constraint_violated_list,
                        tmp_test_num_constraint_violated_list
                    )
                )
            if freq_constraint_violated_list != tmp_test_freq_constraint_violated_list:
                print(
                    '\n\n#######################################################################################\n'
                    '#####################################ALERT#############################################\n'
                    'freq_constraint_violated_list={} \nwhile \ntmp_test_freq_constraint_violated_list={}\n\n'.format(
                        freq_constraint_violated_list,
                        tmp_test_freq_constraint_violated_list
                    )
                )


if __name__ == '__main__':

    stored_results_dir = sys.argv[1]  # e.g., '/home/georgepap/PycharmProjects/HAI-MAZE_master/HAI-MAZE/tmp_results_1'
    keep_games = sys.argv[2]  # 'all' or list of games (e.g., [0, 32])
    if keep_games != 'all':
        keep_games = eval(sys.argv[2])
    experiment = Experiment(stored_results_dir, keep_games)

    results_dir = os.path.join('./', 'tmp_results')
    os.mkdir(results_dir)
    tmp_results_dir = os.path.join(results_dir, 'tmp')
    os.mkdir(tmp_results_dir)
    plot_results_dir = os.path.join(tmp_results_dir, 'plot')
    os.mkdir(plot_results_dir)
    save_test_logs_and_plot(experiment, tmp_results_dir, plot_results_dir)

    print("\n##################################\nTest results extraction completed!\n##################################")
