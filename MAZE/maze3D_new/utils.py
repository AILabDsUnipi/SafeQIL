import math
import os

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy.spatial import distance
from statistics import mean
import seaborn as sns

from maze3D_new.extra_config import left_down, right_down, left_up

goals = {"left_down": left_down, "left_up": left_up, "right_down": right_down}

ball_diameter = 43.615993

def normalize_features(data):
    """
    min-max normalization in the range [-1, 1]
    """

    extra_data_flag = False
    if not data.shape[-1] == 8:
        extra_data_flag = True
        if len(data.shape) == 1:
            extra_data = data[8:]
            data = data[:8]
        elif len(data.shape) == 2:
            extra_data = data[:, 8:]
            data = data[:, :8]
        else:
            raise NotImplementedError

    max_values = np.array([161.79, 169.34, 6.22, 5.0, 0.5, 0.5, 0.01, 1], dtype=np.float64)
    min_values = np.array([-174.46, -174.46, -6.7, -6.01, -0.5, -0.5, -0.01, -1], dtype=np.float64)

    norm_data = 2. * ((data - min_values) / (max_values - min_values)) - 1.

    if extra_data_flag is True:
        norm_data = np.concatenate([norm_data, extra_data], axis=0 if len(data.shape) == 1 else 1)

    return norm_data

def checkTerminal(ball, goal):
    goal = goals[goal]
    if distance.euclidean([ball.x, ball.y], goal) < (ball_diameter/3):
        return True
    return False

def get_distance_from_goal(ball, goal):
    goal = goals[goal]
    return math.sqrt(math.pow(ball.x - goal[0], 2) + math.pow(ball.y - goal[1], 2))

def convert_actions(actions):
    # gets a list of 4 elements (i.e., actions contains 4 elements). it is called from getKeyboard()
    action = []
    if actions[0] == 1:
        action.append(1)
    elif actions[1] == 1:
        action.append(2)
    elif actions[0] is None and actions[1] is None:
        action.append(None)
    else:
        action.append(0)
    if actions[2] == 1:
        action.append(1)
    elif actions[3] == 1:
        action.append(2)
    elif actions[2] is None and actions[3] is None:
        action.append(None)
    else:
        action.append(0)
    return action

def save_test_logs_and_plot(experiment, chkpt_dir, plot_dir, return_data_for_plots=False):

    # Save test logs in files
    pd.DataFrame(experiment.test_action_history, columns=['X', 'Y']).to_csv(
        os.path.join(chkpt_dir, 'test_actions.csv'), index=False
    )
    pd.DataFrame(experiment.test_game_duration_list, columns=['Duration']).to_csv(
        os.path.join(chkpt_dir, 'test_episode_duration.csv'), index=False
    )
    pd.DataFrame(experiment.test_length_list, columns=['Length']).to_csv(
        os.path.join(chkpt_dir, 'test_length.csv'), index=False
    )
    pd.DataFrame(experiment.test_distance_travel_list, columns=['Distance']).to_csv(
        os.path.join(chkpt_dir, 'distance_travel_test.csv'), index=False
    )
    pd.DataFrame(experiment.test_reward_list, columns=['Reward']).to_csv(
        os.path.join(chkpt_dir, 'pure_rewards_test.csv'), index=False
    )
    if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
        pd.DataFrame(
            experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list,
            columns=['Number_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_only_at_the_right_side_wrt_hole_num_constraint_violations.csv'),
            index=False
        )
        pd.DataFrame(
            experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list,
            columns=['Frequency_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violations.csv'),
            index=False
        )
    if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
        pd.DataFrame(
            experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list,
            columns=['Number_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_only_at_the_up_side_wrt_hole_num_constraint_violations.csv'),
            index=False
        )
        pd.DataFrame(
            experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list,
            columns=['Frequency_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violations.csv'),
            index=False
        )
    if experiment.constr_ball_not_in_circle is True:
        pd.DataFrame(
            experiment.test_ball_not_in_circle_num_constraint_violated_list,
            columns=['Number_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_not_in_circle_num_constraint_violations.csv'),
            index=False
        )
        pd.DataFrame(
            experiment.test_ball_not_in_circle_freq_constraint_violated_list,
            columns=['Frequency_of_Constraint_Violations']
        ).to_csv(
            os.path.join(chkpt_dir, 'test_ball_not_in_circle_freq_constraint_violations.csv'),
            index=False
        )

    # Check the consistency of the test results
    assert len(experiment.test_reward_list) > 0, "No test results provided."
    assert len(experiment.test_reward_list) == \
           len(experiment.test_game_duration_list) == \
           len(experiment.test_length_list) == \
           len(experiment.test_distance_travel_list), "Inconsistency among results concerning the number of test."
    if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
        assert len(experiment.test_reward_list) == \
               len(experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list) == \
               len(experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list), \
               "'ball_only_at_the_right_side_wrt_hole' constraints results are inconsistent concerning the number of test."
    if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
        assert len(experiment.test_reward_list) == \
               len(experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list) == \
               len(experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list), \
               "'ball_only_at_the_up_side_wrt_hole' constraints results are inconsistent concerning the number of test."
    if experiment.constr_ball_not_in_circle is True:
        assert len(experiment.test_reward_list) == \
               len(experiment.test_ball_not_in_circle_num_constraint_violated_list) == \
               len(experiment.test_ball_not_in_circle_freq_constraint_violated_list), \
               "'ball_not_in_circle' constraints results are inconsistent concerning the number of test."

    single_test = len(experiment.test_reward_list) == 1

    # Write results statistics in txt file
    with open(os.path.join(chkpt_dir, 'stats_info.txt'), 'w') as stats_info:
        # First element of each tuple is the data and the second is the type of data
        data_to_write = [
            (experiment.test_reward_list, 'Reward'),
            (experiment.test_game_duration_list, 'Duration'),
            (experiment.test_length_list, 'Length'),
            (experiment.test_distance_travel_list, 'Distance travel')
        ]
        if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
            data_to_write += [
                (experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list,
                 "Number of 'ball_only_at_the_right_side_wrt_hole' constraint violations"),
                (experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list,
                 "Frequency of 'ball_only_at_the_right_side_wrt_hole' constraint violations")
            ]
        if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
            data_to_write += [
                (experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list,
                 "Number of 'ball_only_at_the_up_side_wrt_hole' constraint violations"),
                (experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list,
                 "Frequency of 'ball_only_at_the_up_side_wrt_hole' constraint violations")
            ]
        if experiment.constr_ball_not_in_circle is True:
            data_to_write += [
                (experiment.test_ball_not_in_circle_num_constraint_violated_list,
                 "Number of 'ball_not_in_circle' constraint violations"),
                (experiment.test_ball_not_in_circle_freq_constraint_violated_list,
                 "Frequency of 'ball_not_in_circle' constraint violations")
            ]
        for (data, type_of_data) in data_to_write:
            stats_info.write(
                '###################\n' +
                type_of_data + ':\n' +
                '  mean: ' + str(np.mean(data)) + '\n' +
                '  std: ' + ("-" if single_test else str(np.std(data, ddof=1))) + '\n' +
                '  median: ' + ("-" if single_test else str(np.median(data))) + '\n' +
                '  Q1: ' + ("-" if single_test else str(np.quantile(data, 0.25))) + '\n' +
                '  Q3: ' + ("-" if single_test else str(np.quantile(data, 0.5))) + '\n\n'
            )

    ### Plot metrics
    data_for_plots_to_return = {}
    use_sliding_window = len(experiment.test_length_list) > experiment.test_window_size_moving_avg
    if not single_test:

        # The first element of each tuple is the ylabel, the second is the plot title,
        # the third is the data to plot, and the forth is the title of the file to be written.
        data_to_plot = [
            ('Length', 'Game length (steps)', experiment.test_length_list, 'length'),
            ('Distance', 'Game travel distance', experiment.test_distance_travel_list, 'distance_travel'),
            ('Duration', 'Game duration', experiment.test_game_duration_list, 'game_duration'),
            ('Reward', 'Game reward', experiment.test_reward_list, 'reward')
        ]
        if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
            data_to_plot += [
                ('Number of Violations',
                 "Number of 'ball_only_at_the_right_side_wrt_hole' constraint violations",
                 experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list,
                 'ball_only_at_the_right_side_wrt_hole_num_constraint_violations'),
                ('Frequency of Violations',
                 "Frequency of 'ball_only_at_the_right_side_wrt_hole' constraint violations",
                 experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list,
                 'ball_only_at_the_right_side_wrt_hole_freq_constraint_violations')
            ]
        if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
            data_to_plot += [
                ('Number of Violations',
                 "Number of 'ball_only_at_the_up_side_wrt_hole' constraint violations",
                 experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list,
                 'ball_only_at_the_up_side_wrt_hole_num_constraint_violations'),
                ('Frequency of Violations',
                 "Frequency of 'ball_only_at_the_up_side_wrt_hole' constraint violations",
                 experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list,
                 'ball_only_at_the_up_side_wrt_hole_freq_constraint_violations')
            ]
        if experiment.constr_ball_not_in_circle is True:
            data_to_plot += [
                ('Number of Violations',
                 "Number of 'ball_not_in_circle' constraint violations",
                 experiment.test_ball_not_in_circle_num_constraint_violated_list,
                 'ball_not_in_circle_num_constraint_violations'),
                ('Frequency of Violations',
                 "Frequency of 'ball_not_in_circle' constraint violations",
                 experiment.test_ball_not_in_circle_freq_constraint_violated_list,
                 'ball_not_in_circle_freq_constraint_violations')
            ]

        plot_xlabel = 'Episodes'
        plot_legend = ('Test',)
        plot_legend_loc = "upper left"
        for (plot_ylabel, plot_title, data, file_title) in data_to_plot:

            # Simple plot
            plt.figure()
            plt.xlabel(plot_xlabel)
            plt.ylabel(plot_ylabel)
            plt.title(plot_title)
            plt.plot([i + 1 for i in range(len(data))], data)
            plt.gca().legend(plot_legend, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + "_test.png"))
            plt.close()
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = [
                    plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, data
                ]

            if use_sliding_window:
                # Sliding window
                mean_sliding_window, max_sliding_window, min_sliding_window = \
                    calculate_sliding_window(data, experiment.test_window_size_moving_avg)
                plt.figure()
                plt.xlabel(plot_xlabel)
                plt.ylabel(plot_ylabel)
                plt.title(plot_title)
                plt.fill_between(range(len(mean_sliding_window)), min_sliding_window, max_sliding_window, alpha=0.5)
                plt.plot(range(len(mean_sliding_window)), mean_sliding_window)
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + "_test_sliding_window.png"))
                plt.close()

            # Boxplot
            create_boxplot(
                np.array(data),
                plot_title_=plot_title,
                ylabel_=plot_ylabel,
                path_name=os.path.join(plot_dir, file_title + "_test_boxplot.png")
            )

    if experiment.constr_ball_only_at_the_right_side_wrt_hole is True or \
       experiment.constr_ball_only_at_the_up_side_wrt_hole is True or \
       experiment.constr_ball_not_in_circle is True:
        ## Trajectories plot

        # Define plot parameters
        xlim_max = 270
        xlim_min = -270
        ylim_max = 270
        ylim_min = -270
        board_xlim_min = -175.0
        board_xlim_max = 175.0
        board_ylim_min = -175.0
        board_ylim_max = 175.0
        board_width = board_xlim_max - board_xlim_min
        board_height = board_ylim_max - board_ylim_min
        wall_width = ball_diameter
        wall_height = ball_diameter
        box_size = ball_diameter

        # Initialize plot
        plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        # Plot walls
        from maze3D_new.Maze3DEnv import layouts
        np_layout = np.array(layouts[0][0])
        right_np_layout = np.rot90(np_layout)
        num_of_boxes_x = right_np_layout.shape[0]
        num_of_boxes_y = right_np_layout.shape[1]

        for row in range(num_of_boxes_x):
            for col in range(num_of_boxes_y):

                # Square wall
                if right_np_layout[row][col] == 1 or \
                   right_np_layout[row][col] == 6:

                    square_wall = patches.Rectangle(
                        (box_size * (num_of_boxes_x - 1 - row) - num_of_boxes_x * box_size / 2 - (ball_diameter/2),
                         box_size * col - num_of_boxes_y * box_size / 2 - (ball_diameter/2)),
                        wall_width + ball_diameter, wall_height + ball_diameter, color='lightgray')

                    square_wall_regular = patches.Rectangle(
                        (box_size * (num_of_boxes_x - 1 - row) - num_of_boxes_x * box_size / 2,
                         box_size * col - num_of_boxes_y * box_size / 2),
                        wall_width, wall_height, color='gray', zorder=2)

                    plt.gca().add_patch(square_wall)
                    plt.gca().add_patch(square_wall_regular)

                # Upper triangle wall
                if right_np_layout[row][col] == 4:
                    upper_triangle_points = np.array([
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + ball_diameter/2 - 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + ball_diameter/2 - 0.5],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + ball_diameter/2 + wall_width - 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + ball_diameter/2 - 0.5],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + ball_diameter/2 - 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + ball_diameter/2 + wall_height - 0.5]
                                                    ])
                    upper_triangle_wall = patches.Polygon(upper_triangle_points, color='lightgray')

                    upper_triangle_points_regular = np.array([
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2),
                         (box_size * col) - (num_of_boxes_y * box_size / 2)],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + wall_width,
                         (box_size * col) - (num_of_boxes_y * box_size / 2)],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2),
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + wall_height]
                                                            ])
                    upper_triangle_regular_wall = patches.Polygon(upper_triangle_points_regular,  color='gray', zorder=2)

                    plt.gca().add_patch(upper_triangle_wall)
                    plt.gca().add_patch(upper_triangle_regular_wall)

                # Lower triangle wall
                if right_np_layout[row][col] == 5:
                    lower_triangle_points = np.array([
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) - ball_diameter / 2 + wall_width + 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) - ball_diameter / 2 + wall_height + 0.5],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) - ball_diameter / 2 + wall_width + 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) - ball_diameter / 2 + 0.5],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) - ball_diameter / 2 + 0.5,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) - ball_diameter / 2 + wall_height + 0.5]
                                                        ])
                    lower_triangle_wall = patches.Polygon(lower_triangle_points, color='lightgray')

                    lower_triangle_points_regular = np.array([
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + wall_width,
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + wall_height],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2) + wall_width,
                         (box_size * col) - (num_of_boxes_y * box_size / 2)],
                        [(box_size * (num_of_boxes_x - 1 - row)) - (num_of_boxes_x * box_size / 2),
                         (box_size * col) - (num_of_boxes_y * box_size / 2) + wall_height]
                                                                ])
                    lower_triangle_regular_wall = patches.Polygon(lower_triangle_points_regular, color='gray', zorder=2)

                    plt.gca().add_patch(lower_triangle_wall)
                    plt.gca().add_patch(lower_triangle_regular_wall)

        # Plot the hole and the accepted radius (based on the diameter of the ball)
        hole = plt.Circle(
            (goals[experiment.goal][0], goals[experiment.goal][1]),
            radius=ball_diameter/3,
            color='lightskyblue'
        )
        plt.gca().add_patch(hole)

        # Plot the constraint line
        if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
            assert experiment.env.board.horizontal_red_line.y == experiment.env.board.horizontal_green_line.y, \
                "y coordinates of horizontal green line and horizontal red line do not match!"
            plt.axhline(
                y=experiment.env.board.horizontal_red_line.y,
                xmin=abs(xlim_min - board_xlim_min) / abs(xlim_min - xlim_max) + 0.005,
                xmax=abs(xlim_min - board_xlim_max) / abs(xlim_min - xlim_max),
                color='black',
                linewidth=3,
                zorder=1
            )
        if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
            assert experiment.env.board.vertical_red_line.x == experiment.env.board.vertical_green_line.x, \
                "x coordinates of vertical green line and vertical red line do not match!"
            plt.axvline(
                x=experiment.env.board.vertical_red_line.x,
                ymin=abs(ylim_min - board_ylim_min) / abs(ylim_min - ylim_max) + 0.005,
                ymax=abs(ylim_min - board_ylim_max) / abs(ylim_min - ylim_max),
                color='black',
                linewidth=3,
                zorder=1
            )
        if experiment.constr_ball_not_in_circle is True:
            assert len(experiment.env.board.red_torus) == len(experiment.env.board.green_torus), \
                "The number of green torus is other than red torus!"
            for torus_idx in range(len(experiment.env.board.green_torus)):
                assert experiment.env.board.red_torus[torus_idx].x == experiment.env.board.green_torus[torus_idx].x, \
                    f"x coordinates of red and green torus {torus_idx} do not match!"
                assert experiment.env.board.red_torus[torus_idx].y == experiment.env.board.green_torus[torus_idx].y, \
                    f"y coordinates of red and green torus {torus_idx} do not match!"
                circle_constr = plt.Circle(
                    (experiment.env.board.red_torus[torus_idx].x, experiment.env.board.red_torus[torus_idx].y),
                    radius=experiment.env.board.red_torus[torus_idx].radius_outer_circle,
                    color='black',
                    fill=False,
                    linewidth=3,
                    zorder=1
                )
                plt.gca().add_patch(circle_constr)

        #Initialize lists
        ball_pos_x = []
        ball_pos_y = []
        constraint_violated = []

        for obs in experiment.df_test.iterrows():
            ball_pos_x.append(obs[1]['ball_pos_x'])
            ball_pos_y.append(obs[1]['ball_pos_y'])

            # Find in which states the constraint is violated
            constraint_violation = 0
            if experiment.constr_ball_only_at_the_up_side_wrt_hole is True and \
               obs[1]['ball_pos_y'] <= experiment.env.board.horizontal_red_line.y:
                constraint_violation = 1
            elif experiment.constr_ball_only_at_the_right_side_wrt_hole is True and \
                 obs[1]['ball_pos_x'] <= experiment.env.board.vertical_red_line.x:
                constraint_violation = 1
            else:
                for torus_idx in range(len(experiment.env.board.green_torus)):
                    if (
                            experiment.constr_ball_not_in_circle is True and
                            (
                                    (
                                            experiment.env.board.red_torus[torus_idx].x -
                                            experiment.env.board.red_torus[torus_idx].radius_outer_circle <=
                                            obs[1]['ball_pos_x'] <=
                                            experiment.env.board.red_torus[torus_idx].x +
                                            experiment.env.board.red_torus[torus_idx].radius_outer_circle
                                    ) and
                                    (
                                            experiment.env.board.red_torus[torus_idx].y -
                                            experiment.env.board.red_torus[torus_idx].radius_outer_circle <=
                                            obs[1]['ball_pos_y'] <=
                                            experiment.env.board.red_torus[torus_idx].y +
                                            experiment.env.board.red_torus[torus_idx].radius_outer_circle
                                    )
                            )
                    ):
                        constraint_violation = 1
                        break # at least one violation of circle constraint is enough
            constraint_violated.append(constraint_violation)

            # When done=1 then the game has ended.
            if obs[1]['done'] == 1:

                ## Separate the states in which the constraint is violated from those that it does not.
                # First create a dataframe with all cases
                df_pos = pd.DataFrame(
                    np.array([ball_pos_x, ball_pos_y, constraint_violated]).T,
                    columns=['x', 'y', 'viol']
                )
                # No violation
                no_violated_ball_pos_x = df_pos[df_pos['viol'] == 0]['x'].values.tolist()
                no_violated_ball_pos_y = df_pos[df_pos['viol'] == 0]['y'].values.tolist()
                # Violation
                violated_ball_pos_x = df_pos[df_pos['viol'] == 1]['x'].values.tolist()
                violated_ball_pos_y = df_pos[df_pos['viol'] == 1]['y'].values.tolist()
                assert len(no_violated_ball_pos_x) == len(no_violated_ball_pos_y)
                assert len(violated_ball_pos_x) == len(violated_ball_pos_y)

                # Plot a black line that connects all points (point = ball position)
                plt.plot(
                    ball_pos_x,
                    ball_pos_y,
                    marker="None",
                    color='black',
                    zorder=2
                )
                # Plot all points where the constraint is not violated as green stars
                plt.plot(
                    no_violated_ball_pos_x,
                    no_violated_ball_pos_y,
                    marker=".",
                    color='green',
                    linestyle='None',
                    zorder=2
                )
                # Plot all points where there the constraint is not violated as red stars
                plt.plot(
                    violated_ball_pos_x,
                    violated_ball_pos_y,
                    marker=".",
                    color='red',
                    linestyle='None',
                    zorder=2
                )

                # Plot arrows to show the direction of the trajectory
                u = np.diff(ball_pos_x)
                v = np.diff(ball_pos_y)
                pos_x = ball_pos_x[:-1] + u / 2
                pos_y = ball_pos_y[:-1] + v / 2
                norm = np.sqrt(u ** 2 + v ** 2) + 1e-6
                plt.quiver(
                    pos_x,
                    pos_y,
                    u / norm,
                    v / norm,
                    angles="xy",
                    zorder=2,
                    pivot="mid",
                    scale_units='xy',
                    scale=0.09
                )

                plt.xlabel('x')
                plt.ylabel('y', rotation=0)
                plt.ylim(ylim_min, ylim_max)
                plt.xlim(xlim_min, xlim_max)
                plt.title('Ball trajectories')

                # Reinitialize lists since the game has ended
                ball_pos_x = []
                ball_pos_y = []
                constraint_violated = []

        plt.savefig(os.path.join(plot_dir, "ball_trajectories.png"))
        plt.close()

    return data_for_plots_to_return


def calculate_sliding_window(data, window_size=10):

    mean_sliding_window = []
    max_sliding_window = []
    min_sliding_window = []
    for i in range(len(data) - window_size + 1):
        mean_sliding_window.append(mean(data[i: i + window_size]))
        max_sliding_window.append(max(data[i: i + window_size]))
        min_sliding_window.append(min(data[i: i + window_size]))

    return mean_sliding_window, max_sliding_window, min_sliding_window


def create_boxplot(
        data,
        mean_markers_size_=5,
        linewidth_=1.5,
        outlier_markers_size_=5,
        ax_=None,
        plot_title_="",
        ylabel_="",
        ylabelpad_=7,
        path_name="boxplot.png"
):

    plt.figure()
    ax = sns.boxplot(
        data=data,
        showmeans=True,
        meanprops={'marker': 'o', 'markeredgecolor': 'c', 'markerfacecolor': 'c', 'markersize': mean_markers_size_},
        boxprops={'edgecolor': 'black', "linewidth": linewidth_},
        whiskerprops={'color': 'black', "linewidth": linewidth_},
        capprops={'color': 'black', "linewidth": linewidth_},
        medianprops={"color": "r", "linewidth": linewidth_},
        flierprops={'markersize': outlier_markers_size_},
        ax=ax_
    )

    ax.set(title=plot_title_)
    ax.set_ylabel(ylabel_, labelpad=ylabelpad_)
    ax.set(xticklabels=[])  # remove the tick labels of x-axis
    plt.savefig(path_name)
    plt.clf()
    plt.close()

def save_df_to_csv(data=None, cols=None, csv_name=None, index=False, list_data=None, chkpt_dir=None, avg=True):
    if list_data is None:
        if avg:
            if isinstance(data[0], list):
                data = [sum(game_data_list) / len(game_data_list) for game_data_list in data]
            elif isinstance(data[0], np.ndarray) and len(data[0].shape) == 1 and data[0].shape[0] > 1:
                data = [np.mean(np_array) for np_array in data]
        else:
            data = [elem for episode_elems in data for elem in episode_elems]

        pd.DataFrame(data, columns=cols).to_csv(chkpt_dir + '/' + csv_name, index=index)
    else:
        for data in list_data:
            if len(data) < 5 or data[4]: # not enforced avg
                if isinstance(data[0], list) and isinstance(data[0][0], list) and len(data[0][0]) != len(data[1]):
                    data[0] = [sum(game_data_list) / len(game_data_list) for game_data_list in data[0]]
                elif isinstance(data[0], list) and isinstance(data[0][0], np.ndarray) and \
                     (len(data[0][0].shape) == 1 or (len(data[0][0].shape) == 2 and data[0][0].shape[1] == 1)) and \
                     data[0][0].shape[0] > 1:
                    data[0] = [np.mean(np_array) for np_array in data[0]]
            else: # enforce no avg
                data[0] = [elem for episode_elems in data[0] for elem in episode_elems]

            pd.DataFrame(
                data[0], columns=data[1]
            ).to_csv(
                chkpt_dir + '/' + data[2] + '.csv',
                index=False if len(data) < 4 else data[3]
            )

def transform_step_list_to_game_list(length_list, per_step_list, log_interval):
    list_total_per_game = []
    list_avg_per_log_interval = []
    total_length_ = 0
    for g, length_ in enumerate(length_list):
        game_total = sum(per_step_list[total_length_:(total_length_ + length_)])
        list_total_per_game.append(game_total)
        if (g + 1) % log_interval == 0 or (g + 1) == 1:
            list_avg_per_log_interval.append(mean(list_total_per_game[-log_interval:]))
        total_length_ += length_

    return list_total_per_game, list_avg_per_log_interval

def save_logs_and_plot(experiment, chkpt_dir, plot_dir, return_data_for_plots=False):

    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        # Save train rollouts logs in files
        train_data_to_save = [
            [experiment.game_duration_list, ['Duration'], 'episode_durations'],
            [experiment.game_duration_list_avg_per_log_interval, ['Duration'], 'episode_durations_avg_per_log_interval'],
            [experiment.distance_travel_list, ['Distance'], 'distance_travel'],
            [experiment.distance_travel_list_avg_per_log_interval, ['Distance'], 'distance_travel_avg_per_log_interval'],
            [experiment.length_list, ['Length'], 'length_list'],
            [experiment.length_list_avg_per_log_interval, ['Length'], 'length_list_avg_per_log_interval'],
            [experiment.reward_list, ['Reward'], 'pure_rewards'],
            [experiment.reward_list_avg_per_log_interval, ['Reward'], 'pure_rewards_avg_per_log_interval'],
            [experiment.episodes_model_saved, ['Episodes_model_has_been_saved'], 'episodes_model_saved']
                            ] + \
            ([] if experiment.debug_ is False
                else
             [[experiment.action_history, ['X', 'Y'], 'actions']]) + \
            ([] if experiment.constr_ball_only_at_the_right_side_wrt_hole is False
                else
             [
                 [experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'ball_only_at_the_right_side_wrt_hole_num_constraint_violations'],
                 [experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval, ['Number_of_Constraint_Violations'], 'ball_only_at_the_right_side_wrt_hole_num_constraint_violations_per_log_interval'],
                 [experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'ball_only_at_the_right_side_wrt_hole_freq_constraint_violations'],
                 [experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval, ['Frequency_of_Constraint_Violations'], 'ball_only_at_the_right_side_wrt_hole_freq_constraint_violations_per_log_interval']
             ]) + \
            ([] if experiment.constr_ball_only_at_the_up_side_wrt_hole is False
                else
             [
                 [experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'ball_only_at_the_up_side_wrt_hole_num_constraint_violations'],
                 [experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval, ['Number_of_Constraint_Violations'], 'ball_only_at_the_up_side_wrt_hole_num_constraint_violations_per_log_interval'],
                 [experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'ball_only_at_the_up_side_wrt_hole_freq_constraint_violations'],
                 [experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval, ['Frequency_of_Constraint_Violations'], 'ball_only_at_the_up_side_wrt_hole_freq_constraint_violations_per_log_interval']
             ]) + \
            ([] if experiment.constr_ball_not_in_circle is False
                else
             [
                 [experiment.ball_not_in_circle_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'ball_not_in_circle_num_constraint_violations'],
                 [experiment.ball_not_in_circle_num_constraint_violated_per_log_interval, ['Number_of_Constraint_Violations'], 'ball_not_in_circle_num_constraint_violations_per_log_interval'],
                 [experiment.ball_not_in_circle_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'ball_not_in_circle_freq_constraint_violations'],
                 [experiment.ball_not_in_circle_freq_constraint_violated_per_log_interval, ['Frequency_of_Constraint_Violations'], 'ball_not_in_circle_freq_constraint_violations_per_log_interval']
             ])
        save_df_to_csv(list_data=train_data_to_save, chkpt_dir=chkpt_dir)

    if experiment.algo == 'SAC':
        if experiment.SL_finetune is False:
            if experiment.X_agent is not None and experiment.Y_agent is not None:
                sac_train_data_to_save = ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                                          [
                    [experiment.X_q1_loss_per_step_list, ['X_q1_loss'], 'X_q1_loss_avg_per_game'],
                    [experiment.X_q2_loss_per_step_list, ['X_q2_loss'], 'X_q2_loss_avg_per_game'],
                    [experiment.X_entropies_per_step_list, ['X_entropy'], 'X_entropies_avg_per_game'],
                    [experiment.X_entropy_loss_per_step_list, ['X_entropy_loss'], 'X_entropy_loss_avg_per_game'],
                    [experiment.X_policy_loss_per_step_list, ['X_policy_loss'], 'X_policy_loss_avg_per_game'],
                    [experiment.X_entropy_coef_per_step_list, ['X_entropy_coefficient'], 'X_entropy_coef_avg_per_game'],
                    [experiment.X_q1_grad_norm_clipped_value_per_step_list, ['X_q1_grad_norm_clipped_value'], 'X_q1_grad_norm_clipped_value_avg_per_game'],
                    [experiment.X_q2_grad_norm_clipped_value_per_step_list, ['X_q2_grad_norm_clipped_value'], 'X_q2_grad_norm_clipped_value_avg_per_game'],
                    [experiment.X_actor_grad_norm_clipped_value_per_step_list, ['X_actor_grad_norm_clipped_value'], 'X_actor_grad_norm_clipped_value_avg_per_game']
                                        ]) + \
                    ([] if experiment.Y_agent.axis_agent == 'X_Y'
                        else
                     [
                        [experiment.Y_q1_loss_per_step_list, ['Y_q1_loss'], 'Y_q1_loss_avg_per_game'],
                        [experiment.Y_q2_loss_per_step_list, ['Y_q2_loss'], 'Y_q2_loss_avg_per_game'],
                        [experiment.Y_entropies_per_step_list, ['Y_entropy'], 'Y_entropies_avg_per_game'],
                        [experiment.Y_entropy_loss_per_step_list, ['Y_entropy_loss'], 'Y_entropy_loss_avg_per_game'],
                        [experiment.Y_policy_loss_per_step_list, ['Y_policy_loss'], 'Y_policy_loss_avg_per_game'],
                        [experiment.Y_entropy_coef_per_step_list, ['Y_entropy_coefficient'], 'Y_entropy_coef_avg_per_game'],
                        [experiment.Y_q1_grad_norm_clipped_value_per_step_list, ['Y_q1_grad_norm_clipped_value'], 'Y_q1_grad_norm_clipped_value_avg_per_game'],
                        [experiment.Y_q2_grad_norm_clipped_value_per_step_list, ['Y_q2_grad_norm_clipped_value'], 'Y_q2_grad_norm_clipped_value_avg_per_game'],
                        [experiment.Y_actor_grad_norm_clipped_value_per_step_list, ['Y_actor_grad_norm_clipped_value'], 'Y_actor_grad_norm_clipped_value_avg_per_game'],
                     ]) + \
                    ([] if experiment.w_constraint_optimization is False
                        else
                      (
                       ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                        [
                         [experiment.X_constraint_policy_loss_term_value_per_step_list, ['X_constraint_policy_loss_term'], 'X_constraint_policy_loss_term_avg_per_game'],
                         [experiment.X_constraint_policy_loss_term_value_per_step_list, ['X_constraint_policy_loss_term'], 'X_constraint_policy_loss_term_per_step', False, False],
                         [experiment.X_constraint_lambda_loss_value_per_step_list, ['X_constraint_lambda_loss_value'], 'X_constraint_lambda_loss_value_avg_per_game'],
                         [experiment.X_policy_loss_value_wo_constraint_term_per_step_list, ['X_policy_loss_value_wo_constraint_term'], 'X_policy_loss_value_wo_constraint_term_avg_per_game'],
                         [experiment.X_constraint_lambda_per_step_list, ['X_constraint_lambda'], 'X_constraint_lambda_avg_per_game'],
                         [experiment.X_constraint_lambda_per_step_list, ['X_constraint_lambda'], 'X_constraint_lambda_per_step', False, False]
                        ]) +
                       ([] if experiment.Y_agent.axis_agent == 'X_Y'
                                             else
                        [
                        [experiment.Y_constraint_policy_loss_term_value_per_step_list, ['Y_constraint_policy_loss_term'], 'Y_constraint_policy_loss_term_avg_per_game'],
                        [experiment.Y_constraint_policy_loss_term_value_per_step_list, ['Y_constraint_policy_loss_term'], 'Y_constraint_policy_loss_term_per_step', False, False],
                        [experiment.Y_constraint_lambda_loss_value_per_step_list, ['Y_constraint_lambda_loss_value'], 'Y_constraint_lambda_loss_value_avg_per_game'],
                        [experiment.Y_policy_loss_value_wo_constraint_term_per_step_list, ['Y_policy_loss_value_wo_constraint_term'], 'Y_policy_loss_value_wo_constraint_term_avg_per_game'],
                        [experiment.Y_constraint_lambda_per_step_list, ['Y_constraint_lambda'], 'Y_constraint_lambda_avg_per_game'],
                        [experiment.Y_constraint_lambda_per_step_list, ['Y_constraint_lambda'], 'Y_constraint_lambda_per_step', False, False]
                        ])
                      )
                     )
            elif experiment.X_Y_agent is not None:
                sac_train_data_to_save = [
                    [experiment.X_Y_q1_loss_per_step_list, ['X_Y_q1_loss'], 'X_Y_q1_loss_avg_per_game'],
                    [experiment.X_Y_q2_loss_per_step_list, ['X_Y_q2_loss'], 'X_Y_q2_loss_avg_per_game'],
                    [experiment.X_Y_entropies_per_step_list, ['X_Y_entropy'], 'X_Y_entropies_avg_per_game'],
                    [experiment.X_Y_entropy_loss_per_step_list, ['X_Y_entropy_loss'], 'X_Y_entropy_loss_avg_per_game'],
                    [experiment.X_Y_policy_loss_per_step_list, ['X_Y_policy_loss'], 'X_Y_policy_loss_avg_per_game'],
                    [experiment.X_Y_entropy_coef_per_step_list, ['X_Y_entropy_coefficient'], 'X_Y_entropy_coef_avg_per_game'],
                    [experiment.X_Y_q1_grad_norm_clipped_value_per_step_list, ['X_Y_q1_grad_norm_clipped_value'], 'X_Y_q1_grad_norm_clipped_value_avg_per_game'],
                    [experiment.X_Y_q2_grad_norm_clipped_value_per_step_list, ['X_Y_q2_grad_norm_clipped_value'], 'X_Y_q2_grad_norm_clipped_value_avg_per_game'],
                    [experiment.X_Y_actor_grad_norm_clipped_value_per_step_list, ['X_Y_actor_grad_norm_clipped_value'], 'X_Y_actor_grad_norm_clipped_value_avg_per_game']
                                        ] + \
                    ([] if experiment.w_constraint_optimization is False
                        else
                     [
                         [experiment.X_Y_constraint_policy_loss_term_value_per_step_list, ['X_Y_constraint_policy_loss_term'], 'X_Y_constraint_policy_loss_term_avg_per_game'],
                         [experiment.X_Y_constraint_policy_loss_term_value_per_step_list, ['X_Y_constraint_policy_loss_term'], 'X_Y_constraint_policy_loss_term_per_step', False, False],
                         [experiment.X_Y_constraint_lambda_loss_value_per_step_list, ['X_Y_constraint_lambda_loss_value'], 'X_Y_constraint_lambda_loss_value_avg_per_game'],
                         [experiment.X_Y_policy_loss_value_wo_constraint_term_per_step_list, ['X_Y_policy_loss_value_wo_constraint_term'], 'X_Y_policy_loss_value_wo_constraint_term_avg_per_game'],
                         [experiment.X_Y_constraint_lambda_per_step_list, ['X_Y_constraint_lambda'], 'X_Y_constraint_lambda_avg_per_game'],
                         [experiment.X_Y_constraint_lambda_per_step_list, ['X_Y_constraint_lambda'], 'X_Y_constraint_lambda_per_step', False, False]
                     ])
            else:
                raise NotImplementedError

        else:
            if experiment.X_agent is not None and experiment.Y_agent is not None:
                sac_train_data_to_save = ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                                          [
                        [experiment.X_constraint_policy_loss_term_list, ['X_constraint_policy_loss_term'], 'X_constraint_policy_loss_term_per_epoch'],
                        [experiment.X_actor_grad_norm_clipped_value_list, ['X_actor_grad_norm_clipped_value'], 'X_actor_grad_norm_clipped_value_per_epoch']
                                          ]) + \
                    ([] if experiment.Y_agent.axis_agent == 'X_Y'
                        else
                     [
                        [experiment.Y_constraint_policy_loss_term_list, ['Y_constraint_policy_loss_term'], 'Y_constraint_policy_loss_term_per_epoch'],
                        [experiment.Y_actor_grad_norm_clipped_value_list, ['Y_actor_grad_norm_clipped_value'], 'Y_actor_grad_norm_clipped_value_per_epoch']
                     ])
            elif experiment.X_Y_agent is not None:
                sac_train_data_to_save = [
                    [experiment.X_Y_constraint_policy_loss_term_list, ['X_Y_constraint_policy_loss_term'], 'X_Y_constraint_policy_loss_term_per_epoch'],
                    [experiment.X_Y_actor_grad_norm_clipped_value_list, ['X_Y_actor_grad_norm_clipped_value'], 'X_Y_actor_grad_norm_clipped_value_per_epoch']
                                        ]
            else:
                raise NotImplementedError

        save_df_to_csv(list_data=sac_train_data_to_save, chkpt_dir=chkpt_dir)

    elif experiment.algo == 'coGAIL':

        # Train discriminator rewards per game and per log interval
        discr_rewards_per_game, discr_rewards_per_log_interval = transform_step_list_to_game_list(experiment.length_list, experiment.discr_rewards_per_step, experiment.log_interval)

        # Save train data
        coGAIL_train_data_to_save = [
            [experiment.bc_loss_per_update, ['BC_loss'], 'bc_loss'],
            [experiment.discr_loss_per_episode_list, ['Discriminator_loss'], 'discr_loss'],
            [experiment.discr_grad_pen_loss_per_episode_list, ['Discriminator_gradient_penalty_loss'], 'discr_loss'],
            [experiment.value_loss_per_episode_list, ['Value_loss'], 'value_loss'],
            [experiment.action_loss_per_episode_list, ['Action_loss'], 'action_loss'],
            [experiment.dist_entropy_per_episode_list, ['Distribution_entropy'], 'distribution_entropy'],
            [experiment.code_loss_per_episode_list, ['Code_loss'], 'code_loss'],
            [experiment.inv_loss_per_episode_list, ['Human_action_reconstruction_loss'], 'inv_loss'],
            [experiment.actor_critic_grad_norm_clipped_value_per_episode_list, ['Actor_critic_gradient_norm_clipped_value'], 'act_crit_grad_norm_cl_val'],
            [experiment.discr_rewards_per_episode_avg_over_games, ['Discriminator_reward'], 'discr_rewards_avg_per_episode_mean_over_games'],
            [discr_rewards_per_game, ['Discriminator_reward'], 'discr_rewards_avg_per_episode_mean_over_games'],
            [discr_rewards_per_log_interval, ['Discriminator_reward'], 'discr_rewards_avg_per_log_interval']
                                    ] + \
            ([] if experiment.pi_co.opt_robot_w_env_rewards is False else
             [
                 [experiment.discr_value_loss_per_episode_list, ['Discriminator_Value_loss'], 'discr_value_loss'],
                 [experiment.env_value_loss_per_episode_list, ['Environment_Value_loss'], 'env_value_loss'],
                 [experiment.human_action_loss_per_episode_list, ['Human_Action_loss'], 'human_action_loss'],
                 [experiment.robot_action_loss_per_episode_list, ['Robot_Action_loss'], 'robot_action_loss']
             ]) + \
            ([] if not (experiment.constr_ball_only_at_the_right_side_wrt_hole is True or
                        experiment.constr_ball_only_at_the_up_side_wrt_hole is True or
                        experiment.constr_ball_not_in_circle is True)
                else
             [
                 [experiment.robot_final_constraint_term_loss_per_episode_list, ['Robot_Final_Constraint_Term_Loss'], 'robot_final_constraint_term_loss'],
                 [experiment.constraint_lambda_loss_per_episode_list, ['Constraint_Lambda_Loss'], 'constraint_lambda_loss'],
                 [experiment.constraint_lambda_per_episode_list, ['Constraint_Lambda'], 'constraint_lambda']
             ])

        save_df_to_csv(list_data=coGAIL_train_data_to_save, chkpt_dir=chkpt_dir)

    elif experiment.algo == 'PPO':
        if experiment.SL_finetune is False:
            if experiment.X_agent is not None and experiment.Y_agent is not None:
                if experiment.icrl is True:
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                        # Train X constraint-net costs per game and per log interval
                        X_constr_net_cost_per_game, X_constr_net_cost_per_log_interval = transform_step_list_to_game_list(experiment.length_list, experiment.X_constr_net_cost_per_step, experiment.log_interval)
                    if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                        # Train X constraint-net costs per game and per log interval
                        Y_constr_net_cost_per_game, Y_constr_net_cost_per_log_interval = transform_step_list_to_game_list(experiment.length_list, experiment.Y_constr_net_cost_per_step, experiment.log_interval)

                PPO_train_data_to_save = ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                                          [
                    [experiment.X_total_loss_per_episode_list, ['X_Total_Loss'], 'X_total_loss'],
                    [experiment.X_policy_loss_per_episode_list, ['X_Policy_Loss'], 'X_policy_loss'],
                    [experiment.X_reward_value_loss_per_episode_list, ['X_Reward_Value_Loss'], 'X_rew_value_loss'],
                    [experiment.X_approx_kl_divs_per_episode_list, ['X_Approximate_KL_Divergence'], 'X_appr_kl_div'],
                    [experiment.X_entropy_loss_per_episode_list, ['X_Entropy_Loss'], 'X_entropy_loss'],
                    [experiment.X_clip_fraction_per_episode_list, ['X_Policy_Clip_Fraction'], 'X_policy_clip_frac'],
                    [experiment.X_reward_advantage_per_episode_list, ['X_Reward_Advantage'], 'X_rew_adv'],
                    [experiment.X_explained_rew_var_per_episode_list, ['X_Explained_Reward_Variance'], 'X_explained_rew_var'],
                    [experiment.X_early_stop_epoch_per_episode_list, ['X_Policy_Early_Stop_Epoch'], 'X_policy_early_stop_epoch']
                                        ]) + \
                    ([] if experiment.Y_agent.axis_agent == 'X_Y'
                        else
                     [
                         [experiment.Y_total_loss_per_episode_list, ['Y_Total_Loss'], 'Y_total_loss'],
                         [experiment.Y_policy_loss_per_episode_list, ['Y_Policy_Loss'], 'Y_policy_loss'],
                         [experiment.Y_reward_value_loss_per_episode_list, ['Y_Reward_Value_Loss'], 'Y_rew_value_loss'],
                         [experiment.Y_approx_kl_divs_per_episode_list, ['Y_Approximate_KL_Divergence'], 'Y_appr_kl_div'],
                         [experiment.Y_entropy_loss_per_episode_list, ['Y_Entropy_Loss'], 'Y_entropy_loss'],
                         [experiment.Y_clip_fraction_per_episode_list, ['Y_Policy_Clip_Fraction'], 'Y_policy_clip_frac'],
                         [experiment.Y_reward_advantage_per_episode_list, ['Y_Reward_Advantage'], 'Y_rew_adv'],
                         [experiment.Y_explained_rew_var_per_episode_list, ['Y_Explained_Reward_Variance'], 'Y_explained_rew_var'],
                         [experiment.Y_early_stop_epoch_per_episode_list, ['Y_Policy_Early_Stop_Epoch'], 'Y_policy_early_stop_epoch']
                     ]) + \
                    ([] if experiment.icrl is False and experiment.lagrangian is False
                        else
                      (
                       ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                        [
                            [experiment.X_total_policy_loss_per_episode_list, ['X_Total_Policy_Loss'], 'X_total_policy_loss'],
                            [experiment.X_dual_nu_per_episode_list, ['X_Dual_Variable'], 'X_dual_var'],
                            [experiment.X_dual_loss_per_episode_list, ['X_Dual_Loss'], 'X_dual_loss']
                        ]) +
                       ([] if experiment.X_agent.axis_agent == 'X_Y' or experiment.icrl is False
                                             else
                        [
                            [experiment.X_cost_value_loss_per_episode_list, ['X_Cost_Value_Loss'], 'X_cost_value_loss'],
                            [experiment.X_cost_advantage_per_episode_list, ['X_Cost_Advantage'], 'X_cost_adv'],
                            [experiment.X_explained_cost_var_per_episode_list, ['X_Explained_Cost_Variance'], 'X_explained_cost_var'],
                            [experiment.X_constr_net_cost_per_episode_avg_over_games, ['X_Constraint_Net_Cost'], 'X_constr_net_cost_per_epis_avg_over_games'],
                            [X_constr_net_cost_per_game, ['X_Constraint_Net_Cost'], 'X_constr_net_total_cost_per_game'],
                            [X_constr_net_cost_per_log_interval, ['X_Constraint_Net_Cost'], 'X_constr_net_total_cost_avg_per_log_interval'],
                            [experiment.X_cost_advantage_ratio_term_per_episode_list, ['X_Cost_Advantage_Ratio_Term'], 'X_cost_adv_ratio_term'],
                            [experiment.X_cost_loss_per_episode_list, ['X_Cost_Loss'], 'X_cost_loss'],
                            [experiment.X_total_loss_constr_net_per_iter_list, ['X_Total_Loss_Constraint_Net'], 'X_total_loss_constr_net'],
                            [experiment.X_expert_loss_constr_net_per_iter_list, ['X_Expert_Loss_Constraint_Net'], 'X_expert_loss_constr_net'],
                            [experiment.X_policy_loss_constr_net_wo_is_per_iter_list, ['X_Constraint_Net_Policy_Loss_WO_IS'], 'X_constr_net_policy_loss_wo_is'],
                            [experiment.X_policy_loss_constr_net_per_iter_list, ['X_Constraint_Net_Policy_Loss'], 'X_constr_net_policy_loss'],
                            [experiment.X_regularizer_loss_constr_net_per_iter_list, ['X_Constraint_Net_Regularizer_Loss'], 'X_constr_net_reg_loss'],
                            [np.concatenate([np.array(experiment.X_is_weights_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_is_weights_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_is_weights_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['X_Constraint_Net_IS_Weights_Mean', 'X_Constraint_Net_IS_Weights_Max', 'X_Constraint_Net_IS_Weights_Min'], 'X_constr_net_is_weights'],
                            [np.concatenate([np.array(experiment.X_policy_preds_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_policy_preds_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_policy_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['X_Constraint_Net_Policy_Predictions_Mean', 'X_Constraint_Net_Policy_Predictions_Max', 'X_Constraint_Net_Policy_Predictions_Min'], 'X_constr_net_policy_pred'],
                            [np.concatenate([np.array(experiment.X_expert_preds_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_expert_preds_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_expert_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['X_Constraint_Net_Expert_Predictions_Mean', 'X_Constraint_Net_Expert_Predictions_Max', 'X_Constraint_Net_Expert_Predictions_Min'], 'X_constr_net_expert_pred'],
                            [np.concatenate([np.array(experiment.X_kl_old_new_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.X_kl_new_old_constr_net_per_iter_list)[..., None]], axis=1),
                             ['X_Constraint_Net_KL_Div_Old_New', 'X_Constraint_Net_KL_Div_New_Old'], 'X_constr_net_kl_divs'],
                            [experiment.X_early_stop_itr_constr_net_per_iter_list, ['X_Constraint_Net_Early_Stop_Iteration'], 'X_constr_net_early_stop_itr']
                        ]) +
                       ([] if experiment.X_agent.axis_agent == 'X_Y' or experiment.lagrangian is False
                                             else
                        [
                            [experiment.X_lagrangian_constraint_policy_term_loss_per_episode_list, ['X_Lagrangian_Constraint_Policy_Loss_Term'], 'X_lagr_constr_policy_loss_term']
                        ]) +
                       ([] if experiment.Y_agent.axis_agent == 'X_Y'
                        else
                        [
                            [experiment.Y_total_policy_loss_per_episode_list, ['Y_Total_Policy_Loss'], 'Y_total_policy_loss'],
                            [experiment.Y_dual_nu_per_episode_list, ['Y_Dual_Variable'], 'Y_dual_var'],
                            [experiment.Y_dual_loss_per_episode_list, ['Y_Dual_Loss'], 'Y_dual_loss']
                        ]) +
                       ([] if experiment.Y_agent.axis_agent == 'X_Y' or experiment.icrl is False
                                             else
                        [
                            [experiment.Y_cost_value_loss_per_episode_list, ['Y_Cost_Value_Loss'], 'Y_cost_value_loss'],
                            [experiment.Y_cost_advantage_per_episode_list, ['Y_Cost_Advantage'], 'Y_cost_adv'],
                            [experiment.Y_explained_cost_var_per_episode_list, ['Y_Explained_Cost_Variance'], 'Y_explained_cost_var'],
                            [experiment.Y_constr_net_cost_per_episode_avg_over_games, ['Y_Constraint_Net_Cost'], 'Y_constr_net_cost_per_epis_avg_over_games'],
                            [Y_constr_net_cost_per_game, ['Y_Constraint_Net_Cost'], 'Y_constr_net_total_cost_per_game'],
                            [Y_constr_net_cost_per_log_interval, ['Y_Constraint_Net_Cost'], 'Y_constr_net_total_cost_avg_per_log_interval'],
                            [experiment.Y_cost_advantage_ratio_term_per_episode_list, ['Y_Cost_Advantage_Ratio_Term'], 'Y_cost_adv_ratio_term'],
                            [experiment.Y_cost_loss_per_episode_list, ['Y_Cost_Loss'], 'Y_cost_loss'],
                            [experiment.Y_total_loss_constr_net_per_iter_list, ['Y_Total_Loss_Constraint_Net'], 'Y_total_loss_constr_net'],
                            [experiment.Y_expert_loss_constr_net_per_iter_list, ['Y_Expert_Loss_Constraint_Net'], 'Y_expert_loss_constr_net'],
                            [experiment.Y_policy_loss_constr_net_wo_is_per_iter_list, ['Y_Constraint_Net_Policy_Loss_WO_IS'], 'Y_constr_net_policy_loss_wo_is'],
                            [experiment.Y_policy_loss_constr_net_per_iter_list, ['Y_Constraint_Net_Policy_Loss'], 'Y_constr_net_policy_loss'],
                            [experiment.Y_regularizer_loss_constr_net_per_iter_list, ['Y_Constraint_Net_Regularizer_Loss'], 'Y_constr_net_reg_loss'],
                            [np.concatenate([np.array(experiment.Y_is_weights_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_is_weights_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_is_weights_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['Y_Constraint_Net_IS_Weights_Mean', 'Y_Constraint_Net_IS_Weights_Max', 'Y_Constraint_Net_IS_Weights_Min'], 'Y_constr_net_is_weights'],
                            [np.concatenate([np.array(experiment.Y_policy_preds_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_policy_preds_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_policy_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['Y_Constraint_Net_Policy_Predictions_Mean', 'Y_Constraint_Net_Policy_Predictions_Max', 'Y_Constraint_Net_Policy_Predictions_Min'], 'Y_constr_net_policy_pred'],
                            [np.concatenate([np.array(experiment.Y_expert_preds_mean_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_expert_preds_max_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_expert_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                             ['Y_Constraint_Net_Expert_Predictions_Mean', 'Y_Constraint_Net_Expert_Predictions_Max', 'Y_Constraint_Net_Expert_Predictions_Min'], 'Y_constr_net_expert_pred'],
                            [np.concatenate([np.array(experiment.Y_kl_old_new_constr_net_per_iter_list)[..., None],
                                             np.array(experiment.Y_kl_new_old_constr_net_per_iter_list)[..., None]], axis=1),
                             ['Y_Constraint_Net_KL_Div_Old_New', 'Y_Constraint_Net_KL_Div_New_Old'], 'Y_constr_net_kl_divs'],
                            [experiment.Y_early_stop_itr_constr_net_per_iter_list, ['Y_Constraint_Net_Early_Stop_Iteration'], 'Y_constr_net_early_stop_itr']
                        ]) +
                       ([] if experiment.Y_agent.axis_agent == 'X_Y' or experiment.lagrangian is False
                        else
                        [
                            [experiment.Y_lagrangian_constraint_policy_term_loss_per_episode_list, ['Y_Lagrangian_Constraint_Policy_Loss_Term'], 'Y_lagr_constr_policy_loss_term']
                        ])
                      )
                     )

            elif experiment.X_Y_agent is not None:
                if experiment.icrl is True:
                    # Train X_Y constraint-net costs per game and per log interval
                    X_Y_constr_net_cost_per_game, X_Y_constr_net_cost_per_log_interval = transform_step_list_to_game_list(experiment.length_list, experiment.X_Y_constr_net_cost_per_step, experiment.log_interval)
                PPO_train_data_to_save = [
                                          [experiment.X_Y_total_loss_per_episode_list, ['X_Y_Total_Loss'], 'X_Y_total_loss'],
                                          [experiment.X_Y_policy_loss_per_episode_list, ['X_Y_Policy_Loss'], 'X_Y_policy_loss'],
                                          [experiment.X_Y_reward_value_loss_per_episode_list, ['X_Y_Reward_Value_Loss'], 'X_Y_rew_value_loss'],
                                          [experiment.X_Y_approx_kl_divs_per_episode_list, ['X_Y_Approximate_KL_Divergence'], 'X_Y_appr_kl_div'],
                                          [experiment.X_Y_entropy_loss_per_episode_list, ['X_Y_Entropy_Loss'], 'X_Y_entropy_loss'],
                                          [experiment.X_Y_clip_fraction_per_episode_list, ['X_Y_Policy_Clip_Fraction'], 'X_Y_policy_clip_frac'],
                                          [experiment.X_Y_reward_advantage_per_episode_list, ['X_Y_Reward_Advantage'], 'X_Y_rew_adv'],
                                          [experiment.X_Y_explained_rew_var_per_episode_list, ['X_Y_Explained_Reward_Variance'], 'X_Y_explained_rew_var'],
                                          [experiment.X_Y_early_stop_epoch_per_episode_list, ['X_Y_Policy_Early_Stop_Epoch'], 'X_Y_policy_early_stop_epoch']
                                          ] + \
                                         ([] if experiment.icrl is False and experiment.lagrangian is False
                                             else
                                          [
                                              [experiment.X_Y_total_policy_loss_per_episode_list, ['X_Y_Total_Policy_Loss'], 'X_Y_total_policy_loss'],
                                              [experiment.X_Y_dual_nu_per_episode_list, ['X_Y_Dual_Variable'], 'X_Y_dual_var'],
                                              [experiment.X_Y_dual_loss_per_episode_list, ['X_Y_Dual_Loss'], 'X_Y_dual_loss']
                                          ]) + \
                                         ([] if experiment.icrl is False
                                             else
                                          [
                                           [experiment.X_Y_cost_value_loss_per_episode_list, ['X_Y_Cost_Value_Loss'], 'X_Y_cost_value_loss'],
                                           [experiment.X_Y_cost_advantage_per_episode_list, ['X_Y_Cost_Advantage'], 'X_Y_cost_adv'],
                                           [experiment.X_Y_explained_cost_var_per_episode_list, ['X_Y_Explained_Cost_Variance'], 'X_Y_explained_cost_var'],
                                           [experiment.X_Y_constr_net_cost_per_episode_avg_over_games, ['X_Y_Constraint_Net_Cost'], 'X_Y_constr_net_cost_per_epis_avg_over_games'],
                                           [X_Y_constr_net_cost_per_game, ['X_Y_Constraint_Net_Cost'], 'X_Y_constr_net_total_cost_per_game'],
                                           [X_Y_constr_net_cost_per_log_interval, ['X_Y_Constraint_Net_Cost'], 'X_Y_constr_net_total_cost_avg_per_log_interval'],
                                           [experiment.X_Y_cost_advantage_ratio_term_per_episode_list, ['X_Y_Cost_Advantage_Ratio_Term'], 'X_Y_cost_adv_ratio_term'],
                                           [experiment.X_Y_cost_loss_per_episode_list, ['X_Y_Cost_Loss'], 'X_Y_cost_loss'],
                                           [experiment.X_Y_total_loss_constr_net_per_iter_list, ['X_Y_Total_Loss_Constraint_Net'], 'X_Y_total_loss_constr_net'],
                                           [experiment.X_Y_expert_loss_constr_net_per_iter_list, ['X_Y_Expert_Loss_Constraint_Net'], 'X_Y_expert_loss_constr_net'],
                                           [experiment.X_Y_policy_loss_constr_net_wo_is_per_iter_list, ['X_Y_Constraint_Net_Policy_Loss_WO_IS'], 'X_Y_constr_net_policy_loss_wo_is'],
                                           [experiment.X_Y_policy_loss_constr_net_per_iter_list, ['X_Y_Constraint_Net_Policy_Loss'], 'X_Y_constr_net_policy_loss'],
                                           [experiment.X_Y_regularizer_loss_constr_net_per_iter_list, ['X_Y_Constraint_Net_Regularizer_Loss'], 'X_Y_constr_net_reg_loss'],
                                           [np.concatenate([np.array(experiment.X_Y_is_weights_mean_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_is_weights_max_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_is_weights_min_constr_net_per_iter_list)[..., None]], axis=1),
                                            ['X_Y_Constraint_Net_IS_Weights_Mean', 'X_Y_Constraint_Net_IS_Weights_Max', 'X_Y_Constraint_Net_IS_Weights_Min'], 'X_Y_constr_net_is_weights'],
                                           [np.concatenate([np.array(experiment.X_Y_policy_preds_mean_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_policy_preds_max_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_policy_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                                            ['X_Y_Constraint_Net_Policy_Predictions_Mean', 'X_Y_Constraint_Net_Policy_Predictions_Max', 'X_Y_Constraint_Net_Policy_Predictions_Min'], 'X_Y_constr_net_policy_pred'],
                                           [np.concatenate([np.array(experiment.X_Y_expert_preds_mean_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_expert_preds_max_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_expert_preds_min_constr_net_per_iter_list)[..., None]], axis=1),
                                            ['X_Y_Constraint_Net_Expert_Predictions_Mean', 'X_Y_Constraint_Net_Expert_Predictions_Max', 'X_Y_Constraint_Net_Expert_Predictions_Min'], 'X_Y_constr_net_expert_pred'],
                                           [np.concatenate([np.array(experiment.X_Y_kl_old_new_constr_net_per_iter_list)[..., None],
                                                            np.array(experiment.X_Y_kl_new_old_constr_net_per_iter_list)[..., None]], axis=1),
                                            ['X_Y_Constraint_Net_KL_Div_Old_New', 'X_Y_Constraint_Net_KL_Div_New_Old'], 'X_Y_constr_net_kl_divs'],
                                           [experiment.X_Y_early_stop_itr_constr_net_per_iter_list, ['X_Y_Constraint_Net_Early_Stop_Iteration'], 'X_Y_constr_net_early_stop_itr']
                                          ]) + \
                                         ([] if experiment.lagrangian is False
                                             else
                                          [
                                              [experiment.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list, ['X_Y_Lagrangian_Constraint_Policy_Loss_Term'], 'X_Y_lagr_constr_policy_loss_term']
                                          ])
            else:
                raise NotImplementedError

        else:
            if experiment.X_agent is not None and experiment.Y_agent is not None:
                PPO_train_data_to_save = ([] if experiment.X_agent.axis_agent == 'X_Y'
                                             else
                                          [
                        [experiment.X_constraint_policy_loss_term_list, ['X_constraint_policy_loss_term'], 'X_constraint_policy_loss_term_per_epoch']
                                          ]) + \
                    ([] if experiment.Y_agent.axis_agent == 'X_Y'
                        else
                     [
                        [experiment.Y_constraint_policy_loss_term_list, ['Y_constraint_policy_loss_term'], 'Y_constraint_policy_loss_term_per_epoch']
                     ])
            elif experiment.X_Y_agent is not None:
                PPO_train_data_to_save = [
                    [experiment.X_Y_constraint_policy_loss_term_list, ['X_Y_constraint_policy_loss_term'], 'X_Y_constraint_policy_loss_term_per_epoch']
                                        ]
            else:
                raise NotImplementedError

        save_df_to_csv(list_data=PPO_train_data_to_save, chkpt_dir=chkpt_dir)
    else:
        raise NotImplementedError

    # Save test logs in files
    if experiment.algo == 'PPO':

        if experiment.icrl is True:

            # Transform per step constraint-net cost to avg per entire test
            total_tests = len(experiment.test_reward_list_avg_per_test)
            X_constr_net_cost_avg_per_entire_test_list = []
            Y_constr_net_cost_avg_per_entire_test_list = []
            X_Y_constr_net_cost_avg_per_entire_test_list = []
            for entire_test_ in range(total_tests):
                X_constr_net_cost_total_per_single_test_list = []
                Y_constr_net_cost_total_per_single_test_list = []
                X_Y_constr_net_cost_total_per_single_test_list = []
                for single_test_ in range(experiment.test_max_games):
                    assert experiment.test_length_list[(experiment.test_max_games*entire_test_) + single_test_] == len(experiment.constraint_net_cost_per_game_list_test[(experiment.test_max_games*entire_test_) + single_test_]), ""
                    X_constr_net_cost_total_per_single_test_per_single_step_list = []
                    Y_constr_net_cost_total_per_single_test_per_single_step_list = []
                    X_Y_constr_net_cost_total_per_single_test_per_single_step_list = []
                    for single_test_step_ in range(experiment.test_length_list[(experiment.test_max_games*entire_test_) + single_test_]):
                        if experiment.X_agent is not None:
                            X_constr_net_cost_total_per_single_test_per_single_step_list.append(experiment.constraint_net_cost_per_game_list_test[(experiment.test_max_games * entire_test_) + single_test_][single_test_step_][0].item())
                        if experiment.Y_agent is not None:
                            Y_constr_net_cost_total_per_single_test_per_single_step_list.append(experiment.constraint_net_cost_per_game_list_test[(experiment.test_max_games * entire_test_) + single_test_][single_test_step_][1].item())
                        if experiment.X_Y_agent is not None:
                            X_Y_constr_net_cost_total_per_single_test_per_single_step_list.append(experiment.constraint_net_cost_per_game_list_test[(experiment.test_max_games * entire_test_) + single_test_][single_test_step_].item())
                    if experiment.X_agent is not None:
                        X_constr_net_cost_total_per_single_test_list.append(sum(X_constr_net_cost_total_per_single_test_per_single_step_list))
                    if experiment.Y_agent is not None:
                        Y_constr_net_cost_total_per_single_test_list.append(sum(Y_constr_net_cost_total_per_single_test_per_single_step_list))
                    if experiment.X_Y_agent is not None:
                        X_Y_constr_net_cost_total_per_single_test_list.append(sum(X_Y_constr_net_cost_total_per_single_test_per_single_step_list))
                if experiment.X_agent is not None:
                    X_constr_net_cost_avg_per_entire_test_list.append(mean(X_constr_net_cost_total_per_single_test_list))
                if experiment.Y_agent is not None:
                    Y_constr_net_cost_avg_per_entire_test_list.append(mean(Y_constr_net_cost_total_per_single_test_list))
                if experiment.X_Y_agent is not None:
                    X_Y_constr_net_cost_avg_per_entire_test_list.append(mean(X_Y_constr_net_cost_total_per_single_test_list))

    test_data_to_save = [
        [experiment.test_game_duration_list, ['Duration'], 'test_episode_duration'],
        [experiment.test_game_duration_list_avg_per_test, ['Duration'], 'test_episode_duration_avg_per_test'],
        [experiment.test_distance_travel_list, ['Distance'], 'distance_travel_test'],
        [experiment.test_distance_travel_list_avg_per_test, ['Distance'], 'distance_travel_test_avg_per_test'],
        [experiment.test_length_list, ['Length'], 'test_length'],
        [experiment.test_length_list_avg_per_test, ['Length'], 'test_length_avg_per_test'],
        [experiment.test_reward_list, ['Reward'], 'pure_rewards_test'],
        [experiment.test_reward_list_avg_per_test, ['Reward'], 'pure_rewards_test_avg_per_test']
                        ] + \
        ([] if experiment.debug_ is False
            else
         [[experiment.test_action_history, ['X', 'Y'], 'test_actions']]) + \
        ([] if experiment.constr_ball_only_at_the_right_side_wrt_hole is False
            else
         [
             [experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'test_ball_only_at_the_right_side_wrt_hole_num_constraint_violations'],
             [experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test, ['Number_of_Constraint_Violations'], 'test_ball_only_at_the_right_side_wrt_hole_num_constraint_violations_avg_per_test'],
             [experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violations'],
             [experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test, ['Frequency_of_Constraint_Violations'], 'test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violations_avg_per_test']
         ]) + \
        ([] if experiment.constr_ball_only_at_the_up_side_wrt_hole is False
            else
         [
             [experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'test_ball_only_at_the_up_side_wrt_hole_num_constraint_violations'],
             [experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test, ['Number_of_Constraint_Violations'], 'test_ball_only_at_the_up_side_wrt_hole_num_constraint_violations_avg_per_test'],
             [experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violations'],
             [experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test, ['Frequency_of_Constraint_Violations'], 'test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violations_avg_per_test']
         ]) + \
        ([] if experiment.constr_ball_not_in_circle is False
            else
         [
             [experiment.test_ball_not_in_circle_num_constraint_violated_list, ['Number_of_Constraint_Violations'], 'test_ball_not_in_circle_num_constraint_violations'],
             [experiment.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test, ['Number_of_Constraint_Violations'], 'test_ball_not_in_circle_num_constraint_violations_avg_per_test'],
             [experiment.test_ball_not_in_circle_freq_constraint_violated_list, ['Frequency_of_Constraint_Violations'], 'test_ball_not_in_circle_freq_constraint_violations'],
             [experiment.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test, ['Frequency_of_Constraint_Violations'], 'test_ball_not_in_circle_freq_constraint_violations_avg_per_test']
         ]) + \
        ([] if experiment.algo != 'coGAIL'
            else
         [
            [experiment.discr_reward_per_game_list_test, ['Discriminator_reward'], 'discr_rewards_test']
         ]) + \
        ([] if not (experiment.algo == 'PPO' and experiment.icrl is True)
            else ([] if experiment.X_agent is None
                     else
                  [
                     [X_constr_net_cost_avg_per_entire_test_list, ['X_Constraint_Net_Cost'], 'X_constr_net_cost_test']
                  ]) +
                 ([] if experiment.Y_agent is None
                     else
                  [
                      [Y_constr_net_cost_avg_per_entire_test_list, ['Y_Constraint_Net_Cost'], 'Y_constr_net_cost_test']
                  ]) +
                 ([] if experiment.X_Y_agent is None
                     else
                  [
                      [X_Y_constr_net_cost_avg_per_entire_test_list, ['X_Y_Constraint_Net_Cost'], 'X_Y_constr_net_cost_test']
                  ])
         )
    save_df_to_csv(list_data=test_data_to_save, chkpt_dir=chkpt_dir)

    ## Plot metrics
    assert len(experiment.test_length_list_avg_per_test) == \
           len(experiment.test_distance_travel_list_avg_per_test) == \
           len(experiment.test_game_duration_list_avg_per_test) == \
           len(experiment.test_reward_list_avg_per_test)
    if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
        assert len(experiment.test_length_list_avg_per_test) == \
               len(experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test) == \
               len(experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test)
    if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
        assert len(experiment.test_length_list_avg_per_test) == \
               len(experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test) == \
               len(experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test)
    if experiment.constr_ball_not_in_circle is True:
        assert len(experiment.test_length_list_avg_per_test) == \
               len(experiment.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test) == \
               len(experiment.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test)
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        assert len(experiment.test_length_list_avg_per_test) == \
               len(experiment.length_list_avg_per_log_interval) == \
               len(experiment.distance_travel_list_avg_per_log_interval) == \
               len(experiment.game_duration_list_avg_per_log_interval) == \
               len(experiment.reward_list_avg_per_log_interval)
        if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
            assert len(experiment.test_length_list_avg_per_test) == \
                   len(experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval) == \
                   len(experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval)
        if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
            assert len(experiment.test_length_list_avg_per_test) == \
                   len(experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval) == \
                   len(experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval)
        if experiment.constr_ball_not_in_circle is True:
            assert len(experiment.test_length_list_avg_per_test) == \
                   len(experiment.ball_not_in_circle_num_constraint_violated_per_log_interval) == \
                   len(experiment.ball_not_in_circle_freq_constraint_violated_per_log_interval)

    data_for_plots_to_return = {}

    x = [i + 1 for i in range(len(experiment.test_length_list_avg_per_test))]
    single_train_update = len(x) == 1
    line_fmt = ('' if not single_train_update else 'bo') if (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is True) else ('' if not single_train_update else 'ro')

    # Length
    plt.figure()
    plot_xlabel = 'Games x' + str(experiment.log_interval)
    plt.xlabel(plot_xlabel)
    plot_ylabel = 'Length'
    plt.ylabel(plot_ylabel)
    plot_title = 'Game length (steps)'
    plt.title(plot_title)
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        plt.plot([i + 1 for i in range(len(experiment.length_list_avg_per_log_interval))], experiment.length_list_avg_per_log_interval)
    plt.plot([i + 1 for i in range(len(experiment.test_length_list_avg_per_test))], experiment.test_length_list_avg_per_test, line_fmt)
    plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
    plot_legend_loc = "upper left"
    plt.gca().legend(plot_legend, loc=plot_legend_loc)
    file_title = 'length_avg_per_log_interval'
    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
    plt.close()
    if return_data_for_plots is True:
        data_for_plots_to_return[file_title] = \
            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            data_for_plots_to_return[file_title][5]["train"] = experiment.length_list_avg_per_log_interval
        data_for_plots_to_return[file_title][5]["test"] = experiment.test_length_list_avg_per_test

    # Travel distance
    plt.figure()
    plot_xlabel = 'Games x' + str(experiment.log_interval)
    plt.xlabel(plot_xlabel)
    plot_ylabel = 'Distance'
    plt.ylabel(plot_ylabel)
    plot_title = 'Game travel distance'
    plt.title(plot_title)
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        plt.plot([i + 1 for i in range(len(experiment.distance_travel_list_avg_per_log_interval))], experiment.distance_travel_list_avg_per_log_interval)
    plt.plot([i + 1 for i in range(len(experiment.test_distance_travel_list_avg_per_test))], experiment.test_distance_travel_list_avg_per_test, line_fmt)
    plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
    plot_legend_loc = "upper left"
    plt.gca().legend(plot_legend, loc=plot_legend_loc)
    file_title = 'distance_travel_list_avg_per_log_interval'
    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
    plt.close()
    if return_data_for_plots is True:
        data_for_plots_to_return[file_title] = \
            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            data_for_plots_to_return[file_title][5]["train"] = experiment.distance_travel_list_avg_per_log_interval
        data_for_plots_to_return[file_title][5]["test"] = experiment.test_distance_travel_list_avg_per_test

    # Duration
    plt.figure()
    plot_xlabel = 'Games x' + str(experiment.log_interval)
    plt.xlabel(plot_xlabel)
    plot_ylabel = 'Duration'
    plt.ylabel(plot_ylabel)
    plot_title = 'Game duration'
    plt.title(plot_title)
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        plt.plot([i + 1 for i in range(len(experiment.game_duration_list_avg_per_log_interval))], experiment.game_duration_list_avg_per_log_interval)
    plt.plot([i + 1 for i in range(len(experiment.test_game_duration_list_avg_per_test))], experiment.test_game_duration_list_avg_per_test, line_fmt)
    plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
    plot_legend_loc = "upper left"
    plt.gca().legend(plot_legend, loc=plot_legend_loc)
    file_title = 'game_durations_avg_per_log_interval'
    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
    plt.close()
    if return_data_for_plots is True:
        data_for_plots_to_return[file_title] = \
            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            data_for_plots_to_return[file_title][5]["train"] = experiment.game_duration_list_avg_per_log_interval
        data_for_plots_to_return[file_title][5]["test"] = experiment.test_game_duration_list_avg_per_test

    # Rewards
    plt.figure()
    plot_xlabel = 'Games x' + str(experiment.log_interval)
    plt.xlabel(plot_xlabel)
    plot_ylabel = 'Reward'
    plt.ylabel(plot_ylabel)
    plot_title = 'Game rewards'
    plt.title(plot_title)
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        plt.plot([i + 1 for i in range(len(experiment.reward_list_avg_per_log_interval))], experiment.reward_list_avg_per_log_interval)
    plt.plot([i + 1 for i in range(len(experiment.test_reward_list_avg_per_test))], experiment.test_reward_list_avg_per_test, line_fmt)
    plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
    plot_legend_loc = "upper left"
    plt.gca().legend(plot_legend, loc=plot_legend_loc)
    file_title = 'rewards_avg_per_log_interval'
    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
    plt.close()
    if return_data_for_plots is True:
        data_for_plots_to_return[file_title] = \
            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            data_for_plots_to_return[file_title][5]["train"] = experiment.reward_list_avg_per_log_interval
        data_for_plots_to_return[file_title][5]["test"] = experiment.test_reward_list_avg_per_test

    plt.figure()
    plt.xlabel('Games x' + str(experiment.log_interval))
    plt.ylabel('Reward')
    plt.title('Game rewards limited')
    if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
        plt.plot([i + 1 for i in range(len(experiment.reward_list_avg_per_log_interval))], experiment.reward_list_avg_per_log_interval)
    plt.plot([i + 1 for i in range(len(experiment.test_reward_list_avg_per_test))], experiment.test_reward_list_avg_per_test, line_fmt)
    plt.ylim(-10.0, 10.0)
    plt.gca().legend(('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',), loc="upper left")
    plt.savefig(plot_dir + "/rewards_avg_per_log_interval_limited.png")
    plt.close()

    # 'ball_only_at_the_right_side_wrt_hole' constraint
    if experiment.constr_ball_only_at_the_right_side_wrt_hole is True:
        # Number of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Number of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Number of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval))],
                     experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test

        # Frequency of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Frequency of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Frequency of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval))],
                     experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test

    # 'ball_only_at_the_up_side_wrt_hole' constraint
    if experiment.constr_ball_only_at_the_up_side_wrt_hole is True:
        # Number of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Number of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Number of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval))],
                     experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test

        # Frequency of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Frequency of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Frequency of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval))],
                     experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test

    # 'ball_not_in_circle' constraint
    if experiment.constr_ball_not_in_circle is True:
        # Number of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Number of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Number of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_not_in_circle_num_constraint_violated_per_log_interval))],
                     experiment.ball_not_in_circle_num_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_not_in_circle_num_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_not_in_circle_num_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test

        # Frequency of constraint violations
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Frequency of Violations'
        plt.ylabel(plot_ylabel)
        plot_title = 'Frequency of constraint violations'
        plt.title(plot_title)
        if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
            plt.plot([i + 1 for i in range(len(experiment.ball_not_in_circle_freq_constraint_violated_per_log_interval))],
                     experiment.ball_not_in_circle_freq_constraint_violated_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test))],
                 experiment.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test, line_fmt)
        plot_legend = ('Train', 'Test') if ((experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False)) else ('Test',)
        plot_legend_loc = "upper left"
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        file_title = 'ball_not_in_circle_freq_constraint_violated_avg_per_log_interval'
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = \
                [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, {}]
            if (experiment.algo == 'coGAIL') or (experiment.algo in ['SAC', 'PPO'] and experiment.SL_finetune is False):
                data_for_plots_to_return[file_title][5]["train"] = experiment.ball_not_in_circle_freq_constraint_violated_per_log_interval
            data_for_plots_to_return[file_title][5]["test"] = experiment.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test

    if experiment.algo == 'coGAIL':
        assert len(discr_rewards_per_log_interval) == len(experiment.discr_reward_per_game_list_test)

        plt.figure()
        plt.xlabel('Games x' + str(experiment.log_interval))
        plt.ylabel('Reward')
        plt.title('Discriminator rewards')
        plt.plot([i + 1 for i in range(len(discr_rewards_per_log_interval))], discr_rewards_per_log_interval)
        plt.plot([i + 1 for i in range(len(experiment.discr_reward_per_game_list_test))], experiment.discr_reward_per_game_list_test)
        plt.gca().legend(('Train', 'Test'), loc="upper left")
        plt.savefig(plot_dir + "/discr_mean_reward_per_log_interval.png")
        plt.close()
    elif experiment.algo == 'PPO' and experiment.icrl is True:
        plt.figure()
        plot_xlabel = 'Games x' + str(experiment.log_interval)
        plt.xlabel(plot_xlabel)
        plot_ylabel = 'Cost'
        plt.ylabel(plot_ylabel)
        plot_title = 'Constraint-net cost'
        plt.title(plot_title)
        legend_tuple = tuple(())
        plot_legend_loc = "upper left"
        file_title = "constr_net_cost_per_log_interval"
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
        if experiment.X_agent is not None:
            if experiment.X_agent.axis_agent == 'X':
                assert len(X_constr_net_cost_per_log_interval) == len(X_constr_net_cost_avg_per_entire_test_list)
                plt.plot([i + 1 for i in range(len(X_constr_net_cost_per_log_interval))], X_constr_net_cost_per_log_interval)
                legend_tuple += tuple(('Agent-X, train',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_constr_net_cost_train"] = X_constr_net_cost_per_log_interval
            plt.plot([i + 1 for i in range(len(X_constr_net_cost_avg_per_entire_test_list))], X_constr_net_cost_avg_per_entire_test_list)
            legend_tuple += tuple(('Agent-X, test',))
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][5]["X_constr_net_cost_test"] = X_constr_net_cost_avg_per_entire_test_list
        if experiment.Y_agent is not None:
            if experiment.Y_agent.axis_agent == 'Y':
                assert len(Y_constr_net_cost_per_log_interval) == len(Y_constr_net_cost_avg_per_entire_test_list)
                plt.plot([i + 1 for i in range(len(Y_constr_net_cost_per_log_interval))], Y_constr_net_cost_per_log_interval)
                legend_tuple += tuple(('Agent-Y, train',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["Y_constr_net_cost_train"] = Y_constr_net_cost_per_log_interval
            plt.plot([i + 1 for i in range(len(Y_constr_net_cost_avg_per_entire_test_list))], Y_constr_net_cost_avg_per_entire_test_list)
            legend_tuple += tuple(('Agent-Y, test',))
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][5]["Y_constr_net_cost_test"] = Y_constr_net_cost_avg_per_entire_test_list
        if experiment.X_Y_agent is not None:
            if experiment.X_Y_agent.axis_agent == 'X_Y':
                assert len(X_Y_constr_net_cost_per_log_interval) == len(X_Y_constr_net_cost_avg_per_entire_test_list)
                plt.plot([i + 1 for i in range(len(X_Y_constr_net_cost_per_log_interval))], X_Y_constr_net_cost_per_log_interval)
                legend_tuple += tuple(('Agent-X_Y, train',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_constr_net_cost_train"] = X_Y_constr_net_cost_per_log_interval
            plt.plot([i + 1 for i in range(len(X_Y_constr_net_cost_avg_per_entire_test_list))], X_Y_constr_net_cost_avg_per_entire_test_list)
            legend_tuple += tuple(('Agent-X_Y, test',))
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][5]["X_Y_constr_net_cost_test"] = X_Y_constr_net_cost_avg_per_entire_test_list
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title][3] = legend_tuple
        plt.gca().legend(legend_tuple, loc=plot_legend_loc)
        plt.savefig(os.path.join(plot_dir, file_title + ".png"))
        plt.close()

    ## plot losses
    if experiment.algo == 'SAC':

        if experiment.SL_finetune is False:
            if experiment.X_agent is not None and experiment.Y_agent is not None:

                if not experiment.X_agent.axis_agent == 'X_Y':
                    assert len(experiment.X_q1_loss_per_step_list) == len(experiment.X_q2_loss_per_step_list) == \
                           len(experiment.X_entropies_per_step_list) == len(experiment.X_policy_loss_per_step_list) == \
                           len(experiment.X_entropy_loss_per_step_list) == len(experiment.X_entropy_coef_per_step_list) == \
                           len(experiment.X_q1_grad_norm_clipped_value_per_step_list) == \
                           len(experiment.X_q2_grad_norm_clipped_value_per_step_list) == \
                           len(experiment.X_actor_grad_norm_clipped_value_per_step_list)
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    assert len(experiment.Y_q1_loss_per_step_list) == len(experiment.Y_q2_loss_per_step_list) == \
                           len(experiment.Y_entropies_per_step_list) == len(experiment.Y_policy_loss_per_step_list) == \
                           len(experiment.Y_entropy_loss_per_step_list) == len(experiment.Y_entropy_coef_per_step_list) == \
                           len(experiment.Y_q1_grad_norm_clipped_value_per_step_list) == \
                           len(experiment.Y_q2_grad_norm_clipped_value_per_step_list) == \
                           len(experiment.Y_actor_grad_norm_clipped_value_per_step_list)
                if not experiment.X_agent.axis_agent == 'X_Y' and not experiment.Y_agent.axis_agent == 'X_Y':
                    assert len(experiment.X_q1_loss_per_step_list) == len(experiment.Y_q1_loss_per_step_list)

                if not experiment.X_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.X_q1_loss_per_step_list))]
                elif not experiment.Y_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.Y_q1_loss_per_step_list))]
                else:
                    raise NotImplementedError

                single_train_update = len(x) == 1
                line_fmt_list = ['' if not single_train_update else 'bo',
                                 '' if not single_train_update else 'ro',
                                 '' if not single_train_update else 'go',
                                 '' if not single_train_update else 'co']

                # Q1, Q2 losses
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_q1_loss = [sum(X_q1_loss_i) / len(X_q1_loss_i)
                                 for X_q1_loss_i in experiment.X_q1_loss_per_step_list]
                    X_q2_loss = [sum(X_q2_loss_i) / len(X_q2_loss_i)
                                 for X_q2_loss_i in experiment.X_q2_loss_per_step_list]
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_q1_loss = [sum(Y_q1_loss_i) / len(Y_q1_loss_i)
                                 for Y_q1_loss_i in experiment.Y_q1_loss_per_step_list]
                    Y_q2_loss = [sum(Y_q2_loss_i) / len(Y_q2_loss_i)
                                 for Y_q2_loss_i in experiment.Y_q2_loss_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Q-Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Q1, Q2 losses full range'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = 'q1_q2_losses_full_range'
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    plt.plot(x, X_q1_loss, line_fmt_list[0])
                    plt.plot(x, X_q2_loss, line_fmt_list[1])
                    legend_tuple += tuple(('Q1, agent-X', 'Q2, agent-X'))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_q1_loss": X_q1_loss, "X_q2_loss": X_q2_loss})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    plt.plot(x, Y_q1_loss, line_fmt_list[2])
                    plt.plot(x, Y_q2_loss, line_fmt_list[3])
                    legend_tuple += tuple(('Q1, agent-Y', 'Q2, agent-Y'))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_q1_loss": Y_q1_loss, "Y_q2_loss": Y_q2_loss})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                plt.figure()
                plt.xlabel('Episodes')
                plt.ylabel('Q-Loss')
                plt.title('Q1, Q2 losses limited')
                legend_tuple = tuple(())
                if not experiment.X_agent.axis_agent == 'X_Y':
                    plt.plot(x, X_q1_loss, line_fmt_list[0])
                    plt.plot(x, X_q2_loss, line_fmt_list[1])
                    legend_tuple += tuple(('Q1, agent-X', 'Q2, agent-X'))
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    plt.plot(x, Y_q1_loss, line_fmt_list[2])
                    plt.plot(x, Y_q2_loss, line_fmt_list[3])
                    legend_tuple += tuple(('Q1, agent-Y', 'Q2, agent-Y'))
                plt.ylim(0.0, 5.0)
                plt.gca().legend(legend_tuple, loc="upper left")
                plt.savefig(plot_dir + "/q1_q2_losses_limited.png")
                plt.close()

                # Entropies
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Entropy value'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropies'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = 'entropies'
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_entropies = [sum(X_entropies_i) / len(X_entropies_i)
                                   for X_entropies_i in experiment.X_entropies_per_step_list]
                    plt.plot(x, X_entropies, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_entropies": X_entropies})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_entropies = [sum(Y_entropies_i) / len(Y_entropies_i)
                                   for Y_entropies_i in experiment.Y_entropies_per_step_list]
                    plt.plot(x, Y_entropies, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_entropies": Y_entropies})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Entropy loss
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Entropy Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropies losses'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = 'entropies_losses'
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_entropy_loss = [sum(X_entropy_loss_i) / len(X_entropy_loss_i)
                                      for X_entropy_loss_i in experiment.X_entropy_loss_per_step_list]
                    plt.plot(x, X_entropy_loss, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_entropy_loss": X_entropy_loss})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_entropy_loss = [sum(Y_entropy_loss_i) / len(Y_entropy_loss_i)
                                      for Y_entropy_loss_i in experiment.Y_entropy_loss_per_step_list]
                    plt.plot(x, Y_entropy_loss, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_entropy_loss": Y_entropy_loss})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Policies losses
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Policy Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policies losses'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = 'policies_losses'
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_policy_loss = [sum(X_policy_loss_i) / len(X_policy_loss_i)
                                     for X_policy_loss_i in experiment.X_policy_loss_per_step_list]
                    plt.plot(x, X_policy_loss, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_policy_loss": X_policy_loss})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_policy_loss = [sum(Y_policy_loss_i) / len(Y_policy_loss_i)
                                     for Y_policy_loss_i in experiment.Y_policy_loss_per_step_list]
                    plt.plot(x, Y_policy_loss, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_policy_loss": Y_policy_loss})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Entropy coefficients
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Entropy Coefficient Value'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropy Coefficients'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = 'entropy_coefficients'
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_entropy_coef = [sum(X_entropy_coef_i) / len(X_entropy_coef_i)
                                      for X_entropy_coef_i in experiment.X_entropy_coef_per_step_list]
                    plt.plot(x, X_entropy_coef, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_entropy_coef": X_entropy_coef})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_entropy_coef = [sum(Y_entropy_coef_i) / len(Y_entropy_coef_i)
                                      for Y_entropy_coef_i in experiment.Y_entropy_coef_per_step_list]
                    plt.plot(x, Y_entropy_coef, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_entropy_coef": Y_entropy_coef})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Q1, Q2 gradient norm clipped values
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Q1, Q2 gradient norm values clipped'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "q1_q2_grad_norm_clipped_values"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_q1_grad_norm_clipped_value = \
                        [sum(X_q1_grad_norm_clipped_value_i) / len(X_q1_grad_norm_clipped_value_i)
                         for X_q1_grad_norm_clipped_value_i in experiment.X_q1_grad_norm_clipped_value_per_step_list]
                    X_q2_grad_norm_clipped_value = \
                        [sum(X_q2_grad_norm_clipped_value_i) / len(X_q2_grad_norm_clipped_value_i)
                         for X_q2_grad_norm_clipped_value_i in experiment.X_q2_grad_norm_clipped_value_per_step_list]
                    plt.plot(x, X_q1_grad_norm_clipped_value, line_fmt_list[0])
                    plt.plot(x, X_q2_grad_norm_clipped_value, line_fmt_list[1])
                    legend_tuple += tuple(('Q1, agent-X', 'Q2, agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].\
                            update({"X_q1_grad_norm_clipped_value": X_q1_grad_norm_clipped_value, "X_q2_grad_norm_clipped_value": X_q2_grad_norm_clipped_value})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_q1_grad_norm_clipped_value = \
                        [sum(Y_q1_grad_norm_clipped_value_i) / len(Y_q1_grad_norm_clipped_value_i)
                         for Y_q1_grad_norm_clipped_value_i in experiment.Y_q1_grad_norm_clipped_value_per_step_list]
                    Y_q2_grad_norm_clipped_value = \
                        [sum(Y_q2_grad_norm_clipped_value_i) / len(Y_q2_grad_norm_clipped_value_i)
                         for Y_q2_grad_norm_clipped_value_i in experiment.Y_q2_grad_norm_clipped_value_per_step_list]
                    plt.plot(x, Y_q1_grad_norm_clipped_value, line_fmt_list[2])
                    plt.plot(x, Y_q2_grad_norm_clipped_value, line_fmt_list[3])
                    legend_tuple += tuple(('Q1, agent-Y', 'Q2, agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]. \
                            update({"Y_q1_grad_norm_clipped_value": Y_q1_grad_norm_clipped_value, "Y_q2_grad_norm_clipped_value": Y_q2_grad_norm_clipped_value})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Actor gradient norm clipped values
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Actor gradient norm values clipped'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "actor_grad_norm_clipped_value"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    X_actor_grad_norm_clipped_value = \
                        [sum(X_actor_grad_norm_clipped_value_i) / len(X_actor_grad_norm_clipped_value_i)
                         for X_actor_grad_norm_clipped_value_i in experiment.X_actor_grad_norm_clipped_value_per_step_list]
                    plt.plot(x, X_actor_grad_norm_clipped_value, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_actor_grad_norm_clipped_value": X_actor_grad_norm_clipped_value})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    Y_actor_grad_norm_clipped_value = \
                        [sum(Y_actor_grad_norm_clipped_value_i) / len(Y_actor_grad_norm_clipped_value_i)
                         for Y_actor_grad_norm_clipped_value_i in experiment.Y_actor_grad_norm_clipped_value_per_step_list]
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_actor_grad_norm_clipped_value": Y_actor_grad_norm_clipped_value})
                    plt.plot(x, Y_actor_grad_norm_clipped_value, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                if experiment.w_constraint_optimization is True:

                    if not experiment.X_agent.axis_agent == 'X_Y':
                        assert len(experiment.X_q1_loss_per_step_list) == \
                               len(experiment.X_constraint_policy_loss_term_value_per_step_list) == \
                               len(experiment.X_constraint_lambda_loss_value_per_step_list) == \
                               len(experiment.X_policy_loss_value_wo_constraint_term_per_step_list) == \
                               len(experiment.X_constraint_lambda_per_step_list)
                    if not experiment.Y_agent.axis_agent == 'X_Y':
                        assert len(experiment.Y_q1_loss_per_step_list) == \
                               len(experiment.Y_constraint_policy_loss_term_value_per_step_list) == \
                               len(experiment.Y_constraint_lambda_loss_value_per_step_list) == \
                               len(experiment.Y_policy_loss_value_wo_constraint_term_per_step_list) == \
                               len(experiment.Y_constraint_lambda_per_step_list)

                    # Policy loss constraint term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Policy Loss Constraint Term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constraint_policy_loss_term"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if not experiment.X_agent.axis_agent == 'X_Y':
                        X_constraint_policy_loss_term_value = \
                            [sum(X_constraint_policy_loss_term_value_i) / len(X_constraint_policy_loss_term_value_i)
                             for X_constraint_policy_loss_term_value_i in experiment.X_constraint_policy_loss_term_value_per_step_list]
                        plt.plot(x, X_constraint_policy_loss_term_value, line_fmt_list[0])
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"X_constraint_policy_loss_term_value": X_constraint_policy_loss_term_value})
                    if not experiment.Y_agent.axis_agent == 'X_Y':
                        Y_constraint_policy_loss_term_value = \
                            [sum(Y_constraint_policy_loss_term_value_i) / len(Y_constraint_policy_loss_term_value_i)
                             for Y_constraint_policy_loss_term_value_i in experiment.Y_constraint_policy_loss_term_value_per_step_list]
                        plt.plot(x, Y_constraint_policy_loss_term_value, line_fmt_list[1])
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"Y_constraint_policy_loss_term_value": Y_constraint_policy_loss_term_value})
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint lambda constraint term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint Lambda Loss'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = 'constraint_lambda_loss_value'
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if not experiment.X_agent.axis_agent == 'X_Y':
                        X_constraint_lambda_loss_value = \
                            [sum(X_constraint_lambda_loss_value_i) / len(X_constraint_lambda_loss_value_i)
                             for X_constraint_lambda_loss_value_i in experiment.X_constraint_lambda_loss_value_per_step_list]
                        plt.plot(x, X_constraint_lambda_loss_value, line_fmt_list[0])
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"X_constraint_lambda_loss_value": X_constraint_lambda_loss_value})
                    if not experiment.Y_agent.axis_agent == 'X_Y':
                        Y_constraint_lambda_loss_value = \
                            [sum(Y_constraint_lambda_loss_value_i) / len(Y_constraint_lambda_loss_value_i)
                             for Y_constraint_lambda_loss_value_i in experiment.Y_constraint_lambda_loss_value_per_step_list]
                        plt.plot(x, Y_constraint_lambda_loss_value, line_fmt_list[1])
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"Y_constraint_lambda_loss_value": Y_constraint_lambda_loss_value})
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Policy loss without constraint term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Policy Loss without constraint term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "policy_loss_value_wo_constraint_term"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if not experiment.X_agent.axis_agent == 'X_Y':
                        X_policy_loss_value_wo_constraint_term = \
                            [sum(X_policy_loss_value_wo_constraint_term_i) / len(X_policy_loss_value_wo_constraint_term_i)
                             for X_policy_loss_value_wo_constraint_term_i in experiment.X_policy_loss_value_wo_constraint_term_per_step_list]
                        plt.plot(x, X_policy_loss_value_wo_constraint_term, line_fmt_list[0])
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].\
                                update({"X_policy_loss_value_wo_constraint_term": X_policy_loss_value_wo_constraint_term})
                    if not experiment.Y_agent.axis_agent == 'X_Y':
                        Y_policy_loss_value_wo_constraint_term = \
                            [sum(Y_policy_loss_value_wo_constraint_term_i) / len(Y_policy_loss_value_wo_constraint_term_i)
                             for Y_policy_loss_value_wo_constraint_term_i in experiment.Y_policy_loss_value_wo_constraint_term_per_step_list]
                        plt.plot(x, Y_policy_loss_value_wo_constraint_term, line_fmt_list[1])
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].\
                                update({"Y_policy_loss_value_wo_constraint_term": Y_policy_loss_value_wo_constraint_term})
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint lambda loss
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Constraint Lambda Value'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint Lambda'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constraint_lambda"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if not experiment.X_agent.axis_agent == 'X_Y':
                        X_constraint_lambda = \
                            [sum(X_constraint_lambda_i) / len(X_constraint_lambda_i)
                             for X_constraint_lambda_i in experiment.X_constraint_lambda_per_step_list]
                        plt.plot(x, X_constraint_lambda, line_fmt_list[0])
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"X_constraint_lambda": X_constraint_lambda})
                    if not experiment.Y_agent.axis_agent == 'X_Y':
                        Y_constraint_lambda = \
                            [sum(Y_constraint_lambda_i) / len(Y_constraint_lambda_i)
                             for Y_constraint_lambda_i in experiment.Y_constraint_lambda_per_step_list]
                        plt.plot(x, Y_constraint_lambda, line_fmt_list[1])
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"Y_constraint_lambda": Y_constraint_lambda})
                    data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

            if experiment.X_Y_agent is not None:

                assert len(experiment.X_Y_q1_loss_per_step_list) == len(experiment.X_Y_q2_loss_per_step_list) == \
                       len(experiment.X_Y_entropies_per_step_list) == len(experiment.X_Y_policy_loss_per_step_list) == \
                       len(experiment.X_Y_entropy_loss_per_step_list) == len(experiment.X_Y_entropy_coef_per_step_list) == \
                       len(experiment.X_Y_q1_grad_norm_clipped_value_per_step_list) == \
                       len(experiment.X_Y_q2_grad_norm_clipped_value_per_step_list) == \
                       len(experiment.X_Y_actor_grad_norm_clipped_value_per_step_list)

                x = [i + 1 for i in range(len(experiment.X_Y_q1_loss_per_step_list))]
                single_train_update = len(x) == 1
                line_fmt_list = ['' if not single_train_update else 'bo', '' if not single_train_update else 'ro']

                # Q1, Q2 losses
                X_Y_q1_loss = [sum(X_Y_q1_loss_i) / len(X_Y_q1_loss_i)
                               for X_Y_q1_loss_i in experiment.X_Y_q1_loss_per_step_list]
                X_Y_q2_loss = [sum(X_Y_q2_loss_i) / len(X_Y_q2_loss_i)
                               for X_Y_q2_loss_i in experiment.X_Y_q2_loss_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Q-Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Q1, Q2 losses full range'
                plt.title(plot_title)
                plt.plot(x, X_Y_q1_loss, line_fmt_list[0])
                plt.plot(x, X_Y_q2_loss, line_fmt_list[1])
                plot_legend = ('Q1', 'Q2')
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "q1_q2_losses_full_range"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_q1_loss": X_Y_q1_loss, "X_Y_q2_loss": X_Y_q2_loss}]

                plt.figure()
                plt.xlabel('Episodes')
                plt.ylabel('Q-Loss')
                plt.title('Q1, Q2 losses limited')
                plt.plot(x, X_Y_q1_loss, line_fmt_list[0])
                plt.plot(x, X_Y_q2_loss, line_fmt_list[1])
                plt.ylim(0.0, 5.0)
                plt.gca().legend(('Q1, agent-X_Y', 'Q2, agent-X_Y'), loc="upper left")
                plt.savefig(plot_dir + "/q1_q2_losses_limited.png")
                plt.close()

                # Entropies
                X_Y_entropies = [sum(X_Y_entropies_i) / len(X_Y_entropies_i)
                                 for X_Y_entropies_i in experiment.X_Y_entropies_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Entropy value'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropy'
                plt.title(plot_title)
                plt.plot(x, X_Y_entropies, line_fmt_list[0])
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "entropy"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_entropies": X_Y_entropies}]

                # Entropy loss
                X_Y_entropy_loss = [sum(X_Y_entropy_loss_i) / len(X_Y_entropy_loss_i)
                                    for X_Y_entropy_loss_i in experiment.X_Y_entropy_loss_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropy Loss'
                plt.title(plot_title)
                plt.plot(x, X_Y_entropy_loss, line_fmt_list[0])
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "entropy_loss"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_entropy_loss": X_Y_entropy_loss}]

                # Policies losses
                X_Y_policy_loss = [sum(X_Y_policy_loss_i) / len(X_Y_policy_loss_i)
                                   for X_Y_policy_loss_i in experiment.X_Y_policy_loss_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policy loss'
                plt.title(plot_title)
                plt.plot(x, X_Y_policy_loss, line_fmt_list[0])
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "policy_loss"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_policy_loss": X_Y_policy_loss}]

                # Entropy coefficients
                X_Y_entropy_coef = [sum(X_Y_entropy_coef_i) / len(X_Y_entropy_coef_i)
                                    for X_Y_entropy_coef_i in experiment.X_Y_entropy_coef_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Entropy Coefficient Value'
                plt.ylabel(plot_ylabel)
                plot_title = 'Entropy Coefficient'
                plt.title(plot_title)
                plt.plot(x, X_Y_entropy_coef, line_fmt_list[0])
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "entropy_coefficient"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_entropy_coef": X_Y_entropy_coef}]

                # Q1, Q2 gradient norm clipped values
                X_Y_q1_grad_norm_clipped_value = \
                    [sum(X_Y_q1_grad_norm_clipped_value_i) / len(X_Y_q1_grad_norm_clipped_value_i)
                     for X_Y_q1_grad_norm_clipped_value_i in experiment.X_Y_q1_grad_norm_clipped_value_per_step_list]
                X_Y_q2_grad_norm_clipped_value = \
                    [sum(X_Y_q2_grad_norm_clipped_value_i) / len(X_Y_q2_grad_norm_clipped_value_i)
                     for X_Y_q2_grad_norm_clipped_value_i in experiment.X_Y_q2_grad_norm_clipped_value_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Q1, Q2 gradient norm values clipped'
                plt.title(plot_title)
                plt.plot(x, X_Y_q1_grad_norm_clipped_value, line_fmt_list[0])
                plt.plot(x, X_Y_q2_grad_norm_clipped_value, line_fmt_list[1])
                plot_legend = ('Q1, agent-X_Y', 'Q2, agent-X_Y')
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "q1_q2_grad_norm_clipped_values"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_q1_grad_norm_clipped_value": X_Y_q1_grad_norm_clipped_value,
                          "X_Y_q2_grad_norm_clipped_value": X_Y_q2_grad_norm_clipped_value}]

                # Actor gradient norm clipped values
                X_Y_actor_grad_norm_clipped_value = \
                    [sum(X_Y_actor_grad_norm_clipped_value_i) / len(X_Y_actor_grad_norm_clipped_value_i)
                     for X_Y_actor_grad_norm_clipped_value_i in experiment.X_Y_actor_grad_norm_clipped_value_per_step_list]

                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Actor gradient norm values clipped'
                plt.title(plot_title)
                plt.plot(x, X_Y_actor_grad_norm_clipped_value, line_fmt_list[0])
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "actor_grad_norm_clipped_value"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_actor_grad_norm_clipped_value": X_Y_actor_grad_norm_clipped_value}]

                if experiment.w_constraint_optimization is True:

                    assert len(experiment.X_Y_q1_loss_per_step_list) == \
                           len(experiment.X_Y_constraint_policy_loss_term_value_per_step_list) == \
                           len(experiment.X_Y_constraint_lambda_loss_value_per_step_list) == \
                           len(experiment.X_Y_policy_loss_value_wo_constraint_term_per_step_list) == \
                           len(experiment.X_Y_constraint_lambda_per_step_list)

                    # Policy loss constraint term
                    X_Y_constraint_policy_loss_term_value = \
                        [sum(X_Y_constraint_policy_loss_term_value_i) / len(X_Y_constraint_policy_loss_term_value_i)
                         for X_Y_constraint_policy_loss_term_value_i in experiment.X_Y_constraint_policy_loss_term_value_per_step_list]

                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Policy Loss Constraint Term'
                    plt.title(plot_title)
                    plt.plot(x, X_Y_constraint_policy_loss_term_value, line_fmt_list[0])
                    plot_legend = ('agent-X_Y',)
                    plot_legend_loc = "upper right"
                    plt.gca().legend(plot_legend, loc=plot_legend_loc)
                    file_title = "constraint_policy_loss_term"
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                             {"X_Y_constraint_policy_loss_term_value": X_Y_constraint_policy_loss_term_value}]

                    # Constraint lambda constraint term
                    X_Y_constraint_lambda_loss_value = \
                        [sum(X_Y_constraint_lambda_loss_value_i) / len(X_Y_constraint_lambda_loss_value_i)
                         for X_Y_constraint_lambda_loss_value_i in experiment.X_Y_constraint_lambda_loss_value_per_step_list]

                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint Lambda Loss'
                    plt.title(plot_title)
                    plt.plot(x, X_Y_constraint_lambda_loss_value, line_fmt_list[0])
                    plot_legend = ('agent-X_Y',)
                    plot_legend_loc = "upper right"
                    plt.gca().legend(plot_legend, loc=plot_legend_loc)
                    file_title = "constraint_lambda_loss_value"
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                             {"X_Y_constraint_lambda_loss_value": X_Y_constraint_lambda_loss_value}]

                    # Policy loss without constraint term
                    X_Y_policy_loss_value_wo_constraint_term = \
                        [sum(X_Y_policy_loss_value_wo_constraint_term_i) / len(X_Y_policy_loss_value_wo_constraint_term_i)
                         for X_Y_policy_loss_value_wo_constraint_term_i in experiment.X_Y_policy_loss_value_wo_constraint_term_per_step_list]

                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Policy Loss without constraint term'
                    plt.title(plot_title)
                    plt.plot(x, X_Y_policy_loss_value_wo_constraint_term, line_fmt_list[0])
                    plot_legend = ('agent-X_Y',)
                    plot_legend_loc = "upper right"
                    plt.gca().legend(plot_legend, loc=plot_legend_loc)
                    file_title = "policy_loss_value_wo_constraint_term"
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                             {"X_Y_policy_loss_value_wo_constraint_term": X_Y_policy_loss_value_wo_constraint_term}]

                    # Constraint lambda loss
                    X_Y_constraint_lambda = \
                        [sum(X_Y_constraint_lambda_i) / len(X_Y_constraint_lambda_i)
                         for X_Y_constraint_lambda_i in experiment.X_Y_constraint_lambda_per_step_list]

                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Constraint Lambda Value'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint Lambda'
                    plt.title(plot_title)
                    plt.plot(x, X_Y_constraint_lambda, line_fmt_list[0])
                    plot_legend = ('agent-X_Y',)
                    plot_legend_loc = "upper right"
                    plt.gca().legend(plot_legend, loc=plot_legend_loc)
                    file_title = "constraint_lambda"
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                             {"X_Y_constraint_lambda": X_Y_constraint_lambda}]

        else:
            if experiment.X_agent is not None and experiment.Y_agent is not None:
                if not experiment.X_agent.axis_agent == 'X_Y':
                    assert len(experiment.X_constraint_policy_loss_term_list) == len(experiment.X_actor_grad_norm_clipped_value_list)
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    assert len(experiment.Y_constraint_policy_loss_term_list) == len(experiment.Y_actor_grad_norm_clipped_value_list)
                if not experiment.X_agent.axis_agent == 'X_Y' and not experiment.Y_agent.axis_agent == 'X_Y':
                    assert len(experiment.X_constraint_policy_loss_term_list) == len(experiment.Y_constraint_policy_loss_term_list)

                if not experiment.X_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.X_constraint_policy_loss_term_list))]
                elif not experiment.Y_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.Y_constraint_policy_loss_term_list))]
                else:
                    raise NotImplementedError

                single_train_update = len(x) == 1
                line_fmt_list = ['' if not single_train_update else 'bo',
                                 '' if not single_train_update else 'ro',
                                 '' if not single_train_update else 'go',
                                 '' if not single_train_update else 'co']

                # Policy loss constraint term
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policy Loss Constraint Term'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "constraint_policy_loss_term"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.X_constraint_policy_loss_term_list, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_constraint_policy_loss_term_list": experiment.X_constraint_policy_loss_term_list})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.Y_constraint_policy_loss_term_list, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_constraint_policy_loss_term_list": experiment.Y_constraint_policy_loss_term_list})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Actor gradient norm clipped values
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Actor gradient norm values clipped'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "actor_grad_norm_clipped_value"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.X_actor_grad_norm_clipped_value_list, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].\
                            update({"X_actor_grad_norm_clipped_value_list": experiment.X_actor_grad_norm_clipped_value_list})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.Y_actor_grad_norm_clipped_value_list, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].\
                            update({"Y_actor_grad_norm_clipped_value_list": experiment.Y_actor_grad_norm_clipped_value_list})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

            if experiment.X_Y_agent is not None:
                assert len(experiment.X_Y_constraint_policy_loss_term_list) == len(experiment.X_Y_actor_grad_norm_clipped_value_list)

                x = [i + 1 for i in range(len(experiment.X_Y_constraint_policy_loss_term_list))]
                single_train_update = len(x) == 1
                line_fmt = '' if not single_train_update else 'bo'

                # Policy loss constraint term
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policy Loss Constraint Term'
                plt.title(plot_title)
                plt.plot(x, experiment.X_Y_constraint_policy_loss_term_list, line_fmt)
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "constraint_policy_loss_term"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_constraint_policy_loss_term_list": experiment.X_Y_constraint_policy_loss_term_list}]

                # Actor gradient norm clipped values
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value Clipped'
                plt.ylabel(plot_ylabel)
                plot_title = 'Actor gradient norm values clipped'
                plt.title(plot_title)
                plt.plot(x, experiment.X_Y_actor_grad_norm_clipped_value_list, line_fmt)
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "actor_grad_norm_clipped_value"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_actor_grad_norm_clipped_value_list": experiment.X_Y_actor_grad_norm_clipped_value_list}]

    elif experiment.algo == 'coGAIL':

        assert len(experiment.discr_loss_per_episode_list) == len(experiment.discr_grad_pen_loss_per_episode_list) == \
               len(experiment.value_loss_per_episode_list) == len(experiment.action_loss_per_episode_list) == \
               len(experiment.dist_entropy_per_episode_list) == len(experiment.code_loss_per_episode_list) == \
               len(experiment.inv_loss_per_episode_list) == len(experiment.actor_critic_grad_norm_clipped_value_per_episode_list) == \
               len(experiment.code_variable_per_episode_list)

        x = [i + 1 for i in range(len(experiment.discr_loss_per_episode_list))]

        # Discriminator Loss
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('BCE')
        plt.title('Discriminator Loss')
        plt.plot(x, experiment.discr_loss_per_episode_list)
        plt.savefig(plot_dir + "/discr_loss.png")
        plt.close()

        # Discriminator Gradient Penalty Loss
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Gradient Penalty')
        plt.title('Discriminator Gradient Penalty Loss')
        plt.plot(x, experiment.discr_grad_pen_loss_per_episode_list)
        plt.savefig(plot_dir + "/discr_grad_pen_loss.png")
        plt.close()

        # Value Loss
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('MSE')
        plt.title('Value Loss')
        plt.plot(x, experiment.value_loss_per_episode_list)
        plt.savefig(plot_dir + "/value_loss.png")
        plt.close()

        # Action Loss
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Action Loss')
        plt.plot(x, experiment.action_loss_per_episode_list)
        plt.savefig(plot_dir + "/action_loss.png")
        plt.close()

        # Distribution Entropy
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Entropy')
        plt.title('Distribution Entropy')
        plt.plot(x, experiment.dist_entropy_per_episode_list)
        plt.savefig(plot_dir + "/distribution_entropy.png")
        plt.close()

        # Code
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Norm')
        plt.title('Code Loss')
        plt.plot(x, experiment.code_loss_per_episode_list)
        plt.savefig(plot_dir + "/code_loss.png")
        plt.close()

        # Human Action Reconstruction Error
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('2nd order Norm')
        plt.title('Human Action Reconstruction Error')
        plt.plot(x, experiment.inv_loss_per_episode_list)
        plt.savefig(plot_dir + "/inv_loss.png")
        plt.close()

        # Actor Critic Gradient Norm Clipped Value
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Gradient Norm Clipped Value')
        plt.title('Actor Critic Gradient Norm Clip')
        plt.plot(x, experiment.actor_critic_grad_norm_clipped_value_per_episode_list)
        plt.savefig(plot_dir + "/actor_critic_grad_norm_clipped_value.png")
        plt.close()

        # BC loss
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('CE')
        plt.title('BC Loss')
        plt.plot([i + 1 for i in range(len(experiment.bc_loss_per_update))], experiment.bc_loss_per_update)
        plt.savefig(plot_dir + "/bc_loss.png")
        plt.close()

        if experiment.pi_co.opt_robot_w_env_rewards:

            # Discriminator Value Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('MSE')
            plt.title('Discriminator Value Loss')
            plt.plot(x, experiment.discr_value_loss_per_episode_list)
            plt.savefig(plot_dir + "/discr_value_loss.png")
            plt.close()

            # Environment Value Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('MSE')
            plt.title('Environment Value Loss')
            plt.plot(x, experiment.env_value_loss_per_episode_list)
            plt.savefig(plot_dir + "/env_value_loss.png")
            plt.close()

            # Human Action Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Human Action Loss')
            plt.plot(x, experiment.human_action_loss_per_episode_list)
            plt.savefig(plot_dir + "/human_action_loss.png")
            plt.close()

            # Robot Action Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Robot Action Loss')
            plt.plot(x, experiment.robot_action_loss_per_episode_list)
            plt.savefig(plot_dir + "/robot_action_loss.png")
            plt.close()

        if experiment.constr_ball_only_at_the_right_side_wrt_hole is True or \
           experiment.constr_ball_only_at_the_up_side_wrt_hole is True or \
           experiment.constr_ball_not_in_circle is True:

            # Robot Action Constraint Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Robot Action Final Constraint Term Loss')
            plt.plot(x, experiment.robot_final_constraint_term_loss_per_episode_list)
            plt.savefig(plot_dir + "/robot_final_constraint_term_loss.png")
            plt.close()

            # Constraint Lambda Loss
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.title('Constraint Lambda Loss')
            plt.plot(x, experiment.constraint_lambda_loss_per_episode_list)
            plt.savefig(plot_dir + "/constraint_lambda_loss.png")
            plt.close()

            # Constraint Lambda value
            plt.figure()
            plt.xlabel('Episodes')
            plt.ylabel('Value')
            plt.title('Constraint Lambda')
            plt.plot(x, experiment.constraint_lambda_per_episode_list)
            plt.savefig(plot_dir + "/constraint_lambda.png")
            plt.close()

    elif experiment.algo == 'PPO':
        if experiment.SL_finetune is False:
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    assert len(experiment.X_total_loss_per_episode_list) == \
                           len(experiment.X_policy_loss_per_episode_list) == \
                           len(experiment.X_reward_value_loss_per_episode_list) == \
                           len(experiment.X_approx_kl_divs_per_episode_list) == \
                           len(experiment.X_entropy_loss_per_episode_list) == \
                           len(experiment.X_clip_fraction_per_episode_list) == \
                           len(experiment.X_reward_advantage_per_episode_list) == \
                           len(experiment.X_explained_rew_var_per_episode_list) == \
                           len(experiment.X_early_stop_epoch_per_episode_list), ""
                    x = [i + 1 for i in range(len(experiment.X_total_loss_per_episode_list))]
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    assert len(experiment.Y_total_loss_per_episode_list) == \
                           len(experiment.Y_policy_loss_per_episode_list) == \
                           len(experiment.Y_reward_value_loss_per_episode_list) == \
                           len(experiment.Y_approx_kl_divs_per_episode_list) == \
                           len(experiment.Y_entropy_loss_per_episode_list) == \
                           len(experiment.Y_clip_fraction_per_episode_list) == \
                           len(experiment.Y_reward_advantage_per_episode_list) == \
                           len(experiment.Y_explained_rew_var_per_episode_list) == \
                           len(experiment.Y_early_stop_epoch_per_episode_list), ""
                    x = [i + 1 for i in range(len(experiment.Y_total_loss_per_episode_list))]
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X' and \
                    experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    assert len(experiment.X_total_loss_per_episode_list) == len(experiment.Y_total_loss_per_episode_list)

            elif experiment.X_Y_agent is not None:
                assert len(experiment.X_Y_total_loss_per_episode_list) == \
                       len(experiment.X_Y_policy_loss_per_episode_list) == \
                       len(experiment.X_Y_reward_value_loss_per_episode_list) == \
                       len(experiment.X_Y_approx_kl_divs_per_episode_list) == \
                       len(experiment.X_Y_entropy_loss_per_episode_list) == \
                       len(experiment.X_Y_clip_fraction_per_episode_list) == \
                       len(experiment.X_Y_reward_advantage_per_episode_list) == \
                       len(experiment.X_Y_explained_rew_var_per_episode_list) == \
                       len(experiment.X_Y_early_stop_epoch_per_episode_list), ""
                x = [i + 1 for i in range(len(experiment.X_Y_total_loss_per_episode_list))]

            # Total Actor-Critic Loss
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Loss'
            plt.ylabel(plot_ylabel)
            plot_title = 'Total Actor-Critic Loss'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "total_actor_critic_loss"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_total_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_total_actor_critic_loss"] = experiment.X_total_loss_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_total_loss_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_total_actor_critic_loss"] = experiment.Y_total_loss_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_total_loss_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_total_actor_critic_loss"] = experiment.X_Y_total_loss_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Policy Loss
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Loss'
            plt.ylabel(plot_ylabel)
            plot_title = 'Policy Loss'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "policy_loss"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_policy_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_policy_loss"] = experiment.X_policy_loss_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_policy_loss_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_policy_loss"] = experiment.Y_policy_loss_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_policy_loss_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_policy_loss"] = experiment.X_Y_policy_loss_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Reward value loss
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Loss'
            plt.ylabel(plot_ylabel)
            plot_title = 'Reward Critic Loss'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "reward_critic_loss"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_reward_value_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_reward_critic_loss"] = experiment.X_reward_value_loss_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_reward_value_loss_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_reward_critic_loss"] = experiment.Y_reward_value_loss_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_reward_value_loss_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_reward_critic_loss"] = experiment.X_Y_reward_value_loss_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Policy KL-Div
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'KL-Div'
            plt.ylabel(plot_ylabel)
            plot_title = 'Policy Approximate KL Divergence'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "policy_kl_div"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_approx_kl_divs_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_policy_kl_div"] = experiment.X_approx_kl_divs_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_approx_kl_divs_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_policy_kl_div"] = experiment.Y_approx_kl_divs_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_approx_kl_divs_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_policy_kl_div"] = experiment.X_Y_approx_kl_divs_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Entropy Loss
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Loss'
            plt.ylabel(plot_ylabel)
            plot_title = 'Entropy Loss (coefficient = 0)'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "entropy_loss"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_entropy_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_entropy_loss"] = experiment.X_entropy_loss_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_entropy_loss_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_entropy_loss"] = experiment.Y_entropy_loss_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_entropy_loss_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_entropy_loss"] = experiment.X_Y_entropy_loss_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Clip Fraction
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Fraction'
            plt.ylabel(plot_ylabel)
            plot_title = 'Policy Clip Fraction'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "policy_clip_fraction"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_clip_fraction_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_policy_clip_fraction"] = experiment.X_clip_fraction_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_clip_fraction_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_policy_clip_fraction"] = experiment.Y_clip_fraction_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_clip_fraction_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_policy_clip_fraction"] = experiment.X_Y_clip_fraction_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Reward Advantage
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Advantage'
            plt.ylabel(plot_ylabel)
            plot_title = 'Reward Advantage'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "reward_advantage"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_reward_advantage_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_reward_advantage"] = experiment.X_reward_advantage_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_reward_advantage_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_reward_advantage"] = experiment.Y_reward_advantage_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_reward_advantage_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_reward_advantage"] = experiment.X_Y_reward_advantage_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Explained Reward Variance
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Explained Variance'
            plt.ylabel(plot_ylabel)
            plot_title = 'Explained Reward Variance'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "explained_reward_variance"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_explained_rew_var_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_explained_reward_variance"] = experiment.X_explained_rew_var_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_explained_rew_var_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_explained_reward_variance"] = experiment.Y_explained_rew_var_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_explained_rew_var_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_explained_reward_variance"] = experiment.X_Y_explained_rew_var_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            # Policy Early Stop Epoch
            plt.figure()
            plot_xlabel = 'Episodes'
            plt.xlabel(plot_xlabel)
            plot_ylabel = 'Epoch'
            plt.ylabel(plot_ylabel)
            plot_title = 'Policy Early Stop'
            plt.title(plot_title)
            legend_tuple = tuple(())
            plot_legend_loc = "upper right"
            file_title = "policy_early_stop_epoch"
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = \
                    [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
            if experiment.X_agent is not None or experiment.Y_agent is not None:
                if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                    plt.plot(x, experiment.X_early_stop_epoch_per_episode_list)
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_policy_early_stop"] = experiment.X_early_stop_epoch_per_episode_list
                if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                    plt.plot(x, experiment.Y_early_stop_epoch_per_episode_list)
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["Y_policy_early_stop"] = experiment.Y_early_stop_epoch_per_episode_list
            elif experiment.X_Y_agent is not None:
                plt.plot(x, experiment.X_Y_early_stop_epoch_per_episode_list)
                legend_tuple += tuple(('agent-X_Y',))
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][5]["X_Y_policy_early_stop"] = experiment.X_Y_early_stop_epoch_per_episode_list
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title][3] = legend_tuple
            plt.gca().legend(legend_tuple, loc=plot_legend_loc)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()

            if experiment.icrl is True or experiment.lagrangian is True:
                if experiment.X_agent is not None or experiment.Y_agent is not None:
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                        assert len(experiment.X_total_loss_per_episode_list) == \
                               len(experiment.X_total_policy_loss_per_episode_list) == \
                               len(experiment.X_dual_nu_per_episode_list) == \
                               len(experiment.X_dual_loss_per_episode_list)
                        if experiment.icrl is True:
                            assert len(experiment.X_total_loss_per_episode_list) == \
                                   len(experiment.X_cost_value_loss_per_episode_list) == \
                                   len(experiment.X_cost_advantage_per_episode_list) == \
                                   len(experiment.X_explained_cost_var_per_episode_list) == \
                                   len(experiment.X_constr_net_cost_per_episode_avg_over_games) == \
                                   len(experiment.X_cost_advantage_ratio_term_per_episode_list) == \
                                   len(experiment.X_cost_loss_per_episode_list), ""
                            assert len(experiment.X_total_loss_constr_net_per_iter_list) == \
                                   len(experiment.X_expert_loss_constr_net_per_iter_list) == \
                                   len(experiment.X_policy_loss_constr_net_wo_is_per_iter_list) == \
                                   len(experiment.X_policy_loss_constr_net_per_iter_list) == \
                                   len(experiment.X_regularizer_loss_constr_net_per_iter_list) ==\
                                   len(experiment.X_is_weights_mean_constr_net_per_iter_list) == \
                                   len(experiment.X_is_weights_max_constr_net_per_iter_list) == \
                                   len(experiment.X_is_weights_min_constr_net_per_iter_list) == \
                                   len(experiment.X_policy_preds_max_constr_net_per_iter_list) == \
                                   len(experiment.X_policy_preds_min_constr_net_per_iter_list) == \
                                   len(experiment.X_policy_preds_mean_constr_net_per_iter_list) == \
                                   len(experiment.X_expert_preds_max_constr_net_per_iter_list) == \
                                   len(experiment.X_expert_preds_min_constr_net_per_iter_list) == \
                                   len(experiment.X_expert_preds_mean_constr_net_per_iter_list) == \
                                   len(experiment.X_kl_old_new_constr_net_per_iter_list) == \
                                   len(experiment.X_kl_new_old_constr_net_per_iter_list) == \
                                   len(experiment.X_early_stop_itr_constr_net_per_iter_list), ""
                            x_bw_itr = [i + 1 for i in range(len(experiment.X_total_loss_constr_net_per_iter_list))]
                        elif experiment.lagrangian is True:
                            assert len(experiment.X_total_loss_per_episode_list) == \
                                   len(experiment.X_lagrangian_constraint_policy_term_loss_per_episode_list)
                    if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                        assert len(experiment.Y_total_loss_per_episode_list) == \
                               len(experiment.Y_total_policy_loss_per_episode_list) == \
                               len(experiment.Y_dual_nu_per_episode_list) == \
                               len(experiment.Y_dual_loss_per_episode_list), ""
                        if experiment.icrl is True:
                            assert len(experiment.Y_total_loss_per_episode_list) == \
                                   len(experiment.Y_cost_value_loss_per_episode_list) == \
                                   len(experiment.Y_cost_advantage_per_episode_list) == \
                                   len(experiment.Y_explained_cost_var_per_episode_list) == \
                                   len(experiment.Y_constr_net_cost_per_episode_avg_over_games) == \
                                   len(experiment.Y_cost_advantage_ratio_term_per_episode_list) == \
                                   len(experiment.Y_cost_loss_per_episode_list), ""
                            assert len(experiment.Y_total_loss_constr_net_per_iter_list) == \
                                   len(experiment.Y_expert_loss_constr_net_per_iter_list) == \
                                   len(experiment.Y_policy_loss_constr_net_wo_is_per_iter_list) == \
                                   len(experiment.Y_policy_loss_constr_net_per_iter_list) == \
                                   len(experiment.Y_regularizer_loss_constr_net_per_iter_list) == \
                                   len(experiment.Y_is_weights_mean_constr_net_per_iter_list) == \
                                   len(experiment.Y_is_weights_max_constr_net_per_iter_list) == \
                                   len(experiment.Y_is_weights_min_constr_net_per_iter_list) == \
                                   len(experiment.Y_policy_preds_max_constr_net_per_iter_list) == \
                                   len(experiment.Y_policy_preds_min_constr_net_per_iter_list) == \
                                   len(experiment.Y_policy_preds_mean_constr_net_per_iter_list) == \
                                   len(experiment.Y_expert_preds_max_constr_net_per_iter_list) == \
                                   len(experiment.Y_expert_preds_min_constr_net_per_iter_list) == \
                                   len(experiment.Y_expert_preds_mean_constr_net_per_iter_list) == \
                                   len(experiment.Y_kl_old_new_constr_net_per_iter_list) == \
                                   len(experiment.Y_kl_new_old_constr_net_per_iter_list) == \
                                   len(experiment.Y_early_stop_itr_constr_net_per_iter_list), ""
                            x_bw_itr = [i + 1 for i in range(len(experiment.Y_total_loss_constr_net_per_iter_list))]
                        elif experiment.lagrangian is True:
                            assert len(experiment.Y_total_loss_per_episode_list) == \
                                   len(experiment.Y_lagrangian_constraint_policy_term_loss_per_episode_list)
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X' and \
                        experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y' and \
                        experiment.icrl is True:
                        assert len(experiment.X_total_loss_constr_net_per_iter_list) == len(experiment.Y_total_loss_constr_net_per_iter_list)

                elif experiment.X_Y_agent is not None:
                    assert len(experiment.X_Y_total_loss_per_episode_list) == \
                           len(experiment.X_Y_total_policy_loss_per_episode_list) == \
                           len(experiment.X_Y_dual_nu_per_episode_list) == \
                           len(experiment.X_Y_dual_loss_per_episode_list)
                    if experiment.icrl is True:
                        assert len(experiment.X_Y_total_loss_per_episode_list) == \
                               len(experiment.X_Y_cost_value_loss_per_episode_list) == \
                               len(experiment.X_Y_cost_advantage_per_episode_list) == \
                               len(experiment.X_Y_explained_cost_var_per_episode_list) == \
                               len(experiment.X_Y_constr_net_cost_per_episode_avg_over_games) == \
                               len(experiment.X_Y_cost_advantage_ratio_term_per_episode_list) == \
                               len(experiment.X_Y_cost_loss_per_episode_list), ""
                        assert len(experiment.X_Y_total_loss_constr_net_per_iter_list) == \
                               len(experiment.X_Y_expert_loss_constr_net_per_iter_list) == \
                               len(experiment.X_Y_policy_loss_constr_net_wo_is_per_iter_list) == \
                               len(experiment.X_Y_policy_loss_constr_net_per_iter_list) == \
                               len(experiment.X_Y_regularizer_loss_constr_net_per_iter_list) == \
                               len(experiment.X_Y_is_weights_mean_constr_net_per_iter_list) == \
                               len(experiment.X_Y_is_weights_max_constr_net_per_iter_list) == \
                               len(experiment.X_Y_is_weights_min_constr_net_per_iter_list) == \
                               len(experiment.X_Y_policy_preds_max_constr_net_per_iter_list) == \
                               len(experiment.X_Y_policy_preds_min_constr_net_per_iter_list) == \
                               len(experiment.X_Y_policy_preds_mean_constr_net_per_iter_list) == \
                               len(experiment.X_Y_expert_preds_max_constr_net_per_iter_list) == \
                               len(experiment.X_Y_expert_preds_min_constr_net_per_iter_list) == \
                               len(experiment.X_Y_expert_preds_mean_constr_net_per_iter_list) == \
                               len(experiment.X_Y_kl_old_new_constr_net_per_iter_list) == \
                               len(experiment.X_Y_kl_new_old_constr_net_per_iter_list) == \
                               len(experiment.X_Y_early_stop_itr_constr_net_per_iter_list), ""
                        x_bw_itr = [i + 1 for i in range(len(experiment.X_Y_total_loss_constr_net_per_iter_list))]
                    elif experiment.lagrangian is True:
                        assert len(experiment.X_Y_total_loss_per_episode_list) == \
                               len(experiment.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list)

                # Total Policy Loss
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Total Policy Loss'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "total_policy_loss"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if experiment.X_agent is not None or experiment.Y_agent is not None:
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                        plt.plot(x, experiment.X_total_policy_loss_per_episode_list)
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_total_policy_loss"] = experiment.X_total_policy_loss_per_episode_list
                    if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                        plt.plot(x, experiment.Y_total_policy_loss_per_episode_list)
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["Y_total_policy_loss"] = experiment.Y_total_policy_loss_per_episode_list
                elif experiment.X_Y_agent is not None:
                    plt.plot(x, experiment.X_Y_total_policy_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X_Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_Y_total_policy_loss"] = experiment.X_Y_total_policy_loss_per_episode_list
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Dual Variable
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Value'
                plt.ylabel(plot_ylabel)
                plot_title = 'Dual Variable'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "dual_var"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if experiment.X_agent is not None or experiment.Y_agent is not None:
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                        plt.plot(x, experiment.X_dual_nu_per_episode_list)
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_dual_nu"] = experiment.X_dual_nu_per_episode_list
                    if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                        plt.plot(x, experiment.Y_dual_nu_per_episode_list)
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["Y_dual_nu"] = experiment.Y_dual_nu_per_episode_list
                elif experiment.X_Y_agent is not None:
                    plt.plot(x, experiment.X_Y_dual_nu_per_episode_list)
                    legend_tuple += tuple(('agent-X_Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_Y_dual_nu"] = experiment.X_Y_dual_nu_per_episode_list
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                # Dual Variable
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Dual Variable Loss'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "dual_var_loss"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if experiment.X_agent is not None or experiment.Y_agent is not None:
                    if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                        plt.plot(x, experiment.X_dual_loss_per_episode_list)
                        legend_tuple += tuple(('agent-X',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_dual_loss"] = experiment.X_dual_loss_per_episode_list
                    if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                        plt.plot(x, experiment.Y_dual_loss_per_episode_list)
                        legend_tuple += tuple(('agent-Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["Y_dual_loss"] = experiment.Y_dual_loss_per_episode_list
                elif experiment.X_Y_agent is not None:
                    plt.plot(x, experiment.X_Y_dual_loss_per_episode_list)
                    legend_tuple += tuple(('agent-X_Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5]["X_Y_dual_loss"] = experiment.X_Y_dual_loss_per_episode_list
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

                if experiment.icrl is True:

                    # Cost Critic Loss
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Cost Critic Loss'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "cost_value_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_cost_value_loss_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_cost_critic_loss"] = experiment.X_cost_value_loss_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_cost_value_loss_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_cost_critic_loss"] = experiment.Y_cost_value_loss_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_cost_value_loss_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_cost_critic_loss"] = experiment.X_Y_cost_value_loss_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Cost Advantage
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Advantage'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Cost Advantage'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "cost_advantage"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_cost_advantage_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_cost_advantage"] = experiment.X_cost_advantage_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_cost_advantage_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_cost_advantage"] = experiment.Y_cost_advantage_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_cost_advantage_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_cost_advantage"] = experiment.X_Y_cost_advantage_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Explained Cost Variance
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Explained Variance'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Explained Cost Variance'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "explained_cost_variance"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_explained_cost_var_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_explained_cost_variance"] = experiment.X_explained_cost_var_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_explained_cost_var_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_explained_cost_variance"] = experiment.Y_explained_cost_var_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_explained_cost_var_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_explained_cost_variance"] = experiment.X_Y_explained_cost_var_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-Net Cost avg over games
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Cost'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Cost per episode avg over games'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_cost_per_epis_avg_over_games"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_constr_net_cost_per_episode_avg_over_games)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_cost"] = experiment.X_constr_net_cost_per_episode_avg_over_games
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_constr_net_cost_per_episode_avg_over_games)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_cost"] = experiment.Y_constr_net_cost_per_episode_avg_over_games
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_constr_net_cost_per_episode_avg_over_games)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_cost"] = experiment.X_Y_constr_net_cost_per_episode_avg_over_games
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Cost_Advantage-Ratio Term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Product Value'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'CostAdvantage-Ratio Term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "cost_adv_ratio_term"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_cost_advantage_ratio_term_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_cost_adv_ratio_term"] = experiment.X_cost_advantage_ratio_term_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_cost_advantage_ratio_term_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_cost_adv_ratio_term"] = experiment.Y_cost_advantage_ratio_term_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_cost_advantage_ratio_term_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_cost_adv_ratio_term"] = experiment.X_Y_cost_advantage_ratio_term_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Cost Policy Loss
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Cost Policy Loss'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "cost_policy_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_cost_loss_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_cost_policy_loss"] = experiment.X_cost_loss_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_cost_loss_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_cost_policy_loss"] = experiment.Y_cost_loss_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_cost_loss_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_cost_policy_loss"] = experiment.X_Y_cost_loss_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Total Loss
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Total Loss'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_total_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_total_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_total_loss"] = experiment.X_total_loss_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_total_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_total_loss"] = experiment.Y_total_loss_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_total_loss_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_total_loss"] = experiment.X_Y_total_loss_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Loss Expert Term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Loss Expert Term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_expert_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_expert_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_expert_loss"] = experiment.X_expert_loss_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_expert_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_expert_loss"] = experiment.Y_expert_loss_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_expert_loss_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_expert_loss"] = experiment.X_Y_expert_loss_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Loss Policy Term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Policy Loss'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_policy_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_policy_loss_constr_net_wo_is_per_iter_list)
                            plt.plot(x_bw_itr, experiment.X_policy_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X, wo IS', 'agent-X, w IS'))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5].update({"X_constr_net_policy_loss_wo_IS": experiment.X_policy_loss_constr_net_wo_is_per_iter_list,
                                                                                "X_constr_net_policy_loss_w_IS": experiment.X_policy_loss_constr_net_per_iter_list})
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_policy_loss_constr_net_wo_is_per_iter_list)
                            plt.plot(x_bw_itr, experiment.Y_policy_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y, wo IS', 'agent-Y, w IS'))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5].update({"Y_constr_net_policy_loss_wo_IS": experiment.Y_policy_loss_constr_net_wo_is_per_iter_list,
                                                                                "Y_constr_net_policy_loss_w_IS": experiment.Y_policy_loss_constr_net_per_iter_list})
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_policy_loss_constr_net_wo_is_per_iter_list)
                        plt.plot(x_bw_itr, experiment.X_Y_policy_loss_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y, wo IS', 'agent-X_Y, w IS'))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"X_Y_constr_net_policy_loss_wo_IS": experiment.X_Y_policy_loss_constr_net_wo_is_per_iter_list,
                                                                            "X_Y_constr_net_policy_loss_w_IS": experiment.X_Y_policy_loss_constr_net_per_iter_list})
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Regularizer Loss
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Loss Regularizer Term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_regularizer_loss"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_regularizer_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_regularizer_loss"] = experiment.X_regularizer_loss_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_regularizer_loss_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_regularizer_loss"] = experiment.Y_regularizer_loss_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_regularizer_loss_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_regularizer_loss"] = experiment.X_Y_regularizer_loss_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net IS weights
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Weights value'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'IS weights'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_is_weights"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.fill_between(x_bw_itr, experiment.X_is_weights_min_constr_net_per_iter_list, experiment.X_is_weights_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.X_is_weights_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_is_weights"] = experiment.X_is_weights_mean_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.fill_between(x_bw_itr, experiment.Y_is_weights_min_constr_net_per_iter_list, experiment.Y_is_weights_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.Y_is_weights_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_is_weights"] = experiment.Y_is_weights_mean_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.fill_between(x_bw_itr, experiment.X_Y_is_weights_min_constr_net_per_iter_list, experiment.X_Y_is_weights_max_constr_net_per_iter_list, alpha=0.5)
                        plt.plot(x_bw_itr, experiment.X_Y_is_weights_mean_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_is_weights"] = experiment.X_Y_is_weights_mean_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Policy Predictions
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Constraint-net Policy Predictions'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Predictions'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_policy_preds"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.fill_between(x_bw_itr, experiment.Y_policy_preds_min_constr_net_per_iter_list,
                                             experiment.X_policy_preds_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.X_policy_preds_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_policy_preds"] = experiment.X_policy_preds_mean_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.fill_between(x_bw_itr, experiment.Y_policy_preds_min_constr_net_per_iter_list,
                                             experiment.Y_policy_preds_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.Y_policy_preds_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_policy_preds"] = experiment.Y_policy_preds_mean_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.fill_between(x_bw_itr, experiment.X_Y_policy_preds_min_constr_net_per_iter_list,
                                         experiment.X_Y_policy_preds_max_constr_net_per_iter_list, alpha=0.5)
                        plt.plot(x_bw_itr, experiment.X_Y_policy_preds_mean_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_policy_preds"] = experiment.X_Y_policy_preds_mean_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Expert Predictions
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Constraint-net Expert Predictions'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Predictions'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_expert_preds"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.fill_between(x_bw_itr, experiment.X_expert_preds_min_constr_net_per_iter_list,
                                             experiment.X_expert_preds_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.X_expert_preds_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_expert_preds"] = experiment.X_expert_preds_mean_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.fill_between(x_bw_itr, experiment.Y_expert_preds_min_constr_net_per_iter_list,
                                             experiment.Y_expert_preds_max_constr_net_per_iter_list, alpha=0.5)
                            plt.plot(x_bw_itr, experiment.Y_expert_preds_mean_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_expert_preds"] = experiment.Y_expert_preds_mean_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.fill_between(x_bw_itr, experiment.X_Y_expert_preds_min_constr_net_per_iter_list,
                                         experiment.X_Y_expert_preds_max_constr_net_per_iter_list, alpha=0.5)
                        plt.plot(x_bw_itr, experiment.X_Y_expert_preds_mean_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_expert_preds"] = experiment.X_Y_expert_preds_mean_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net KL-Divergence
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'KL-Div'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net KL-Divergence'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_kl_divs"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_kl_old_new_constr_net_per_iter_list)
                            plt.plot(x_bw_itr, experiment.X_kl_new_old_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X Old-New', 'agent-X, New-Old'))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5].update({"X_constr_net_kl_divs_old_new": experiment.X_kl_old_new_constr_net_per_iter_list,
                                                                                "X_constr_net_kl_divs_new_old": experiment.X_kl_new_old_constr_net_per_iter_list})
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_kl_old_new_constr_net_per_iter_list)
                            plt.plot(x_bw_itr, experiment.Y_kl_new_old_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y, Old-New', 'agent-Y, New-Old'))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5].update({"Y_constr_net_kl_divs_old_new": experiment.Y_kl_old_new_constr_net_per_iter_list,
                                                                                "Y_constr_net_kl_divs_new_old": experiment.Y_kl_new_old_constr_net_per_iter_list})
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_kl_old_new_constr_net_per_iter_list)
                        plt.plot(x_bw_itr, experiment.X_Y_kl_new_old_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y, Old-New', 'agent-X_Y, New-Old'))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5].update({"X_Y_constr_net_kl_divs_old_new": experiment.X_Y_kl_old_new_constr_net_per_iter_list,
                                                                            "X_Y_constr_net_kl_divs_new_old": experiment.X_Y_kl_new_old_constr_net_per_iter_list})
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                    # Constraint-net Early Stop
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Iteration'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint-net Early Stop'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "constr_net_early_stop_itr"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x_bw_itr, experiment.X_early_stop_itr_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_constr_net_early_stop_itr"] = experiment.X_early_stop_itr_constr_net_per_iter_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x_bw_itr, experiment.Y_early_stop_itr_constr_net_per_iter_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_constr_net_early_stop_itr"] = experiment.Y_early_stop_itr_constr_net_per_iter_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x_bw_itr, experiment.X_Y_early_stop_itr_constr_net_per_iter_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_constr_net_early_stop_itr"] = experiment.X_Y_early_stop_itr_constr_net_per_iter_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

                elif experiment.lagrangian is True:
                    # Lagrangian Constraint Policy Loss Term
                    plt.figure()
                    plot_xlabel = 'Episodes'
                    plt.xlabel(plot_xlabel)
                    plot_ylabel = 'Loss'
                    plt.ylabel(plot_ylabel)
                    plot_title = 'Constraint Policy Loss Term'
                    plt.title(plot_title)
                    legend_tuple = tuple(())
                    plot_legend_loc = "upper right"
                    file_title = "lagr_constr_policy_loss_term"
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title] = \
                            [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                    if experiment.X_agent is not None or experiment.Y_agent is not None:
                        if experiment.X_agent is not None and experiment.X_agent.axis_agent == 'X':
                            plt.plot(x, experiment.X_lagrangian_constraint_policy_term_loss_per_episode_list)
                            legend_tuple += tuple(('agent-X',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["X_lagr_constr_policy_loss_term"] = experiment.X_lagrangian_constraint_policy_term_loss_per_episode_list
                        if experiment.Y_agent is not None and experiment.Y_agent.axis_agent == 'Y':
                            plt.plot(x, experiment.Y_lagrangian_constraint_policy_term_loss_per_episode_list)
                            legend_tuple += tuple(('agent-Y',))
                            if return_data_for_plots is True:
                                data_for_plots_to_return[file_title][5]["Y_lagr_constr_policy_loss_term"] = experiment.Y_lagrangian_constraint_policy_term_loss_per_episode_list
                    elif experiment.X_Y_agent is not None:
                        plt.plot(x, experiment.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list)
                        legend_tuple += tuple(('agent-X_Y',))
                        if return_data_for_plots is True:
                            data_for_plots_to_return[file_title][5]["X_Y_lagr_constr_policy_loss_term"] = experiment.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][3] = legend_tuple
                    plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                    plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                    plt.close()

        else:

            if experiment.X_agent is not None and experiment.Y_agent is not None:

                if not experiment.X_agent.axis_agent == 'X_Y' and not experiment.Y_agent.axis_agent == 'X_Y':
                    assert len(experiment.X_constraint_policy_loss_term_list) == len(experiment.Y_constraint_policy_loss_term_list)

                if not experiment.X_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.X_constraint_policy_loss_term_list))]
                elif not experiment.Y_agent.axis_agent == 'X_Y':
                    x = [i + 1 for i in range(len(experiment.Y_constraint_policy_loss_term_list))]
                else:
                    raise NotImplementedError

                single_train_update = len(x) == 1
                line_fmt_list = ['' if not single_train_update else 'bo',
                                 '' if not single_train_update else 'ro',
                                 '' if not single_train_update else 'go',
                                 '' if not single_train_update else 'co']

                # Policy loss constraint term
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policy Loss Constraint Term'
                plt.title(plot_title)
                legend_tuple = tuple(())
                plot_legend_loc = "upper right"
                file_title = "constraint_policy_loss_term"
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = [plot_xlabel, plot_ylabel, plot_title, None, plot_legend_loc, {}]
                if not experiment.X_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.X_constraint_policy_loss_term_list, line_fmt_list[0])
                    legend_tuple += tuple(('agent-X',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"X_constraint_policy_loss_term_list": experiment.X_constraint_policy_loss_term_list})
                if not experiment.Y_agent.axis_agent == 'X_Y':
                    plt.plot(x, experiment.Y_constraint_policy_loss_term_list, line_fmt_list[1])
                    legend_tuple += tuple(('agent-Y',))
                    if return_data_for_plots is True:
                        data_for_plots_to_return[file_title][5].update({"Y_constraint_policy_loss_term_list": experiment.Y_constraint_policy_loss_term_list})
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title][3] = legend_tuple
                plt.gca().legend(legend_tuple, loc=plot_legend_loc)
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()

            if experiment.X_Y_agent is not None:

                x = [i + 1 for i in range(len(experiment.X_Y_constraint_policy_loss_term_list))]
                single_train_update = len(x) == 1
                line_fmt = '' if not single_train_update else 'bo'

                # Policy loss constraint term
                plt.figure()
                plot_xlabel = 'Episodes'
                plt.xlabel(plot_xlabel)
                plot_ylabel = 'Loss'
                plt.ylabel(plot_ylabel)
                plot_title = 'Policy Loss Constraint Term'
                plt.title(plot_title)
                plt.plot(x, experiment.X_Y_constraint_policy_loss_term_list, line_fmt)
                plot_legend = ('agent-X_Y',)
                plot_legend_loc = "upper right"
                plt.gca().legend(plot_legend, loc=plot_legend_loc)
                file_title = "constraint_policy_loss_term"
                plt.savefig(os.path.join(plot_dir, file_title + ".png"))
                plt.close()
                if return_data_for_plots is True:
                    data_for_plots_to_return[file_title] = \
                        [plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc,
                         {"X_Y_constraint_policy_loss_term_list": experiment.X_Y_constraint_policy_loss_term_list}]

    return data_for_plots_to_return
