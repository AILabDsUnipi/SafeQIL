import math
from datetime import timedelta

column_names = [
    "tray_rot_x",
    "tray_rot_y",
    "tray_rot_vel_x",
    "tray_rot_vel_y",
    "ball_pos_x",
    "ball_pos_y",
    "ball_vel_x",
    "ball_vel_y",
    "done",
    "fixed_done"
]


def get_distance_traveled(dist_travel, prev_observation, observation):
    """
    compounds the distance travelled by the ball
    :param dist_travel: previous distance travelled
    :param prev_observation: previous observation
    :param observation: next observation
    :return: the total travelled distance
    """
    dist_travel += math.sqrt((prev_observation[0] - observation[0]) * (prev_observation[0] - observation[0]) +
                             (prev_observation[1] - observation[1]) * (prev_observation[1] - observation[1]))
    return dist_travel


def get_row_to_store(prev_observation, done, fixed_done):
    # constructs a row to add in a dataframe
    return {
        "ball_pos_x": prev_observation[0],
        "ball_pos_y": prev_observation[1],
        "ball_vel_x": prev_observation[2],
        "ball_vel_y": prev_observation[3],
        "tray_rot_x": prev_observation[4],
        "tray_rot_y": prev_observation[5],
        "tray_rot_vel_x": prev_observation[6],
        "tray_rot_vel_y": prev_observation[7],
        "done": done,
        "fixed_done": fixed_done
    }


def get_env_action(agent_action):
    # Convert agent's action to an environment-compatible one.
    # The actions are discrete.
    if agent_action == 2:
        tmp_agent_action = -1
    elif agent_action == 0 or agent_action == 1:
        tmp_agent_action = agent_action
    else:
        print("\nWrong agent action!")
        exit(0)

    return tmp_agent_action


def test_print_logs(
        avg_score,
        avg_length,
        duration,
        constraints_violated_list=None
):

    """print logs during testing"""
    print('\n##########Average stats for testing##########')
    print('Avg reward: {}\n'
          'Avg length: {}\n'
          'Test duration: {}'.format(round(avg_score, 2),
                                     round(avg_length, 2),
                                     timedelta(seconds=duration))
          )

    if constraints_violated_list is not None:
        ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated = constraints_violated_list[0]
        ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated = constraints_violated_list[1]
        ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated = constraints_violated_list[2]
        ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated = constraints_violated_list[3]
        ball_not_in_circle_avg_num_constraint_violated = constraints_violated_list[4]
        ball_not_in_circle_avg_freq_constraint_violated = constraints_violated_list[5]
        if ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated is not None and \
           ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated is not None:
            print("\nAvg number of 'ball_only_at_the_right_side_wrt_hole' constraint violations: {}\n"
                  "Avg frequency of 'ball_only_at_the_right_side_wrt_hole' constraint violations: {}"
                  .format(round(ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated, 2),
                          round(ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated, 2)))
        if ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated is not None and \
           ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated is not None:
            print("\nAvg number of 'ball_only_at_the_up_side_wrt_hole' constraint violations: {}\n"
                  "Avg frequency of 'ball_only_at_the_up_side_wrt_hole' constraint violations: {}"
                  .format(round(ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated, 2),
                          round(ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated, 2)))
        if ball_not_in_circle_avg_num_constraint_violated is not None and \
           ball_not_in_circle_avg_freq_constraint_violated is not None:
            print("\nAvg number of 'ball_not_in_circle' constraint violations: {}\n"
                  "Avg frequency of 'ball_not_in_circle' constraint violations: {}"
                  .format(round(ball_not_in_circle_avg_num_constraint_violated, 2),
                          round(ball_not_in_circle_avg_freq_constraint_violated, 2)))


def get_agent_only_action(agent_action):
    """convert agent's action to an environment-compatible one when agent is acting alone on the board"""
    # up: 0, down: 1, left: 2, right: 3, up-left: 4, up-right: 5, down-left: 6, down-right: 7
    if agent_action == 0:
        return [1, 0]
    elif agent_action == 1:
        return [-1, 0]
    elif agent_action == 2:
        return [0, -1]
    elif agent_action == 3:
        return [0, 1]
    elif agent_action == 4:
        return [1, -1]
    elif agent_action == 5:
        return [1, 1]
    elif agent_action == 6:
        return [-1, -1]
    elif agent_action == 7:
        return [-1, 1]
    else:
        print("Invalid agent action")
