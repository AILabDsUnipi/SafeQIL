"""
The code of this work is based on the following GitHub repos:
https://github.com/Roboskel-Manipulation/maze_RL_v2
"""

# Virtual environment
import sys
sys.path.append('./')

from maze3D_new.Maze3DEnv import Maze3D as Maze3D_v2
from maze3D_new.utils import save_test_logs_and_plot

# Experiment
from experiment import Experiment

# RL modules
from plot_utils.plot_utils import get_config, get_plot_and_chkpt_dir
from rl_models.sac_discrete.utils import get_sac_agent

import sys
import time
from datetime import timedelta


def main(argv):
    # get configuration
    config = get_config(argv[0])

    # creating environment
    maze = Maze3D_v2(config_file=argv[0])

    chkpt_dir, plot_dir = [None, None]
    if config["game"]["save"]:
        # create the checkpoint and plot directories for this experiment
        chkpt_dir, plot_dir = get_plot_and_chkpt_dir(config, argv[1], argv[0])

    # Check if the selected config file is suitable for the current experiment
    assert config["game"]["test_model"] and config["game"]["load_checkpoint"]
    assert config["game"]["with_human"] and \
           (config["game"]["X_agent"] or config["game"]["Y_agent"]) and \
           not (config["game"]["X_agent"] and config["game"]["Y_agent"])

    assert config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is False and \
           config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is False and \
           config['Experiment']['constraint_ball_not_in_circle'] is False

    # create the SAC agents according to the config file.
    # agent X_sac controls up/down, while agent Y_sac controls right/left.
    # Both are discrete_sac agents
    X_sac = None
    Y_sac = None
    if config["game"]["X_agent"]:
        X_sac = get_sac_agent(config, maze, chkpt_dir, axis_agent="X", only_test=True)
    if config["game"]["Y_agent"]:
        Y_sac = get_sac_agent(config, maze, chkpt_dir, axis_agent="Y", only_test=True)

    # create the experiment
    experiment = Experiment(maze, [X_sac, Y_sac], config=config)

    start_experiment = time.time()

    experiment.human_agent_test()

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('\nTotal Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        # save training logs to a pickle file
        experiment.df_test.to_pickle(chkpt_dir + '/test_logs.pkl')
        # save rest of the experiment logs and plot them
        save_test_logs_and_plot(experiment, chkpt_dir, plot_dir)
        experiment.save_info(chkpt_dir, experiment_duration)


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
