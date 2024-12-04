import sys
sys.path.append('./')

# Virtual environment
from maze3D_new.Maze3DEnv import Maze3D as Maze3D_v2
from maze3D_new.utils import save_test_logs_and_plot

# Experiment
from experiment import Experiment

# RL modules
from plot_utils.plot_utils import get_config, get_plot_and_chkpt_dir

import sys
import time
from datetime import timedelta

"""
The code of this work is based on the following github repos:
https://github.com/Roboskel-Manipulation/maze_RL_v2
"""


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
    assert config["game"]["test_model"]
    assert config["game"]["with_human"] and \
           not config["game"]["X_agent"] and \
           not config["game"]["Y_agent"]
    assert not config["Experiment"]["freeze_motion"] or \
           (config["Experiment"]["freeze_motion"] and config["Experiment"]["render"])
    assert config["Experiment"]["render"] is True

    # create the experiment
    experiment = Experiment(maze, agents=None, config=config)

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
