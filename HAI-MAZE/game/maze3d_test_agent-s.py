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

import os
import time
from datetime import timedelta
import pickle

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

    # Load the requested agents
    agents = None
    agent = None
    if 'SAC' in list(config.items())[1] or 'PPO' in list(config.items())[1]:

        algo = 'SAC' if 'SAC' in list(config.items())[1] else 'PPO'

        # Check agent pairs consistency
        assert not config["game"]["with_human"] and \
               ((config["game"]["X_agent"] and config["game"]["Y_agent"] and not config["game"]["X_Y_agent"]) or
                (not config["game"]["X_agent"] and not config["game"]["Y_agent"] and config["game"]["X_Y_agent"]) or
                (config["game"]["X_agent"] and not config["game"]["Y_agent"] and config["game"]["X_Y_agent_pretrained"]) or
                (not config["game"]["X_agent"] and config["game"]["Y_agent"] and config["game"]["X_Y_agent_pretrained"])) \
               and not (config["game"]["X_Y_agent"] and config["game"]["X_Y_agent_pretrained"])

        if algo == 'SAC':
            from rl_models.sac_discrete.utils import get_sac_agent as get_agent
        elif algo == 'PPO':
            # Also check if only ICRL or lagrangian is selected.
            assert not (config['PPO']['ICRL'] is True and config['PPO']['lagrangian'] is True)
            from rl_models.ppo_discrete.utils import get_ppo_agent as get_agent
        else:
            raise NotImplementedError

        if config['game']['X_agent'] and config['game']['Y_agent']:
            # create the SAC/PPO agents according to the config file.
            # x_agent controls up/down, while Y_agent controls right/left.
            # Both are discrete agents
            X_agent = get_agent(config, maze, chkpt_dir, axis_agent="X", only_test=True)
            Y_agent = get_agent(config, maze, chkpt_dir, axis_agent="Y", only_test=True)
            agents = [X_agent, Y_agent]
        elif config['game']['X_Y_agent']:
            # create a single SAC/PPO agent for both axes.
            X_Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='X_Y', only_test=True)
            agent = [X_Y_agent]
        elif (config['game']['X_agent'] or config['game']['Y_agent']) and config['game']['X_Y_agent_pretrained']:
            # Create two SAC/PPO. One is pretrained to move both axis but now will be loaded
            # and move only one axis, while the other is trained to move other axis (cooperating with X_Y_agent).
            X_Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='X_Y', only_test=True)
            if config['game']['X_agent']:
                X_agent = get_agent(config, maze, chkpt_dir, axis_agent='X', only_test=True)
                agents = [X_agent, X_Y_agent]
            else:
                Y_agent = get_agent_agent(config, maze, chkpt_dir, axis_agent='Y', only_test=True)
                agents = [X_Y_agent, Y_agent]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # check 'agents' and 'agent' consistency
    assert (agents is not None and agent is None) or (agents is None and agent is not None)

    # create the experiment
    experiment = Experiment(maze, agents if agents is not None else agent, config=config)

    start_experiment = time.time()

    # Start the experiment-test
    experiment.agent0s0_test()

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('\nTotal Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        # save test logs to a pickle file
        experiment.df_test.to_pickle(os.path.join(chkpt_dir, 'test_logs.pkl'))
        # save rest of the experiment logs and plot them
        test_data_for_plots = save_test_logs_and_plot(experiment, chkpt_dir, plot_dir, return_data_for_plots=config["game"]["save_data_for_plots"])
        if test_data_for_plots is not None:
            with open(os.path.join(chkpt_dir, 'plots_test_data.pickle'), 'wb') as pickle_file:
                pickle.dump(test_data_for_plots, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        experiment.save_info(chkpt_dir, experiment_duration)


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
