"""
The code of this work is based on the following GitHub repos:
https://github.com/Roboskel-Manipulation/maze_RL_v2
"""

import os
import sys
sys.path.append('./')

# Virtual environment
from maze3D_new.Maze3DEnv import Maze3D as Maze3D_v2
from maze3D_new.utils import save_logs_and_plot

# Experiment
from experiment import Experiment

# Useful utils
from plot_utils.plot_utils import get_config, get_plot_and_chkpt_dir

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

    # Check consistency of 'save', 'save_last', and 'save_constraints'
    assert (config['game']['save'] is True and
            config['game']['save_last'] is False and
            config['game']['save_constraints'] is False) or \
           (config['game']['save'] is True and
            ((config['game']['save_last'] is True and config['game']['save_constraints'] is False) or
             (config['game']['save_last'] is False and config['game']['save_constraints'] is True))) or \
           (config['game']['save'] is False and
            config['game']['save_last'] is False and
            config['game']['save_constraints'] is False)

    # Check the consistency of 'save_constraints', 'constraint_ball_only_at_the_right_side_wrt_hole',
    # 'constraint_ball_only_at_the_up_side_wrt_hole', 'constraint_ball_not_in_circle'
    assert not (config['game']['save_constraints'] is True and
                not (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is True or
                     config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is True or
                     config['Experiment']['constraint_ball_not_in_circle'] is True))

    agents = None
    agent = None
    # Load the requested agents
    if 'SAC' in list(config.items())[1] or 'PPO' in list(config.items())[1]:

        algo = 'SAC' if 'SAC' in list(config.items())[1] else 'PPO'

        ## check config
        # 'log_interval' and 'test_every_episodes' consistency
        assert (algo == 'SAC' and config['Experiment']['log_interval'] == config['Experiment']['test_every_episodes']) or \
               (algo == 'PPO' and config['Experiment']['log_interval'] == config['Experiment']['test_every_games'])
        # One or two agents
        assert (((config['game']['X_agent'] and config['game']['Y_agent']) and not config['game']['X_Y_agent']) or
               (not config['game']['X_agent'] and not config['game']['Y_agent'] and config['game']['X_Y_agent']) or
               (config['game']['X_agent'] and not config['game']['Y_agent'] and config['game']['X_Y_agent_pretrained']) or
               (not config['game']['X_agent'] and config['game']['Y_agent'] and config['game']['X_Y_agent_pretrained']) or
               (not config['game']['X_agent'] and not config['game']['Y_agent'] and config['game']['X_Y_agent_pretrained'])) and \
               not (config["game"]["X_Y_agent"] and config["game"]["X_Y_agent_pretrained"])
        # 'X_Y_agent_pretrained' and 'load_checkpoint'
        assert (config['game']['X_Y_agent_pretrained'] and config['game']['load_checkpoint']) or \
               (not config['game']['X_Y_agent_pretrained'] and not config['game']['load_checkpoint'])

        if algo == 'SAC':
            # Check if only 'num_steps', 'n_train_episodes', or 'n_episodes' is specified
            assert (isinstance(config['SAC']['num_steps'], int) and config['SAC']['n_episodes'] == 'None' and config['SAC']['n_train_episodes'] == 'None') or \
                   (config['SAC']['num_steps'] == 'None' and isinstance(config['SAC']['n_episodes'], int) and config['SAC']['n_train_episodes'] == 'None') or \
                   (config['SAC']['num_steps'] == 'None' and config['SAC']['n_episodes'] == 'None' and isinstance(config['SAC']['n_train_episodes'], int))
            # Check the consistency between 'SL_finetuning' and the constraints
            assert not (config['SAC']['SL_finetuning'] is True and
                        not (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is True or
                             config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is True or
                             config['Experiment']['constraint_ball_not_in_circle'] is True))
            # Check the consistency between 'SL_finetuning', 'save_last', and 'save_all'
            assert not (config['SAC']['SL_finetuning'] is True and config['game']['save_last'] is False and config['game']['save_all'] is False) and \
                   not (config['game']['save_last'] is True and config['game']['save_all'] is True) and \
                   not (config['SAC']['SL_finetuning'] is False and config['game']['save_all'] is True)
            # Check the consistency between 'save' and 'save_all'
            assert not (config['game']['save'] is False and config['game']['save_all'] is True)
            # Check the consistency between 'SL_finetuning' and 'load_checkpoint'
            assert not (config['SAC']['SL_finetuning'] is True and config['game']['load_checkpoint'] is False)
            # Check the consistency between 'w_constraint_optimization',
            # 'constraint_ball_only_at_the_right_side_wrt_hole',
            # 'constraint_ball_only_at_the_up_side_wrt_hole',
            # 'constraint_ball_not_in_circle'
            assert not (config['SAC']['w_dual_grad_desc'] is True and
                        config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is False and
                        config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is False and
                        config['Experiment']['constraint_ball_not_in_circle'] is False)
            # Check the consistency between 'w_constraint_optimization' and 'w_dual_grad_desc'
            assert not (config['SAC']['w_dual_grad_desc'] is True and config['SAC']['w_constraint_optimization'] is False)
            # Check the consistency between 'w_entropy_in_constraint_policy_loss_term' and 'w_dual_grad_desc'
            assert not (config['SAC']['w_entropy_in_constraint_policy_loss_term'] is True and config['SAC']['w_dual_grad_desc'] is False)
            from rl_models.sac_discrete.utils import get_sac_agent as get_agent
        elif algo == 'PPO':
            # Check if only ICRL or lagrangian is selected.
            assert not (config['PPO']['ICRL'] is True and config['PPO']['lagrangian'] is True)
            # Check the consistency between 'SL_finetuning' and the constraints
            assert not (config['PPO']['SL_finetuning'] is True and
                        not (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is True or
                             config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is True or
                             config['Experiment']['constraint_ball_not_in_circle'] is True))
            # Check the consistency between 'SL_finetuning', 'save_last', and 'save_all'
            assert not (config['PPO']['SL_finetuning'] is True and config['game']['save_last'] is False and
                        config['game']['save_all'] is False) and \
                   not (config['game']['save_last'] is True and config['game']['save_all'] is True) and \
                   not (config['PPO']['SL_finetuning'] is False and config['game']['save_all'] is True)
            # Check the consistency between 'save' and 'save_all'
            assert not (config['game']['save'] is False and config['game']['save_all'] is True)
            # Check the consistency between 'SL_finetuning' and 'load_checkpoint'
            assert not (config['PPO']['SL_finetuning'] is True and config['game']['load_checkpoint'] is False)
            # Check the consistency between 'SL_finetuning' and 'ICRL'
            assert not (config['PPO']['SL_finetuning'] is True and config['PPO']['ICRL'] is True)
            # Check the consistency between 'SL_finetuning' and 'lagrangian'
            assert not (config['PPO']['SL_finetuning'] is True and config['PPO']['lagrangian'] is True)
            from rl_models.ppo_discrete.utils import get_ppo_agent as get_agent
        else:
            raise NotImplementedError

        if config['game']['X_agent'] and config['game']['Y_agent']:
            # Create the SAC/PPO agents, one for X-axis and one for Y-axis.
            # X_agent controls up/down, while Y_agent controls right/left.
            # Both are discrete agents
            X_agent = get_agent(config, maze, chkpt_dir, axis_agent='X')
            Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='Y')
            agents = [X_agent, Y_agent]
        elif config['game']['X_Y_agent'] or \
             (config['game']['X_Y_agent_pretrained'] and not (config['game']['X_agent'] or config['game']['Y_agent'])):
            # Create a single SAC/PPO agent for both axes.
            X_Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='X_Y')
            agent = [X_Y_agent]
        elif (config['game']['X_agent'] or config['game']['Y_agent']) and config['game']['X_Y_agent_pretrained']:
            # Create two SAC/PPO agents. One is pretrained to move both axis but now will be loaded
            # and move only one axis, while the other will be trained for the other axis.
            X_Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='X_Y', only_test=True)
            if config['game']['X_agent']:
                X_agent = get_agent(config, maze, chkpt_dir, axis_agent='X')
                agents = [X_agent, X_Y_agent]
            else:
                Y_agent = get_agent(config, maze, chkpt_dir, axis_agent='Y')
                agents = [X_Y_agent, Y_agent]
        else:
            raise NotImplementedError

    elif 'coGAIL' in list(config.items())[1]:

        # check config
        assert config['Experiment']['log_interval'] == config['Experiment']['test_every_games']
        assert (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is False and
                config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is False and
                config['Experiment']['constraint_ball_not_in_circle'] is False) or \
               (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is True and
                config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is False and
                config['Experiment']['constraint_ball_not_in_circle'] is False and
                config['coGAIL']['human_controls_axis'] == 'X') or \
               (config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is False and
                config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is True and
                config['Experiment']['constraint_ball_not_in_circle'] is False and
                config['coGAIL']['human_controls_axis'] == 'Y') or \
               (config['Experiment']['constraint_ball_not_in_circle'] is True and
                config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] is False and
                config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] is False)

        from rl_models.cogail.cogail_agents import get_cogail_agents
        agents = get_cogail_agents(config, maze, chkpt_dir)

    else:
        raise NotImplementedError

    # check 'agents' and 'agent' consistency
    assert (agents is not None and agent is None) or (agents is None and agent is not None)

    # create the experiment
    experiment = Experiment(maze, agents if agents is not None else agent, config=config)

    start_experiment = time.time()

    # Perform the training process
    if 'SAC' in list(config.items())[1]:
        experiment.train_agent0s0_sac()
    elif 'PPO' in list(config.items())[1]:
        experiment.train_agent0s0_ppo()
    elif 'coGAIL' in list(config.items())[1]:
        experiment.train_cogail_agents()

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('\nTotal Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        debug_ = config["Experiment"]["debug_"]
        if debug_ is True:
            # save training logs to a pickle file
            experiment.df.to_pickle(os.path.join(chkpt_dir, 'training_logs.pkl'))
            # save test logs to a pickle file
            experiment.df_test.to_pickle(os.path.join(chkpt_dir, 'test_logs.pkl'))

        # save the rest of the experiment logs and plot them
        data_for_plots = save_logs_and_plot(
            experiment,
            chkpt_dir,
            plot_dir,
            return_data_for_plots=config["game"]["save_data_for_plots"]
        )
        if data_for_plots is not None:
            with open(os.path.join(chkpt_dir, 'plots_data.pickle'), 'wb') as pickle_file:
                pickle.dump(data_for_plots, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        experiment.save_info(chkpt_dir, experiment_duration)


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
