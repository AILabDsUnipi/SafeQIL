# Add source directory to the system path
import os.path

from utils.file_utils import resolve_source_path
resolve_source_path()

import sys
import time
from datetime import timedelta

import safety_gymnasium

# Experiment
from experiments.train_vicrl_experiment import VICRLExperiment
from experiments.test_experiment import TestExperiment

# Utils
from utils.file_utils import get_config, get_result_dirs
from utils.agent_utils import get_vicrl_agent
from utils.plot_utils import save_logs_and_plot, save_test_logs_and_plot
from utils.exp_utils import get_train_seed, get_test_seed
from algos.vicrl.plot_utils import save_constraint_net_logs_and_plot


def main(argv):

    # get configuration
    config = get_config(argv[0])

    # Check if the selected config file is suitable for the current experiment
    assert config["Experiment"]["test_model"] is False
    assert config["Experiment"]["save_model"] is True
    assert config["game"]["with_human"] is False

    ## Check other config settings
    # Check the consistency between 'log_interval' and 'test_every_episodes'
    assert (config['Experiment']['log_interval'] == config['Experiment']['test_every_episodes'])
    # Check the consistency between 'max_timesteps_per_game' and 'n_steps'
    assert (config['Experiment']['max_timesteps_per_game'] < config['VICRL']['n_steps'])

    # Create the environment
    env = safety_gymnasium.make(
        config["game"]["env_id"],
        render_mode="rgb_array",
        max_episode_steps=config["Experiment"]["max_timesteps_per_game"]
    )

    # ======================================== Train ======================================== #

    print("\n# ======================================== Train ======================================== #")

    # Get seed
    seed = get_train_seed(config)

    # Create the directories for files and plots of this experiment
    files_dir, plot_dir, exp_id = get_result_dirs(argv[1], argv[0], test_only=False)

    # Create the ICRL agent
    vicrl_agent = get_vicrl_agent(config, env, seed=seed)

    # Create the experiment
    experiment = VICRLExperiment(env, vicrl_agent, config, files_dir, seed)

    # Run the experiment and time it
    start_experiment = time.time()
    experiment.train()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment)

    # Save the experiment logs and plot them
    save_logs_and_plot(experiment, files_dir, plot_dir)
    save_constraint_net_logs_and_plot(experiment, files_dir, plot_dir)
    # Save the rest experiment info
    experiment.save_info(files_dir)

    # Delete experiment and agent to save memory
    del vicrl_agent, experiment

    # ======================================== Test ======================================== #

    print("\n# ======================================== Test ======================================== #")

    # Edit the config
    config['Experiment']['test_model'] = True
    config['Experiment']['load_checkpoint'] = True
    config['Experiment']['checkpoint_path'] = os.path.join(files_dir, 'chkpts')
    config['Experiment']['save_model'] = False
    config['Experiment']['normalize_rewards'] = False  # Test always with original rewards

    # Get the test seed
    test_seed = get_test_seed(config)

    # Create the directories for files and plots of this test experiment
    test_files_dir, test_plot_dir, test_exp_id = get_result_dirs(
        argv[1],
        argv[0],
        test_only=True,
        seed=test_seed,
        provided_exp_id=exp_id
    )

    # Create the test ICRL agent
    test_vicrl_agent = get_vicrl_agent(config, env, only_test=True, seed=test_seed)

    # Create the test experiment
    test_experiment = TestExperiment(env, test_vicrl_agent, config, test_files_dir, test_seed)

    # Run the test experiment and time it
    test_start_experiment = time.time()
    test_experiment.agent_test()
    test_end_experiment = time.time()
    test_experiment_duration = timedelta(seconds=test_end_experiment - test_start_experiment)

    # Save the test experiment logs and plot them
    save_test_logs_and_plot(test_experiment, test_files_dir, test_plot_dir)
    # Save the rest test experiment info
    test_experiment.save_info(test_files_dir)

    print('\nTotal experiment time: {}'.format(experiment_duration))
    print('Total test experiment time: {}'.format(test_experiment_duration))


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
