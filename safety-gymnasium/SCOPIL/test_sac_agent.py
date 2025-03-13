# Add source directory to the system path
from utils.file_utils import resolve_source_path
resolve_source_path()

import sys
import time
from datetime import timedelta

import safety_gymnasium

# Experiment
from experiments.test_experiment import TestExperiment

# Utils
from utils.file_utils import get_config, get_result_dirs
from utils.agent_utils import get_sac_agent
from utils.plot_utils import save_test_logs_and_plot
from utils.exp_utils import get_seed


def main(argv):

    # get configuration
    config = get_config(argv[0])

    # Check if the selected config file is suitable for the current experiment
    assert config["Experiment"]["test_model"] is True
    assert config["game"]["with_human"] is False
    assert config["Experiment"]["save_model"] is False

    # Create the environment
    env = safety_gymnasium.make(
        config["game"]["env_id"],
        render_mode="rgb_array",
        max_episode_steps=config["Experiment"]["max_timesteps_per_game"]
    )

    # Get seed
    seed = get_seed(config)

    # Create the directories for files and plots of this experiment
    files_dir, plot_dir = get_result_dirs(argv[1], argv[0])

    # Create the SAC agent
    sac_agent = get_sac_agent(config, env, files_dir, only_test=True, seed=seed)

    # Create the experiment
    experiment = TestExperiment(env, sac_agent, config, files_dir, seed)

    # Run the experiment and time it
    start_experiment = time.time()
    experiment.agent_test()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment)
    print('\nTotal Experiment time: {}'.format(experiment_duration))

    # Save the experiment logs and plot them
    save_test_logs_and_plot(experiment, files_dir, plot_dir)
    # Save the rest experiment info
    experiment.save_info(files_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
