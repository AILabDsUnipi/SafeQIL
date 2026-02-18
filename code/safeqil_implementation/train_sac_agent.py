# Add source directory to the system path
from utils.file_utils import resolve_source_path
resolve_source_path()

import os
import sys
import time
from datetime import timedelta

import safety_gymnasium

# Experiment
from experiments.train_sac_experiment import SACExperiment
from experiments.test_experiment import TestExperiment

# Utils
from utils.file_utils import get_config, get_result_dirs
from utils.agent_utils import get_sac_agent
from utils.plot_utils import save_logs_and_plot, save_test_logs_and_plot
from utils.exp_utils import get_train_seed, get_test_seed


def main(argv):

    # get configuration
    config = get_config(argv[0])

    # Check if the selected config file is suitable for the current experiment
    assert config["Experiment"]["test_model"] is False
    assert config["Experiment"]["save_model"] is True
    assert config["game"]["with_human"] is False

    ## Check other config settings
    # Check the consistency between 'log_interval' and 'test_every_episodes' consistency
    assert (config['Experiment']['log_interval'] == config['Experiment']['test_every_episodes'])
    # Check the consistency between 'max_timesteps_per_game' and 'update_every_steps'
    assert (config['Experiment']['max_timesteps_per_game'] >= config['SAC']['update_every_steps'])
    # Check the consistency between 'start_steps' and 'batch_size'
    assert (config['SAC']['start_steps'] >= config['SAC']['batch_size'])
    # Check the consistency between 'w_constraint_optimization' and 'w_dual_grad_desc'
    assert not (config['SAC']['w_dual_grad_desc'] is True and config['SAC']['w_constraint_optimization'] is False)
    # Check the consistency between 'w_kl_div' and 'w_q_values'
    assert not (config['SAC']['w_q_values'] is True and config['SAC']['w_kl_div'] is True)
    # Check the consistency between 'w_constraint_optimization' and 'w_kl_div'
    assert not (config['SAC']['w_kl_div'] is True and config['SAC']['w_constraint_optimization'] is False)
    # Check the consistency between 'w_entropy_in_constraint_policy_loss_term' and 'w_kl_div'
    assert not (
            config['SAC']['w_entropy_in_constraint_policy_loss_term'] is True and
            config['SAC']['w_kl_div'] is False
    )
    # Check the consistency between 'w_entropy_in_constraint_policy_loss_term' and 'adjust_entropy'
    assert not (
            config['SAC']['adjust_entropy'] is True and
            config['SAC']['w_entropy_in_constraint_policy_loss_term'] is False
    )
    # Check the consistency between 'pretrain' and 'w_constraint_optimization'
    assert not (config['SAC']['pretrain'] is True and config['SAC']['w_kl_div'] is False)
    # Check the consistency between 'pretrain' and 'only_pretrain'
    assert not (config['SAC']['only_pretrain'] is True and config['SAC']['pretrain'] is False)
    # Check the consistency between 'w_std_grads' and 'w_constraint_optimization'
    assert not (config['SAC']['w_std_grads'] is True and config['SAC']['w_kl_div'] is False)
    # Check the consistency between 'w_mse' and 'w_kl_div'
    assert not (config['SAC']['w_mse'] is True and config['SAC']['w_kl_div'] is False)
    # Check the consistency between 'w_constraint_optimization' and 'w_q_values'
    assert not (config['SAC']['w_q_values'] is True and config['SAC']['w_constraint_optimization'] is False)
    # Check the consistency between 'w_max_min' and 'w_q_values'
    assert not (config['SAC']['w_max_min'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_lower_bound' and 'w_q_values'
    assert not (config['SAC']['w_lower_bound'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_use_target_critic' and 'w_q_values'
    assert not (config['SAC']['w_use_target_critic'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_discriminator' and 'w_q_values'
    assert not (config['SAC']['w_discriminator'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_discriminator' and 'w_dual_grad_desc'
    assert not (config['SAC']['w_discriminator'] is True and config['SAC']['w_dual_grad_desc'] is True)
    # Check the consistency between 'w_discriminator_icrl_regularization' and 'w_discriminator'
    assert not (
            config['SAC']['w_discriminator_icrl_regularization'] is True and config['SAC']['w_discriminator'] is False
    )
    # Check the consistency between 'w_discriminator_dac_regularization' and 'w_discriminator'
    assert not (
            config['SAC']['w_discriminator_dac_regularization'] is True and config['SAC']['w_discriminator'] is False
    )
    # Check the consistency between 'w_discriminator_dac_regularization' and 'w_discriminator_icrl_regularization'
    assert not (
            config['SAC']['w_discriminator_dac_regularization'] is True and
            config['SAC']['w_discriminator_icrl_regularization'] is True
    )
    # Check the consistency between 'w_compute_analytically_min_dem_q_value' and 'w_q_values'
    assert not (
            config['SAC']['w_compute_analytically_min_dem_q_value'] is True and config['SAC']['w_q_values'] is False
    )
    # Check the consistency between 'w_compute_analytically_min_dem_q_value' and 'w_use_target_critic'
    assert not (
            config['SAC']['w_compute_analytically_min_dem_q_value'] is True and
            config['SAC']['w_use_target_critic'] is True
    )
    # Check the consistency between 'w_demonstrations_rl_term' and 'w_q_values'
    assert not (
            config['SAC']['w_demonstrations_rl_term'] is True and config['SAC']['w_q_values'] is False
    )
    # Check the consistency between 'w_demonstrations_next_actions_in_demonstrations_rl_term' and
    # 'w_demonstrations_rl_term'
    assert not (
            config['SAC']['w_demonstrations_next_actions_in_demonstrations_rl_term'] is True and
            config['SAC']['w_demonstrations_rl_term'] is False
    )
    # Check the consistency between 'w_entropy_in_demonstrations_rl_term' and
    # 'w_demonstrations_rl_term'
    assert not (
            config['SAC']['w_entropy_in_demonstrations_rl_term'] is True and
            config['SAC']['w_demonstrations_rl_term'] is False
    )
    # Check the consistency between 'w_compute_analytically_target_in_demonstrations_rl_term' and
    # 'w_demonstrations_rl_term'
    assert not (
            config['SAC']['w_compute_analytically_target_in_demonstrations_rl_term'] is True and
            config['SAC']['w_demonstrations_rl_term'] is False
    )
    # Check the consistency between 'w_compute_analytically_target_in_demonstrations_rl_term' and
    # 'w_demonstrations_next_actions_in_demonstrations_rl_term'
    assert not (
            config['SAC']['w_compute_analytically_target_in_demonstrations_rl_term'] is True and
            config['SAC']['w_demonstrations_next_actions_in_demonstrations_rl_term'] is True
    )
    # Check the consistency between 'w_compute_analytically_target_in_demonstrations_rl_term' and
    # 'w_entropy_in_demonstrations_rl_term'
    assert not (
            config['SAC']['w_compute_analytically_target_in_demonstrations_rl_term'] is True and
            config['SAC']['w_entropy_in_demonstrations_rl_term'] is True
    )
    # Check the consistency between 'w_ood_rl_term' and 'w_q_values'
    assert not (
            config['SAC']['w_ood_rl_term'] is True and config['SAC']['w_q_values'] is False
    )
    # Check the consistency between 'w_ood_rl_term' and 'w_discriminator'
    assert not (
            config['SAC']['w_ood_rl_term'] is True and config['SAC']['w_discriminator'] is False
    )
    # Check the consistency between 'w_discriminator_rewards_in_ood_rl_term' and 'w_ood_rl_term'
    assert not (
            config['SAC']['w_discriminator_rewards_in_ood_rl_term'] is True and
            config['SAC']['w_ood_rl_term'] is False
    )
    # Check the consistency between 'w_discriminator_discounted_rewards_in_ood_rl_term' and 'w_ood_rl_term'
    assert not (
            config['SAC']['w_discriminator_discounted_rewards_in_ood_rl_term'] is True and
            config['SAC']['w_ood_rl_term'] is False
    )
    # Check the consistency between 'w_discriminator_discounted_rewards_in_ood_rl_term' and
    # 'w_discriminator_rewards_in_ood_rl_term'
    assert not (
            config['SAC']['w_discriminator_discounted_rewards_in_ood_rl_term'] is True and
            config['SAC']['w_discriminator_rewards_in_ood_rl_term'] is True
    )
    # Check the consistency between 'w_entropy_in_ood_rl_term' and 'w_ood_rl_term'
    assert not (
            config['SAC']['w_entropy_in_ood_rl_term'] is True and
            config['SAC']['w_ood_rl_term'] is False
    )
    # Check the consistency between 'w_gail_sac' and 'w_q_values'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_gail_sac' and 'w_discriminator'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_discriminator'] is False)
    # Check the consistency between 'w_gail_sac' and 'w_discriminator'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_actions_in_discriminator'] is False)
    # Check the consistency between 'w_gail_sac' and 'w_max_min'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_max_min'] is True)
    # Check the consistency between 'w_gail_sac' and 'w_compute_analytically_min_dem_q_value'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_compute_analytically_min_dem_q_value'] is True)
    # Check the consistency between 'w_gail_sac' and 'w_lower_bound'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_lower_bound'] is True)
    # Check the consistency between 'w_gail_sac' and 'w_use_target_critic'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_use_target_critic'] is True)
    # Check the consistency between 'w_gail_sac' and 'w_demonstrations_rl_term'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_demonstrations_rl_term'] is True)
    # Check the consistency between 'w_gail_sac' and 'w_ood_rl_term'
    assert not (config['SAC']['w_gail_sac'] is True and config['SAC']['w_ood_rl_term'] is True)
    # Check the consistency between 'w_expectile_loss' and 'w_max_min'
    assert not (config['SAC']['w_expectile_loss'] is True and config['SAC']['w_max_min'] is False)
    # Check the consistency between 'w_threshold_in_discriminator_weights' and 'w_discriminator'
    assert not (
            config['SAC']['w_threshold_in_discriminator_weights'] is True and config['SAC']['w_discriminator'] is False
    )
    # Check the consistency between 'w_threshold_in_discriminator_weights' and 'w_gail_sac'
    assert not (config['SAC']['w_threshold_in_discriminator_weights'] is True and config['SAC']['w_gail_sac'] is True)
    # Check the consistency between 'w_closest_state_min' and 'w_q_values'
    assert not (config['SAC']['w_closest_state_min'] is True and config['SAC']['w_q_values'] is False)
    # Check the consistency between 'w_closest_state_min' and 'w_use_target_critic'
    assert not (config['SAC']['w_closest_state_min'] is True and config['SAC']['w_use_target_critic'] is False)
    # Check the consistency between 'w_closest_state_min' and 'w_compute_analytically_min_dem_q_value'
    assert not (
            config['SAC']['w_closest_state_min'] is True and
            config['SAC']['w_compute_analytically_min_dem_q_value'] is True
    )
    # Check the consistency between 'w_closest_state_min' and 'w_gail_sac'
    assert not (config['SAC']['w_closest_state_min'] is True and config['SAC']['w_gail_sac'] is True)
    # Check the consistency between 'closest_state_min_func', 'w_closest_state_min', 'w_discriminator', and
    # 'w_actions_in_discriminator'
    assert not (
            config['SAC']['w_closest_state_min'] is True and
            config['SAC']['closest_state_min_func'] == 'cosine_sim_w_discr_embed' and
            (config['SAC']['w_discriminator'] is False or config['SAC']['w_actions_in_discriminator'] is True)
    )
    # Check the consistency between the 'w_target_discriminator' and 'w_discriminator'
    assert not (config['SAC']['w_target_discriminator'] is True and config['SAC']['w_discriminator'] is False)

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

    # Create the SAC agent
    sac_agent = get_sac_agent(config, env, seed=seed)

    # Create the experiment
    experiment = SACExperiment(env, sac_agent, config, files_dir, seed)

    # Run the experiment and time it
    start_experiment = time.time()
    experiment.train()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment)

    # Save the experiment logs and plot them
    save_logs_and_plot(experiment, files_dir, plot_dir)
    # Save the rest experiment info
    experiment.save_info(files_dir)

    # Delete experiment and agent to save memory
    del sac_agent, experiment

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

    # Create the test SAC agent
    test_sac_agent = get_sac_agent(config, env, only_test=True, seed=test_seed)

    # Create the test experiment
    test_experiment = TestExperiment(env, test_sac_agent, config, test_files_dir, test_seed)

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
