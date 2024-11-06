import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns


def save_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots=False):

    print("\nSaving logs and plots...")

    data_for_plots_to_return = {}

    # Save test logs in files
    pd.DataFrame(experiment.test_step_list, columns=['Steps']).to_csv(
        os.path.join(files_dir, 'test_steps.csv'), index=False
    )
    pd.DataFrame(experiment.test_step_list_avg_per_log_interval, columns=['Steps']).to_csv(
        os.path.join(files_dir, 'test_steps_avg_per_log_interval.csv'), index=False
    )
    pd.DataFrame(experiment.test_reward_list, columns=['Reward']).to_csv(
        os.path.join(files_dir, 'test_rewards.csv'), index=False
    )
    pd.DataFrame(experiment.test_reward_list_avg_per_log_interval, columns=['Reward']).to_csv(
        os.path.join(files_dir, 'test_rewards_avg_per_log_interval.csv'), index=False
    )
    for constraint_type in experiment.constraint_types:
        constraint_name = constraint_type.replace('cost_', '')
        pd.DataFrame(
            experiment.test_num_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_num_constraint_{constraint_name}.csv'), index=False
        )
        pd.DataFrame(
            experiment.test_num_constraint_violation_per_log_interval[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_num_constraint_{constraint_name}_per_log_interval.csv'), index=False
        )
        pd.DataFrame(
            experiment.test_freq_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_freq_constraint_{constraint_name}.csv'), index=False
        )
        pd.DataFrame(
            experiment.test_freq_constraint_violation_per_log_interval[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_freq_constraint_{constraint_name}_per_log_interval.csv'), index=False
        )

    # Save train logs in files
    pd.DataFrame(experiment.step_list, columns=['Steps']).to_csv(
        os.path.join(files_dir, 'steps.csv'), index=False
    )
    pd.DataFrame(experiment.step_list_avg_per_log_interval, columns=['Steps']).to_csv(
        os.path.join(files_dir, 'steps_avg_per_log_interval.csv'), index=False
    )
    pd.DataFrame(experiment.reward_list, columns=['Reward']).to_csv(
        os.path.join(files_dir, 'rewards.csv'), index=False
    )
    pd.DataFrame(experiment.reward_list_avg_per_log_interval, columns=['Reward']).to_csv(
        os.path.join(files_dir, 'rewards_avg_per_log_interval.csv'), index=False
    )
    for constraint_type in experiment.constraint_types:
        constraint_name = constraint_type.replace('cost_', '')
        pd.DataFrame(
            experiment.num_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'num_constraint_{constraint_name}.csv'), index=False
        )
        pd.DataFrame(
            experiment.num_constraint_violation_per_log_interval[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'num_constraint_{constraint_name}_per_log_interval.csv'), index=False
        )
        pd.DataFrame(
            experiment.freq_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'freq_constraint_{constraint_name}.csv'), index=False
        )
        pd.DataFrame(
            experiment.freq_constraint_violation_per_log_interval[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'freq_constraint_{constraint_name}_per_log_interval.csv'), index=False
        )

    # Save additional train logs in files
    pd.DataFrame(experiment.actor_loss_list, columns=['Actor Loss']).to_csv(
        os.path.join(files_dir, 'actor_loss.csv'), index=False
    )
    pd.DataFrame(experiment.critic_loss_list, columns=['Critic Loss']).to_csv(
        os.path.join(files_dir, 'critic_loss.csv'), index=False
    )
    pd.DataFrame(experiment.ent_coef_loss_list, columns=['Entropy Coefficient Loss']).to_csv(
        os.path.join(files_dir, 'ent_coef_loss.csv'), index=False
    )
    pd.DataFrame(experiment.entr_coef_list, columns=['Entropy Coefficient']).to_csv(
        os.path.join(files_dir, 'entr_coef.csv'), index=False
    )
    pd.DataFrame(experiment.actor_loss_avg_per_log_interval, columns=['Actor Loss Avg Per Log Interval']).to_csv(
        os.path.join(files_dir, 'actor_loss_avg_per_log_interval.csv'), index=False
    )
    pd.DataFrame(experiment.critic_loss_avg_per_log_interval, columns=['Critic Loss Avg Per Log Interval']).to_csv(
        os.path.join(files_dir, 'critic_loss_avg_per_log_interval.csv'), index=False
    )
    pd.DataFrame(experiment.ent_coef_loss_avg_per_log_interval, columns=['Entropy Coefficient Loss Avg Per Log Interval']).to_csv(
        os.path.join(files_dir, 'ent_coef_loss_avg_per_log_interval.csv'), index=False
    )
    pd.DataFrame(experiment.entr_coef_avg_per_log_interval, columns=['Entropy Coefficient Avg Per Log Interval']).to_csv(
        os.path.join(files_dir, 'entr_coef_avg_per_log_interval.csv'), index=False
    )
    if experiment.clip_grad_norm is True:
        pd.DataFrame(experiment.grad_norm_clipped_list, columns=['Gradient norm clipped']).to_csv(
            os.path.join(files_dir, 'grad_norm_clipped.csv'), index=False
        )
        pd.DataFrame(
            experiment.grad_norm_clipped_avg_per_log_interval,
            columns=['Gradient norm clipped Avg Per Log Interval']
        ).to_csv(
            os.path.join(files_dir, 'grad_norm_clipped_avg_per_log_interval.csv'), index=False
        )
    if experiment.w_constraint_optimization is True:
        pd.DataFrame(
            experiment.constraint_policy_loss_term_value_list,
            columns=['Constraint Policy Loss Term Value']
        ).to_csv(
            os.path.join(files_dir, 'constraint_policy_loss_term_value.csv'), index=False
        )
        pd.DataFrame(
            experiment.constraint_lambda_loss_value_list,
            columns=['Constraint Lambda Loss Value']
        ).to_csv(
            os.path.join(files_dir, 'constraint_lambda_loss_value.csv'), index=False
        )
        pd.DataFrame(
            experiment.policy_loss_value_wo_constraint_term_list,
            columns=['Policy Loss Value Without Constraint Term']
        ).to_csv(
            os.path.join(files_dir, 'policy_loss_value_wo_constraint_term.csv'), index=False
        )
        pd.DataFrame(
            experiment.constraint_lambda_list,
            columns=['Constraint Lambda']
        ).to_csv(
            os.path.join(files_dir, 'constraint_lambda.csv'), index=False
        )
        pd.DataFrame(
            experiment.constraint_policy_loss_term_value_avg_per_log_interval,
            columns=['Constraint Policy Loss Term Value Avg Per Log Interval']
        ).to_csv(
            os.path.join(files_dir, 'constraint_policy_loss_term_value_avg_per_log_interval.csv'), index=False
        )
        pd.DataFrame(
            experiment.constraint_lambda_loss_value_avg_per_log_interval,
            columns=['Constraint Lambda Loss Value Avg Per Log Interval']
        ).to_csv(
            os.path.join(files_dir, 'constraint_lambda_loss_value_avg_per_log_interval.csv'), index=False
        )
        pd.DataFrame(
            experiment.policy_loss_value_wo_constraint_term_avg_per_log_interval,
            columns=['Policy Loss Value Without Constraint Term Avg Per Log Interval']
        ).to_csv(
            os.path.join(files_dir, 'policy_loss_value_wo_constraint_term_avg_per_log_interval.csv'), index=False
        )
        pd.DataFrame(
            experiment.constraint_lambda_avg_per_log_interval,
            columns=['Constraint Lambda Avg Per Log Interval']
        ).to_csv(
            os.path.join(files_dir, 'constraint_lambda_avg_per_log_interval.csv'), index=False
        )
        if experiment.pretrain is True:
            pd.DataFrame(
                experiment.pretrain_mse_losses,
                columns=['Pretrain MSE Loss']
            ).to_csv(
                os.path.join(files_dir, 'pretrain_mse_loss.csv'), index=False
            )
            pd.DataFrame(
                experiment.pretrain_nll_losses,
                columns=['Pretrain NLL Loss']
            ).to_csv(
                os.path.join(files_dir, 'pretrain_nll_loss.csv'), index=False
            )
            pd.DataFrame(
                experiment.pretrain_losses,
                columns=['Pretrain Loss']
            ).to_csv(
                os.path.join(files_dir, 'pretrain_loss.csv'), index=False
            )
            pd.DataFrame(
                experiment.pretrain_log_probs,
                columns=['Pretrain Log Probs']
            ).to_csv(
                os.path.join(files_dir, 'pretrain_log_probs.csv'), index=False
            )
            pd.DataFrame(
                experiment.pretrain_probs,
                columns=['Pretrain Probs']
            ).to_csv(
                os.path.join(files_dir, 'pretrain_probs.csv'), index=False
            )
            if experiment.clip_grad_norm is True:
                pd.DataFrame(
                    experiment.pretrain_grad_norms_clipped,
                    columns=['Pretrain Gradient Norms Clipped']
                ).to_csv(
                    os.path.join(files_dir, 'pretrain_grad_norms_clipped.csv'), index=False
                )
    for model_type in experiment.episodes_model_saved:
        pd.DataFrame(
            experiment.episodes_model_saved[model_type],
            columns=[f'Episodes {model_type} Model Saved']
        ).to_csv(
            os.path.join(files_dir, f'episodes_{model_type}_model_saved.csv'), index=False
        )

    # Check the consistency of the test and train results
    assert len(experiment.test_reward_list_avg_per_log_interval) > 0, "No test results provided."
    assert len(experiment.reward_list_avg_per_log_interval) == \
           len(experiment.test_reward_list_avg_per_log_interval) == \
           len(experiment.step_list_avg_per_log_interval) == \
           len(experiment.test_step_list_avg_per_log_interval), \
        "Inconsistency among results concerning the number of test."
    for constraint_type in experiment.test_num_constraint_violation_per_log_interval:
        constraint_name = constraint_type.replace('cost_', '')
        assert len(experiment.test_reward_list_avg_per_log_interval) == \
               len(experiment.num_constraint_violation_per_log_interval[constraint_type]) == \
               len(experiment.test_num_constraint_violation_per_log_interval[constraint_type]) == \
               len(experiment.freq_constraint_violation_per_log_interval[constraint_type]) == \
               len(experiment.test_freq_constraint_violation_per_log_interval[constraint_type]), \
            f"{constraint_name} constraint results are inconsistent concerning the number of test."

    ## Plot metrics for per_log_interval
    # The first element of each tuple is the ylabel,
    # the second is the plot title,
    # the third is the data to plot, and
    # the forth is the title of the file to be written.
    data_to_plot = [
        (
            'Episodic Reward',
            'Reward',
            (experiment.reward_list_avg_per_log_interval, experiment.test_reward_list_avg_per_log_interval),
            'reward_avg_per_log_interval'
        )
    ]
    for constraint_type in experiment.constraint_types:
        constraint_name = constraint_type.replace('cost_', '')
        data_to_plot += [
            (
                'Number of Violations',
                f"Number of '{constraint_name}' constraint violations",
                (
                    experiment.num_constraint_violation_per_log_interval[constraint_type],
                    experiment.test_num_constraint_violation_per_log_interval[constraint_type]
                ),
                f'{constraint_name}_num_constraint_violations_avg_per_log_interval'
            ),
            (
                'Frequency of Violations',
                f"Frequency of '{constraint_name}' constraint violations",
                (
                    experiment.freq_constraint_violation_per_log_interval[constraint_type],
                    experiment.test_freq_constraint_violation_per_log_interval[constraint_type]
                ),
                f'{constraint_name}_freq_constraint_violations_avg_per_log_interval')
        ]

    plot_xlabel = 'Episodes'
    plot_legend = ('Train', 'Test')
    x_axis = [
        (i + 1) * experiment.log_interval if i > 0 else 1
        for i in range(len(experiment.reward_list_avg_per_log_interval))
    ]
    for (plot_ylabel, plot_title, data, file_title) in data_to_plot:
        plt.figure()
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.title(plot_title)
        plt.plot(x_axis, data[0])  # Train
        plt.plot(x_axis, data[1])  # Test
        plt.gca().legend(plot_legend)
        plt.savefig(os.path.join(plot_dir, file_title + "_per_log_interval.png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title + "_per_log_interval"] = [
                plot_xlabel, plot_ylabel, plot_title, plot_legend, data, x_axis
            ]

    ## Plot train metrics for per_log_interval and all steps

    # Check the consistency of training lists
    assert len(experiment.actor_loss_list) == \
           len(experiment.critic_loss_list) == \
           len(experiment.ent_coef_loss_list) == \
           len(experiment.entr_coef_list), \
            "Inconsistency among training lists."
    if experiment.w_constraint_optimization is True:
        assert len(experiment.actor_loss_list) == \
               len(experiment.constraint_policy_loss_term_value_list) == \
               len(experiment.constraint_lambda_loss_value_list) == \
               len(experiment.policy_loss_value_wo_constraint_term_list) == \
               len(experiment.constraint_lambda_list), \
            "Inconsistency among constraint training lists."
    if experiment.clip_grad_norm is True:
        assert len(experiment.actor_loss_list) == \
               len(experiment.grad_norm_clipped_list), \
            "Inconsistency for gradient norm clipped list."

    # Check the consistency of training lists for avg_per_log_interval
    assert len(experiment.actor_loss_avg_per_log_interval) == \
           len(experiment.critic_loss_avg_per_log_interval) == \
           len(experiment.ent_coef_loss_avg_per_log_interval) == \
           len(experiment.entr_coef_avg_per_log_interval), \
        "Inconsistency among training lists for avg_per_log_interval."
    if experiment.w_constraint_optimization is True:
        assert len(experiment.actor_loss_avg_per_log_interval) == \
               len(experiment.constraint_policy_loss_term_value_avg_per_log_interval) == \
               len(experiment.constraint_lambda_loss_value_avg_per_log_interval) == \
               len(experiment.policy_loss_value_wo_constraint_term_avg_per_log_interval) == \
               len(experiment.constraint_lambda_avg_per_log_interval), \
            "Inconsistency among constraint training lists for avg_per_log_interval."
    if experiment.clip_grad_norm is True:
        assert len(experiment.actor_loss_avg_per_log_interval) == \
               len(experiment.grad_norm_clipped_avg_per_log_interval), \
            "Inconsistency for gradient norm clipped for avg_per_log_interval."

    # Plot for all steps.
    # The first element of each tuple is the ylabel,
    # the second is the plot title,
    # the third is the data to plot, and
    # the forth is the title of the file to be written.
    data_to_plot_train = [
        ('Loss', 'Actor Loss', experiment.actor_loss_list, 'actor_loss'),
        ('Loss', 'Critic Loss', experiment.critic_loss_list, 'critic_loss'),
        (
            'Loss',
            'Entropy Coefficient Loss',
            experiment.ent_coef_loss_list,
            'ent_coef_loss'
        ),
        ('Value', 'Entropy Coefficient', experiment.entr_coef_list, 'entr_coef')
    ]
    if experiment.w_constraint_optimization is True:
        data_to_plot_train += [
            (
                'Loss',
                'Constraint Policy Loss',
                experiment.constraint_policy_loss_term_value_list,
                'constraint_policy_loss_term_value'
            ),
            (
                'Loss',
                'Constraint Lambda Loss',
                experiment.constraint_lambda_loss_value_list,
                'constraint_lambda_loss_value'
            ),
            (
                'Loss',
                'Policy Loss without Constraint Term',
                experiment.policy_loss_value_wo_constraint_term_list,
                'policy_loss_value_wo_constraint_term'
            ),
            (
                'Value',
                'Constraint Lambda',
                experiment.constraint_lambda_list,
                'constraint_lambda'
            )
        ]
    if experiment.clip_grad_norm is True:
        data_to_plot_train += [
            (
                'Value',
                'Gradient norm clipped',
                experiment.grad_norm_clipped_list,
                'grad_norm_clipped_list'
            )
        ]
    x_axis = [(i + 1) for i in range(len(experiment.actor_loss_list))]
    for (plot_ylabel, plot_title, data, file_title) in data_to_plot_train:
        x_label = 'Episodes'
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(plot_ylabel)
        plt.title(plot_title)
        plt.plot(x_axis, data)
        plt.savefig(os.path.join(plot_dir, file_title + "_all_episodes.png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title + "_all_episodes"] = [
                x_label, x_axis, plot_ylabel, plot_title, data
            ]

    # Plot for per_log_interval
    # The first element of each tuple is the ylabel,
    # the second is the plot title,
    # the third is the data to plot, and
    # the forth is the title of the file to be written.
    data_to_plot_train_per_log_interval = [
        (
            'Loss',
            'Actor Loss avg per Log Interval',
            experiment.actor_loss_avg_per_log_interval,
            'actor_loss_avg_per_log_interval'
        ),
        (
            'Loss',
            'Critic Loss avg per Log Interval',
            experiment.critic_loss_avg_per_log_interval,
            'critic_loss_avg_per_log_interval'
        ),
        (
            'Loss',
            'Entropy Coefficient Loss avg per Log Interval',
            experiment.ent_coef_loss_avg_per_log_interval,
            'ent_coef_loss_avg_per_log_interval'
        ),
        (
            'Value',
            'Entropy Coefficient avg per Log Interval',
            experiment.entr_coef_avg_per_log_interval,
            'entr_coef_avg_per_log_interval'
        )
    ]
    if experiment.w_constraint_optimization is True:
        data_to_plot_train_per_log_interval += [
            (
                'Loss',
                'Constraint Policy Loss avg per Log Interval',
                experiment.constraint_policy_loss_term_value_avg_per_log_interval,
                'constraint_policy_loss_term_value_avg_per_log_interval'
            ),
            (
                'Loss',
                'Constraint Lambda Loss avg per Log Interval',
                experiment.constraint_lambda_loss_value_avg_per_log_interval,
                'constraint_lambda_loss_value_avg_per_log_interval'
            ),
            (
                'Loss',
                'Policy Loss without Constraint Term avg per Log Interval',
                experiment.policy_loss_value_wo_constraint_term_avg_per_log_interval,
                'policy_loss_value_wo_constraint_term_avg_per_log_interval'
            ),
            (
                'Value',
                'Constraint Lambda avg per Log Interval',
                experiment.constraint_lambda_avg_per_log_interval,
                'constraint_lambda_avg_per_log_interval'
            )
        ]
    if experiment.clip_grad_norm is True:
        data_to_plot_train += [
            (
                'Value',
                'Gradient norm clipped avg per Log Interval',
                experiment.grad_norm_clipped_avg_per_log_interval,
                'grad_norm_clipped_avg_per_log_interval'
            )
        ]
    x_axis = [
        ((i + 1) * experiment.log_interval) if i > 0 else 1
        for i in range(len(experiment.actor_loss_avg_per_log_interval))
    ]
    for (plot_ylabel, plot_title, data, file_title) in data_to_plot_train_per_log_interval:
        x_label = 'Episodes'
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(plot_ylabel)
        plt.title(plot_title)
        plt.plot(x_axis, data)
        plt.savefig(os.path.join(plot_dir, file_title + "_per_log_interval.png"))
        plt.close()
        if return_data_for_plots is True:
            data_for_plots_to_return[file_title + "_per_log_interval"] = [
                x_label, x_axis, plot_ylabel, plot_title, data
            ]

    if experiment.pretrain is True:
        # Check the consistency of pretraining lists
        assert len(experiment.pretrain_mse_losses) == \
               len(experiment.pretrain_nll_losses) == \
               len(experiment.pretrain_losses) == \
               len(experiment.pretrain_log_probs) == \
               len(experiment.pretrain_probs), \
            "Inconsistency among pretraining lists."
        if experiment.clip_grad_norm is True:
            assert len(experiment.pretrain_mse_losses) == \
                   len(experiment.pretrain_grad_norms_clipped), \
                "Inconsistency for pretraining gradient norm clipped list."

        # Plot pretraining metrics.
        # The first element of each tuple is the ylabel,
        # the second is the plot title,
        # the third is the data to plot, and
        # the forth is the title of the file to be written.
        data_to_plot_train = [
            ('Loss', 'MSE Loss', experiment.pretrain_mse_losses, 'pretrain_mse_loss'),
            ('Loss', 'NLL Loss', experiment.pretrain_nll_losses, 'pretrain_nll_loss'),
            ('Loss', 'Loss', experiment.pretrain_losses, 'pretrain_loss'),
            ('Value', 'Log Probs', experiment.pretrain_log_probs, 'pretrain_log_probs'),
            ('Value', 'Probs', experiment.pretrain_probs, 'pretrain_probs')
        ]
        if experiment.clip_grad_norm is True:
            data_to_plot_train += [
                (
                    'Value',
                    'Gradient norm clipped',
                    experiment.pretrain_grad_norms_clipped,
                    'pretrain_grad_norms_clipped'
                )
            ]
        x_axis = [(i + 1) for i in range(len(experiment.pretrain_mse_losses))]
        for (plot_ylabel, plot_title, data, file_title) in data_to_plot_train:
            x_label = 'Epochs'
            plt.figure()
            plt.xlabel(x_label)
            plt.ylabel(plot_ylabel)
            plt.title(plot_title)
            plt.plot(x_axis, data)
            plt.savefig(os.path.join(plot_dir, file_title + ".png"))
            plt.close()
            if return_data_for_plots is True:
                data_for_plots_to_return[file_title] = [
                    x_label, x_axis, plot_ylabel, plot_title, data
                ]

    print(f"Logs and plots saved in '{files_dir}' and '{plot_dir}' successfully!")

    return data_for_plots_to_return


def save_test_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots=False):

    print("\nSaving test logs and plots...")

    # Save test logs in files
    pd.DataFrame(experiment.test_step_list, columns=['Steps']).to_csv(
        os.path.join(files_dir, 'test_steps.csv'), index=False
    )
    pd.DataFrame(experiment.test_reward_list, columns=['Reward']).to_csv(
        os.path.join(files_dir, 'test_rewards.csv'), index=False
    )
    for constraint_type in experiment.constraint_types:
        constraint_name = constraint_type.replace('cost_', '')
        pd.DataFrame(
            experiment.test_num_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_num_constraint_{constraint_name}.csv'), index=False
        )
        pd.DataFrame(
            experiment.test_freq_constraint_violation_list[constraint_type],
            columns=[f'Constraint {constraint_name}']
        ).to_csv(
            os.path.join(files_dir, f'test_freq_constraint_{constraint_name}.csv'), index=False
        )

    # Check the consistency of the test results
    assert len(experiment.test_reward_list) > 0, "No test results provided."
    assert len(experiment.test_reward_list) == \
           len(experiment.test_step_list), "Inconsistency among results concerning the number of test."
    for constraint_type in experiment.test_num_constraint_violation_list:
        constraint_name = constraint_type.replace('cost_', '')
        assert len(experiment.test_reward_list) == \
               len(experiment.test_num_constraint_violation_list[constraint_type]) == \
               len(experiment.test_freq_constraint_violation_list[constraint_type]), \
               f"{constraint_name} constraint results are inconsistent concerning the number of test."

    single_test = len(experiment.test_reward_list) == 1

    # Write results statistics in txt file
    with open(os.path.join(files_dir, 'stats_info.txt'), 'w') as stats_info:
        # The First element of each tuple is the data, and
        # the second is the type of data
        data_to_write = [
            (experiment.test_reward_list, 'Reward'),
            (experiment.test_step_list, 'Steps')
        ]
        for constraint_type in experiment.constraint_types:
            constraint_name = constraint_type.replace('cost_', '')
            data_to_write += [
                (experiment.test_num_constraint_violation_list[constraint_type],
                 f"Number of '{constraint_name}' constraint violations"),
                (experiment.test_freq_constraint_violation_list[constraint_type],
                 f"Frequency of '{constraint_name}' constraint violations")
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
    use_sliding_window = len(experiment.test_reward_list) > experiment.test_window_size_moving_avg
    if not single_test:

        # The first element of each tuple is the ylabel,
        # the second is the plot title,
        # the third is the data to plot, and
        # the forth is the title of the file to be written.
        data_to_plot = [
            ('Reward', 'Game reward', experiment.test_reward_list, 'reward')
        ]
        for constraint_type in experiment.constraint_types:
            constraint_name = constraint_type.replace('cost_', '')
            data_to_plot += [
                (
                    'Number of Violations',
                    f"Number of '{constraint_name}' constraint violations",
                    experiment.test_num_constraint_violation_list[constraint_type],
                    f'{constraint_name}_num_constraint_violations'
                ),
                (
                    'Frequency of Violations',
                    f"Frequency of '{constraint_name}' constraint violations",
                    experiment.test_freq_constraint_violation_list[constraint_type],
                    f'{constraint_name}_freq_constraint_violations')
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

    print(f"Test logs and plots saved in '{files_dir}' and '{plot_dir}' successfully!")

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

