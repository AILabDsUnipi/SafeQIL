import os
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns


def save_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots=False):

    print("\nSaving logs and plots...")

    data_for_plots_to_return = {}

    if experiment.only_pretrain is False:

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

        ## Save additional train logs in files
        # Remove items where the list is empty
        train_logs = {
            k: v for k, v in experiment.train_logs_dict.items() if len(v) > 0
        }
        train_logs_avg_per_log_interval = {
            k: v for k, v in experiment.train_logs_avg_per_log_interval_dict.items() if len(v) > 0
        }
        # Create dataframes
        train_logs = pd.DataFrame(train_logs)
        train_logs_avg_per_log_interval = pd.DataFrame(train_logs_avg_per_log_interval)
        # Save to csv
        train_logs.to_csv(
            os.path.join(files_dir, 'train_logs.csv'), index=False
        )
        train_logs_avg_per_log_interval.to_csv(
            os.path.join(files_dir, 'train_logs_avg_per_log_interval.csv'), index=False
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

        # Plot for all steps.
        # The first element of each tuple is the ylabel,
        # the second is the plot title,
        # the third is the data to plot, and
        # the forth is the title of the file to be written.
        data_to_plot_train = group_metrics(experiment.train_logs_dict)

        x_axis = [(i + 1) for i in range(len(data_to_plot_train[0][2][0]))]
        for (plot_ylabel, plot_title, data, file_title) in data_to_plot_train:
            x_label = 'Episodes'
            plt.figure()
            plt.xlabel(x_label)
            plt.ylabel(plot_ylabel)
            plt.title(plot_title)

            # Plot max and min
            if len(data) > 1:
                plt.fill_between(x_axis, data[2], data[1], alpha=0.5)

            # Plot mean
            plt.plot(x_axis, data[0])

            # Save and close
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
        data_to_plot_train_per_log_interval = group_metrics(experiment.train_logs_avg_per_log_interval_dict)

        x_axis = [
            ((i + 1) * experiment.log_interval) if i > 0 else 1
            for i in range(len(data_to_plot_train_per_log_interval[0][2][0]))
        ]
        for (plot_ylabel, plot_title, data, file_title) in data_to_plot_train_per_log_interval:
            x_label = 'Episodes'
            plt.figure()
            plt.xlabel(x_label)
            plt.ylabel(plot_ylabel)
            plt.title(plot_title)

            # Plot max and min
            if len(data) > 1:
                plt.fill_between(x_axis, data[2], data[1], alpha=0.5)

            # Plot mean
            plt.plot(x_axis, data[0])

            # Save and close
            plt.savefig(os.path.join(plot_dir, file_title + "_per_log_interval.png"))
            plt.close()

            if return_data_for_plots is True:
                data_for_plots_to_return[file_title + "_per_log_interval"] = [
                    x_label, x_axis, plot_ylabel, plot_title, data
                ]

    if experiment.pretrain is True:

        ## Save pre-train logs
        # Remove items where the list is empty
        pretrain_logs = {
            k: v for k, v in experiment.pretrain_logs_dict.items() if len(v) > 0
        }
        # Create dataframe
        pretrain_logs = pd.DataFrame(pretrain_logs)
        # Save to csv
        pretrain_logs.to_csv(
            os.path.join(files_dir, 'pretrain_logs.csv'), index=False
        )

        # Plot pretraining metrics.
        # The first element of each tuple is the ylabel,
        # the second is the plot title,
        # the third is the data to plot, and
        # the forth is the title of the file to be written.
        data_to_plot_train = []
        for key, value in experiment.pretrain_logs_dict.items():
            if len(value) == 0:
                continue
            data_to_plot_train += [('Value', key, value, key)]

        x_axis = [(i + 1) for i in range(len(data_to_plot_train[0][2]))]
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


def group_metrics(metrics_dict):

    # Initialize a dictionary to hold the grouped data
    grouped_metrics = defaultdict(dict)

    # Iterate over your existing data and group mean, max, and min values
    for key, value in metrics_dict.items():
        if len(value) == 0:
            continue

        # Identify the metric type and base name
        if key.startswith('mean_'):
            metric_type = 'mean'
            metric_name = key[5:]  # Remove 'mean_' prefix
        elif key.startswith('max_'):
            metric_type = 'max'
            metric_name = key[4:]  # Remove 'max_' prefix
        elif key.startswith('min_'):
            metric_type = 'min'
            metric_name = key[4:]  # Remove 'min_' prefix
        else:
            # If the key doesn't have a prefix, handle accordingly
            metric_type = 'value'
            metric_name = key

        # Group the data
        grouped_metrics[metric_name][metric_type] = value

    # Now, prepare the data_to_plot list
    data_to_plot = []

    for metric_name, metrics in grouped_metrics.items():
        # Check if all three metrics are present
        if 'mean' in metrics and 'max' in metrics and 'min' in metrics:
            mean_values = metrics['mean']
            max_values = metrics['max']
            min_values = metrics['min']

            # Ensure that all lists have the same length
            assert len(mean_values) == len(max_values) == len(min_values)

            # Create a list containing the mean, max, and min values
            data_values = [mean_values, max_values, min_values]

        else:
            data_values = [metrics['value']]

        # Append to the list
        data_to_plot.append(
            ('Value', metric_name, data_values, metric_name)
        )

    return data_to_plot


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
        # The first element of each tuple is the data, and
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
                '  Q3: ' + ("-" if single_test else str(np.quantile(data, 0.75))) + '\n\n'
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

