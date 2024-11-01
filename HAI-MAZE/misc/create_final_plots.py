import os
import sys
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_csv_data(csv_path):
    df = pd.read_csv(csv_path)

    return df.iloc[:, 0].values

def read_data(path_to_data, is_pickle, data_types):
    # Read pickle file
    if is_pickle:
        with open(path_to_data, 'rb') as pickle_file:
            data_for_plots = pickle.load(pickle_file)
    # Read each csv file
    else:
        data_for_plots = {}
        for data_type in data_types:
            if data_type == 'rewards_avg_per_log_interval':
                csv_path = os.path.join(path_to_data, 'pure_rewards_test_avg_per_test.csv')
            elif data_type == 'ball_only_at_the_up_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq):
                csv_path = os.path.join(path_to_data, 'test_ball_only_at_the_up_side_wrt_hole_num_constraint_violations_avg_per_test.csv')
            elif data_type == 'ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_avg_per_log_interval':
                csv_path = os.path.join(path_to_data, 'test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violations_avg_per_test.csv')
            elif data_type == 'ball_only_at_the_right_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq):
                csv_path = os.path.join(path_to_data, 'test_ball_only_at_the_right_side_wrt_hole_num_constraint_violations_avg_per_test.csv')
            elif data_type == 'ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_avg_per_log_interval':
                csv_path = os.path.join(path_to_data, 'test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violations_avg_per_test.csv')
            elif data_type == 'ball_not_in_circle_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq):
                csv_path = os.path.join(path_to_data, 'test_ball_not_in_circle_num_constraint_violations_avg_per_test.csv')
            elif data_type == 'ball_not_in_circle_freq_constraint_violated_avg_per_log_interval':
                csv_path = os.path.join(path_to_data, 'test_ball_not_in_circle_freq_constraint_violations_avg_per_test.csv')
            else:
                raise ValueError("The current data type: '{}' is not supported.".format(data_type))

            if os.path.exists(csv_path):
                data_for_plot = get_csv_data(csv_path)
                data_for_plots[data_type] = data_for_plot

    return data_for_plots


def final_plots(paths_to_data, method_names, dir_to_save, cut_plots_at_episode):

    # Check if there is at least one provided path
    assert len(paths_to_data) > 0, \
        "The provided list of paths to data is empty! \n'paths_to_data'={}".format(paths_to_data)

    # Check the consistency between the provided 'method_names' and 'paths_to_data'
    assert len(method_names) == len(paths_to_data), \
        ("Inconsistency between 'method_names' and 'paths_to_data': " +
         "\nlen(method_names)={}, len(paths_to_data)={}".format(len(method_names), len(paths_to_data)) +
         "\nmethod_names={}, paths_to_data={}".format(method_names, paths_to_data))

    # Check if the directory to save results already exists and have elements
    assert (os.path.exists(dir_to_save) is False or
           (os.path.exists(dir_to_save) is True and len(os.listdir(dir_to_save)) == 0)), \
        "The provided 'dir_to_save' already exists and has elements!"
    # Create the 'dir_to_save' if not already exists
    if os.path.exists(dir_to_save) is False:
        os.mkdir(dir_to_save)
    # Check the validity of 'cut_plots_at_episode'
    if cut_plots_at_episode != 'None':
        try:
            cut_plots_at_episode = int(cut_plots_at_episode)
        except (Exception,) as e:
            assert False, "'cut_at_episode' argument is not a number not 'None'."

    # Find if the results refer to multi-train-test or single-train-test
    _, file_extension = os.path.splitext(paths_to_data[0])
    is_pickle = file_extension == '.pickle'
    # If the first provided path is not pickle, it should be a directory.
    if is_pickle is False:
        assert os.path.isdir(paths_to_data[0]), \
            "The provided 'paths_to_data[0]': '{}' is not pickle file nor a directory!".format(paths_to_data[0])
    # Check 'is_pickle' consistency across all paths
    for path_to_data in paths_to_data:
        _, file_extension_new = os.path.splitext(path_to_data)
        is_pickle_new = file_extension_new == '.pickle'
        assert is_pickle == is_pickle_new, \
            "Inconsistency between 'is_pickle':{} and 'is_pickle_new':{} for paths '{}' and '{}'." \
                .format(is_pickle, is_pickle_new, paths_to_data[0], path_to_data)
        # If the current provided path is not pickle, it should be a directory.
        if is_pickle_new is False:
            assert os.path.isdir(path_to_data), \
                "The current provided 'path_to_data': '{}' is not pickle file nor a directory!".format(path_to_data)

    data_types = ['rewards_avg_per_log_interval',
                  'ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg_per_log_interval',
                  'ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_avg_per_log_interval',
                  'ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg_per_log_interval',
                  'ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_avg_per_log_interval',
                  'ball_not_in_circle_num_constraint_violated_avg_per_log_interval',
                  'ball_not_in_circle_freq_constraint_violated_avg_per_log_interval',
                  'total_num_freq_constraint_violated_avg_per_log_interval',
                  'total_freq_constraint_violated_avg_per_log_interval']
    plot_xlabel = ['Episodes x40'] * len(data_types)
    plot_ylabel = ['Reward',
                   'Number of H Violations',
                   'Frequency of H Violations',
                   'Number of V Violations',
                   'Frequency of V Violations',
                   'Number of C Violations',
                   'Frequency of C Violations',
                   'Number of Violations',
                   'Frequency of Violations']
    plot_legend_loc = ['lower center',
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None]
    bottom_ylim = [None,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0]
    colors = ["black", "orange", "green", "red", "purple"]

    # Collect data for plots in single dictionary
    data_dict = {data_type: [] for data_type in data_types}
    for path_idx, path_to_data in enumerate(paths_to_data):
        data_for_plots = read_data(path_to_data, is_pickle, data_types)
        for data_type in data_types:
            if data_type in data_for_plots.keys():
                if is_pickle:
                    data_dict[data_type].append({'method_name': method_names[path_idx],
                                                 'mean': data_for_plots[data_type]['test']['mean'],
                                                 'std': data_for_plots[data_type]['test']['std']})
                else:
                    data_dict[data_type].append({'method_name': method_names[path_idx],
                                                 'data': data_for_plots[data_type]})
                # Save memory
                #del data_for_plots[data_type]
            elif "total" in data_type and "constraint" and is_pickle:

                if "num" in data_type:
                    num_or_freq = "num"
                elif "freq" in data_type:
                    num_or_freq = "freq"
                else:
                    raise ValueError

                total_num_freq_constraint_violated_avg_per_log_interval = None
                if 'ball_only_at_the_up_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq) in data_for_plots.keys():
                    min_length = min([len(list_) for list_ in data_for_plots['ball_only_at_the_up_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval = np.array([list_[:min_length] for list_ in data_for_plots['ball_only_at_the_up_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    if total_num_freq_constraint_violated_avg_per_log_interval is None:
                        total_num_freq_constraint_violated_avg_per_log_interval = np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval
                    else:
                        if np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval.shape[1] > total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval = \
                                np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval[:, :total_num_freq_constraint_violated_avg_per_log_interval.shape[1]]
                        elif np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval.shape[1] < total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            total_num_freq_constraint_violated_avg_per_log_interval = \
                                total_num_freq_constraint_violated_avg_per_log_interval[:, :np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval]
                        total_num_freq_constraint_violated_avg_per_log_interval += np_ball_only_at_the_up_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval

                if 'ball_only_at_the_right_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq) in data_for_plots.keys():
                    min_length = min([len(list_) for list_ in data_for_plots['ball_only_at_the_right_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval = np.array([list_[:min_length] for list_ in data_for_plots['ball_only_at_the_right_side_wrt_hole_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    if total_num_freq_constraint_violated_avg_per_log_interval is None:
                        total_num_freq_constraint_violated_avg_per_log_interval = np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval
                    else:
                        if np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval.shape[1] > total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval = \
                                np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval[:, :total_num_freq_constraint_violated_avg_per_log_interval.shape[1]]
                        elif np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval.shape[1] < total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            total_num_freq_constraint_violated_avg_per_log_interval = \
                                total_num_freq_constraint_violated_avg_per_log_interval[:, :np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval]
                        total_num_freq_constraint_violated_avg_per_log_interval += np_ball_only_at_the_right_side_wrt_hole_num_freq_constraint_violated_avg_per_log_interval
                if 'ball_not_in_circle_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq) in data_for_plots.keys():
                    min_length = min([len(list_) for list_ in data_for_plots['ball_not_in_circle_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval = np.array([list_[:min_length] for list_ in data_for_plots['ball_not_in_circle_{}_constraint_violated_avg_per_log_interval'.format(num_or_freq)]['test']['original_data_for_plots']])
                    if total_num_freq_constraint_violated_avg_per_log_interval is None:
                        total_num_freq_constraint_violated_avg_per_log_interval = np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval
                    else:
                        if np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval.shape[1] > total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval = \
                                np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval[:, :total_num_freq_constraint_violated_avg_per_log_interval.shape[1]]
                        elif np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval.shape[1] < total_num_freq_constraint_violated_avg_per_log_interval.shape[1]:
                            total_num_freq_constraint_violated_avg_per_log_interval = \
                                total_num_freq_constraint_violated_avg_per_log_interval[:, :np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval]
                        total_num_freq_constraint_violated_avg_per_log_interval += np_ball_not_in_circle_num_freq_constraint_violated_avg_per_log_interval

                mean_total_num_freq_constraint_violated_avg_per_log_interval = total_num_freq_constraint_violated_avg_per_log_interval.mean(axis=0)
                std_total_num_freq_constraint_violated_avg_per_log_interval = total_num_freq_constraint_violated_avg_per_log_interval.std(axis=0, ddof=1)
                data_dict[data_type].append({'method_name': method_names[path_idx],
                                             'mean': mean_total_num_freq_constraint_violated_avg_per_log_interval,
                                             'std': std_total_num_freq_constraint_violated_avg_per_log_interval})

    # Create figures
    plot_legend = tuple(method_names)
    fontsize = 23
    fontsize_only_ticks = 21
    labelpad = 10
    for data_type_idx, data_type in enumerate(data_types):
        if len(data_dict[data_type]) > 0:
            plt.figure()
            plt.xlabel(plot_xlabel[data_type_idx], fontsize=fontsize, labelpad=labelpad)
            plt.ylabel(plot_ylabel[data_type_idx], fontsize=fontsize, labelpad=labelpad)
            #plt.title(plot_title)
            for method_id, method_results in enumerate(data_dict[data_type]):

                # Define where to cut the plot, if so.
                n_episodes = method_results['mean' if is_pickle else 'data'].shape[0]
                if cut_plots_at_episode != 'None' and n_episodes > cut_plots_at_episode:
                    n_episodes = cut_plots_at_episode

                x_axis_ticks = [i + 1 for i in range(n_episodes)]
                if is_pickle:
                    # Plot the shadowed are based on std
                    plt.fill_between(x_axis_ticks,
                                     (method_results['mean'] - method_results['std']).squeeze()[:n_episodes],
                                     (method_results['mean'] + method_results['std']).squeeze()[:n_episodes],
                                     alpha=0.2,
                                     color=colors[method_id])
                # Plot the mean of data
                plt.plot(x_axis_ticks,
                         method_results['mean' if is_pickle else 'data'].squeeze()[:n_episodes],
                         alpha=0.5,
                         color=colors[method_id])
                # Change the fontsize of xticks and yticks
                plt.xticks(fontsize=fontsize_only_ticks)
                plt.yticks(fontsize=fontsize_only_ticks)

            # Restrict y-axis limit
            plt.ylim(bottom=bottom_ylim[data_type_idx])
            # Plot legend
            if plot_legend_loc[data_type_idx] is not None:
                plt.gca().legend(plot_legend, loc=plot_legend_loc[data_type_idx], fontsize=fontsize)
            # Adjust layout
            plt.tight_layout()
            # Save figure
            plt.savefig(os.path.join(dir_to_save, data_type + ".png"))
            plt.close()


if __name__ == '__main__':

    # Parse list of paths (NOTE: without spaces between commas and paths)
    paths_to_data_for_plots = sys.argv[1] # e.g., [/path/to/plots_data_1.pickle,/path/to/plots_data_2.pickle] for multi-train-test results or [/path/to/tmp/experiment_X/, /path/to/tmp/experiment_Y/]
    paths_to_data_for_plots = paths_to_data_for_plots.split('[')[1].split(']')[0].split(',')

    # Parse list of method names, (NOTE: without spaces between commas and name)
    method_names_for_plots = sys.argv[2]  # e.g., [Method_1,Method_2]
    method_names_for_plots = method_names_for_plots.split('[')[1].split(']')[0].split(',')

    # Parse path where plots will be saved
    save_dir = sys.argv[3]

    # Parse the number of episodes where to cut the plots. 'None' means don't cut.
    cut_at_episode = sys.argv[4]

    final_plots(paths_to_data_for_plots, method_names_for_plots, save_dir, cut_at_episode)

    print("\n##################################\nCreation of final plots completed!\n##################################")
