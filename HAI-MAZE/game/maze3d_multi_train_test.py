import os
import shutil
from itertools import combinations
import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from plot_utils.plot_utils import get_config, write_config
from maze3D_new.utils import create_boxplot


def get_train_test_configs(algo_arg):
    train_config_arg = os.path.join("game", "config", "config_" + algo_arg + ".yaml")
    test_config_arg = os.path.join("game", "config", "config_" + algo_arg + "_agent-s_test.yaml")

    train_config = get_config(train_config_arg)
    test_config = get_config(test_config_arg)

    return train_config_arg, test_config_arg, train_config, test_config

def testing(
        test_config,
        test_config_arg,
        checkpoint_path,
        checkpoint_name,
        seeds_test,
        test_name,
        cur_test_dir_path,
        extra_test_dir_path_suffix="",
        test_print_message="\nTesting ...",
        SL_finetuning=False
):

    print(test_print_message)
    cur_exp_test_data_for_plots_list = []
    for seed_test_idx, seed_test in enumerate(seeds_test):
        print("\nTest with seed {}".format(seed_test))

        ## Copy test config, change the model path, the random seed, and create a new test .yaml file
        # Copy
        test_config_copy = copy.deepcopy(test_config)
        # Change
        test_config_copy['game']['checkpoint_path'] = ['', checkpoint_path]
        test_config_copy['game']['checkpoint_name'] = ['', checkpoint_name]
        test_config_copy['Experiment']['test_seed'] = seed_test
        if SL_finetuning is True and 'PPO' in test_config_copy.keys():
            test_config_copy['PPO']['ICRL'] = False
            test_config_copy['PPO']['lagrangian'] = False
        # Write
        test_tmp_yaml_file = test_config_arg.split('.yaml')[0] + '_tmp.yaml'
        print("Write new test .yaml file to {}".format(test_tmp_yaml_file))
        write_config(test_config_copy, test_tmp_yaml_file)

        # Perform the test
        print("Running the test ...")
        os.system(
            "python " + os.path.join("game", "maze3d_test_agent-s.py") + " " + test_tmp_yaml_file + " " + test_name
        )

        # Remove temporary .yaml file
        print("\nRemove new test .yaml file {}".format(test_tmp_yaml_file))
        os.remove(test_tmp_yaml_file)

        # Store the test directory name to use it in the command of moving the files
        assert len(os.listdir(os.path.join("results", "plots"))) == 1 and \
               len(os.listdir(os.path.join("results", "tmp"))) == 1 and \
               os.listdir(os.path.join("results", "plots"))[0] == os.listdir(os.path.join("results", "tmp"))[0]
        test_dir_name = os.listdir(os.path.join("results", "plots"))[0]
        cur_test_dir_name = "test_results" + (("_seed=" + str(seed_test)) if seed_test_idx != 0 else "")

        # Move the files of testing from "results/plots/<test_dir_name>"
        # to "<cur_test_dir_path>/<cur_test_dir_name>/<extra_test_dir_path_suffix>/plots" and
        # from "results/tmp/<test_dir_name>"
        # to "<cur_test_dir_path>/<cur_test_dir_name>/<extra_test_dir_path_suffix>/tmp"
        print("\nMoving test files ...")
        assert (
                       (extra_test_dir_path_suffix == "" or extra_test_dir_path_suffix == "0") and
                       os.path.exists(os.path.join(cur_test_dir_path, cur_test_dir_name)) is False and
                       os.path.exists(os.path.join(cur_test_dir_path, cur_test_dir_name, extra_test_dir_path_suffix)) is False
               ) or \
               (
                       extra_test_dir_path_suffix != "" and extra_test_dir_path_suffix != "0" and
                       os.path.exists(os.path.join(cur_test_dir_path, cur_test_dir_name)) is True and
                       os.path.exists(os.path.join(cur_test_dir_path, cur_test_dir_name, extra_test_dir_path_suffix)) is False
               )
        # Create folder(s) to move files into
        if os.path.exists(os.path.join(cur_test_dir_path, cur_test_dir_name)) is False:
            os.mkdir(os.path.join(cur_test_dir_path, cur_test_dir_name))
        if extra_test_dir_path_suffix != "":
            os.mkdir(os.path.join(cur_test_dir_path, cur_test_dir_name, extra_test_dir_path_suffix))
        # Move 'plots' files
        test_plots_dir_mv_from = os.path.join("results", "plots", test_dir_name)
        test_plot_dir_mv_to = os.path.join(cur_test_dir_path, cur_test_dir_name, extra_test_dir_path_suffix, "plots")
        print("Move files from {} to {}".format(test_plots_dir_mv_from, test_plot_dir_mv_to))
        os.system("mv " + test_plots_dir_mv_from + " " + test_plot_dir_mv_to)
        # Move 'tmp' files
        test_tmp_dir_mv_from = os.path.join("results", "tmp", test_dir_name)
        test_tmp_dir_mv_to = os.path.join(cur_test_dir_path, cur_test_dir_name, extra_test_dir_path_suffix, "tmp")
        print("Move files from {} to {}".format(test_tmp_dir_mv_from, test_tmp_dir_mv_to))
        os.system("mv " + test_tmp_dir_mv_from + " " + test_tmp_dir_mv_to)

        # Load test data for plots
        test_path_to_data_for_plots = os.path.join(test_tmp_dir_mv_to, 'plots_test_data.pickle')
        with open(test_path_to_data_for_plots, 'rb') as test_pickle_file:
            test_data_for_plots = pickle.load(test_pickle_file)
        cur_exp_test_data_for_plots_list.append(test_data_for_plots)

    return cur_exp_test_data_for_plots_list


def calculate_and_export_testing_results(
        n_exp,
        seeds_test,
        test_data_for_plots_list,
        final_results_dir,
        SL_finetuning=False,
        SL_comb_idx=None,
        SL_ft_model_idx=None,
        testing_results_print_message='Plot test results'
):

    print("\n#############" + testing_results_print_message + "#############")

    for seed_test_idx, seed_test in enumerate(seeds_test):

        print("\nPlot test results with seed {}".format(seed_test))

        # Create the directory to store the final plots of testing
        cur_seed_test_final_results_dir_name = \
            "test_results" + (("_seed=" + str(seed_test)) if seed_test_idx != 0 else "")
        test_final_results_dir = os.path.join(final_results_dir, cur_seed_test_final_results_dir_name)
        assert not (os.path.exists(test_final_results_dir) is True and SL_ft_model_idx == 0)
        if os.path.exists(test_final_results_dir) is False:
            os.mkdir(test_final_results_dir)
        SL_finetuning_extra_test_final_results_dir_path_suffix = ""
        if SL_finetuning is True:
            SL_finetuning_extra_test_final_results_dir_path_suffix = str(SL_ft_model_idx)
            assert os.path.exists(
                os.path.join(test_final_results_dir, SL_finetuning_extra_test_final_results_dir_path_suffix)
            ) is False
            os.mkdir(os.path.join(test_final_results_dir, SL_finetuning_extra_test_final_results_dir_path_suffix))
        test_plots_final_results_dir = os.path.join(
            test_final_results_dir, SL_finetuning_extra_test_final_results_dir_path_suffix, "plots"
        )
        os.mkdir(test_plots_final_results_dir)
        test_tmp_final_results_dir = os.path.join(
            test_final_results_dir, SL_finetuning_extra_test_final_results_dir_path_suffix, "tmp"
        )
        os.mkdir(test_tmp_final_results_dir)

        test_data_for_plots_dict_to_store = {}
        data_to_write_in_txt = ''

        # Use the dict keys (i.e., type of plots) of the first test experiment to create the corresponding plots
        if SL_finetuning is False:
            test_data_for_plots_keys = list(test_data_for_plots_list[0][0].keys())
        else:
            test_data_for_plots_keys = list(test_data_for_plots_list[0][0][0][0].keys())

        for test_plot_type in test_data_for_plots_keys:

            if SL_finetuning is False:
                test_data_for_plots = test_data_for_plots_list[0][0][test_plot_type]
            else:
                test_data_for_plots = test_data_for_plots_list[0][0][0][0][test_plot_type]
            test_plot_xlabel, test_plot_ylabel, test_plot_title, test_plot_legend, test_plot_legend_loc, _ = \
                test_data_for_plots

            # Gather the test data of the current plot type for all experiments
            tmp_test_data_for_plots_list = []
            for exp_id in range(n_exp):
                if SL_finetuning is False:
                    test_data_for_plots = test_data_for_plots_list[exp_id][seed_test_idx][test_plot_type]
                else:
                    test_data_for_plots = \
                        test_data_for_plots_list[exp_id][SL_comb_idx][SL_ft_model_idx][seed_test_idx][test_plot_type]
                _, _, _, _, _, test_plot_data = test_data_for_plots
                tmp_test_data_for_plots_list.append(test_plot_data)

            ## Plot
            plt.figure()
            plt.xlabel(test_plot_xlabel)
            plt.ylabel(test_plot_ylabel)
            plt.title(test_plot_title)
            # Compute mean and std
            test_mean_data = np.mean(tmp_test_data_for_plots_list, axis=0)
            test_std_data = np.std(tmp_test_data_for_plots_list, ddof=1, axis=0)
            # Create ticks for x-axis
            test_x_axis_ticks = [i + 1 for i in range(test_mean_data.shape[0])]
            # Plot the shadowed are based on std
            plt.fill_between(
                test_x_axis_ticks,
                (test_mean_data - test_std_data).squeeze(),
                (test_mean_data + test_std_data).squeeze(),
                alpha=0.5
            )
            # Plot the mean of data
            plt.plot(test_x_axis_ticks, test_mean_data.squeeze())
            plt.gca().legend(test_plot_legend, loc=test_plot_legend_loc)
            plt.savefig(os.path.join(test_plots_final_results_dir, test_plot_type + "_test.png"))
            plt.close()

            # Boxplot for macro average results
            test_data_average_over_test_games = np.mean(tmp_test_data_for_plots_list, axis=1).squeeze()
            create_boxplot(
                test_data_average_over_test_games,
                plot_title_=test_plot_title + " (macro avg)",
                ylabel_=test_plot_ylabel,
                path_name=os.path.join(test_plots_final_results_dir, test_plot_type + "_test_boxplot_macro_avg.png")
            )

            # Boxplot for micro average results
            test_flat_data = np.array(tmp_test_data_for_plots_list).flatten()
            create_boxplot(
                test_flat_data,
                plot_title_=test_plot_title + " (micro avg)",
                ylabel_=test_plot_ylabel,
                path_name=os.path.join(test_plots_final_results_dir, test_plot_type + "_test_boxplot_micro_avg.png")
            )

            # Keep the plot data to store them later
            test_data_for_plots_dict_to_store[test_plot_type] = {
                "mean": test_mean_data,
                "std": test_std_data,
                "data_for_plots": tmp_test_data_for_plots_list,
                "average_over_test_games": test_data_average_over_test_games,
                "flat_data": test_flat_data
            }

            # Write results statistics in txt file
            data_to_write_in_txt += \
                '###################\n' + \
                test_plot_title + ':\n' + \
                ' Macro avg:\n' + \
                '   mean: ' + str(np.mean(test_data_average_over_test_games)) + '\n' + \
                '   std: ' + str(np.std(test_data_average_over_test_games, ddof=1)) + '\n' + \
                '   median: ' + str(np.median(test_data_average_over_test_games)) + '\n' + \
                '   Q1: ' + str(np.quantile(test_data_average_over_test_games, 0.25)) + '\n' + \
                '   Q3: ' + str(np.quantile(test_data_average_over_test_games, 0.5)) + '\n' + \
                ' Micro avg:\n' + \
                '   mean: ' + str(np.mean(test_flat_data)) + '\n' + \
                '   std: ' + str(np.std(test_flat_data, ddof=1)) + '\n' + \
                '   median: ' + str(np.median(test_flat_data)) + '\n' + \
                '   Q1: ' + str(np.quantile(test_flat_data, 0.25)) + '\n' + \
                '   Q3: ' + str(np.quantile(test_flat_data, 0.5)) + '\n\n'
            with open(os.path.join(test_tmp_final_results_dir, 'stats_info.txt'), 'w') as stats_info:
                stats_info.write(data_to_write_in_txt)


def training(
        train_config_arg,
        train_name,
        exp_id=None,
        exp_dir_name=None,
        cur_exp_dir_name=None,
        SL_finetuning=False,
        train_print_message="\nTraining ...\n"
):

    print(train_print_message)
    os.system("python " + os.path.join("game", "maze3d_train_agent-s.py") + " " + train_config_arg + " " + train_name)

    if SL_finetuning is True:
        # Remove temporary .yaml file
        print("\nRemove new ft train .yaml file {}".format(train_config_arg))
        os.remove(train_config_arg)

    # Check the consistency of the created files
    assert len(os.listdir(os.path.join("results", "plots"))) == 1 and \
           len(os.listdir(os.path.join("results", "tmp"))) == 1 and \
           os.listdir(os.path.join("results", "plots"))[0] == os.listdir(os.path.join("results", "tmp"))[0]

    ft_exp_dir_name = None
    if SL_finetuning is False:
        # After the first experiment, store the experiment directory name without the suffix to use it
        # in the command of moving the files of the current and all subsequent experiments
        if exp_id == 0:
            exp_dir_name = "1".join(os.listdir(os.path.join("results", "plots"))[0].split("1")[:-1])
        cur_exp_dir_name = exp_dir_name + str(exp_id + 1)
    else:
        # Store the finetuning experiment directory name to use it in the command of moving the files
        ft_exp_dir_name = os.listdir(os.path.join("results", "plots"))[0]

    # If initial training:
    #   - Move the files of training from "results/plots/<exp_dir_name>1"
    #     to "results/<cur_exp_dir_name>/train_results/plots" and
    #   - from "results/tmp/<exp_dir_name>1"
    #     to "results/<cur_exp_dir_name>/train_results/tmp/<cur_exp_dir_name>"
    # Else if finetuning:
    #   - Move the files of training from "results/plots/<ft_exp_dir_name>"
    #     to "results/<cur_exp_dir_name>/finetuning_results/<ft_exp_dir_name>/train_results/plots" and
    #   - from "results/tmp/<ft_exp_dir_name>"
    #     to "results/<cur_exp_dir_name>/finetuning_results/<ft_exp_dir_name>/train_results/tmp/<ft_exp_dir_name>"
    print("\nMoving " + ("" if SL_finetuning is False else "finetuning ") + "experiment files ...")
    SL_finetuning_intermediate_path = \
        "" if SL_finetuning is False else (os.path.join("finetuning_results", ft_exp_dir_name))
    assert os.path.exists(os.path.join("results", cur_exp_dir_name, SL_finetuning_intermediate_path)) is False
    # Create folders to move files into
    os.mkdir(os.path.join("results", cur_exp_dir_name, SL_finetuning_intermediate_path))
    os.mkdir(os.path.join("results", cur_exp_dir_name, SL_finetuning_intermediate_path, "train_results"))
    os.mkdir(os.path.join("results", cur_exp_dir_name, SL_finetuning_intermediate_path, "train_results", "tmp"))
    # Move 'plots' files
    plots_dir_mv_from = os.path.join(
        "results", "plots", (exp_dir_name + str(1)) if SL_finetuning is False else ft_exp_dir_name
    )
    plot_dir_mv_to = os.path.join(
        "results", cur_exp_dir_name, SL_finetuning_intermediate_path, "train_results", "plots"
    )
    print("Move files from {} to {}".format(plots_dir_mv_from, plot_dir_mv_to))
    os.system("mv " + plots_dir_mv_from + " " + plot_dir_mv_to)
    # Move 'tmp' files
    tmp_dir_mv_from = os.path.join(
        "results", "tmp", (exp_dir_name + str(1)) if SL_finetuning is False else ft_exp_dir_name
    )
    tmp_dir_mv_to = os.path.join(
        "results",
        cur_exp_dir_name,
        SL_finetuning_intermediate_path,
        "train_results",
        "tmp",
        cur_exp_dir_name if SL_finetuning is False else ft_exp_dir_name
    )
    print("Move files from {} to {}".format(tmp_dir_mv_from, tmp_dir_mv_to))
    os.system("mv " + tmp_dir_mv_from + " " + tmp_dir_mv_to)

    # Load train data for plots to use it later in macro and micro average plots
    path_to_data_for_plots = os.path.join(tmp_dir_mv_to, 'plots_data.pickle')
    with open(path_to_data_for_plots, 'rb') as pickle_file:
        data_for_plots = pickle.load(pickle_file)

    if SL_finetuning is False:
        return exp_dir_name, cur_exp_dir_name, data_for_plots
    else:
        return ft_exp_dir_name, tmp_dir_mv_to, data_for_plots


def calculate_and_export_training_results(
        n_exp,
        data_for_plots_list,
        final_results_dir,
        SL_finetuning=False,
        SL_comb_idx=None,
        training_results_print_message='Plot train results'
):

    # Create the directory to store the final plots of training
    train_final_results_dir = os.path.join(final_results_dir, "train_results")
    assert os.path.exists(train_final_results_dir) is False
    os.mkdir(train_final_results_dir)
    train_plots_final_results_dir = os.path.join(train_final_results_dir, "plots")
    os.mkdir(train_plots_final_results_dir)
    train_tmp_final_results_dir = os.path.join(train_final_results_dir, "tmp")
    os.mkdir(train_tmp_final_results_dir)

    print("\n#############" + training_results_print_message + "#############")
    data_for_plots_dict_to_store = {}
    # Use the dict keys (i.e., type of plots) of the first train experiment to create the corresponding plots
    if SL_finetuning is False:
        data_for_plots_list_keys = list(data_for_plots_list[0].keys())
    else:
        data_for_plots_list_keys = list(data_for_plots_list[0][0].keys())
    for plot_type in data_for_plots_list_keys:

        if SL_finetuning is False:
            data_for_plots = data_for_plots_list[0][plot_type]
        else:
            data_for_plots = data_for_plots_list[0][0][plot_type]
        plot_xlabel, plot_ylabel, plot_title, plot_legend, plot_legend_loc, plot_data = data_for_plots
        data_for_plots_dict = {}
        data_for_plots_dict_to_store[plot_type] = {}
        # Initialize the 'min_len' variable in order to search
        # for the experiment with the minimum length of the current 'plot_type'.
        # This is necessary since the number of episodes might be different among the experiments.
        min_len = len(plot_data[list(plot_data.keys())[0]])
        data_types = list(plot_data.keys())

        # Gather the data of the current plot type for all experiments
        for data_type in data_types:
            data_for_plots_dict[data_type] = [plot_data[data_type]]
        for exp_id in range(1, n_exp):
            if SL_finetuning is False:
                data_for_plots = data_for_plots_list[exp_id][plot_type]
            else:
                data_for_plots = data_for_plots_list[exp_id][SL_comb_idx][plot_type]
            _, _, _, _, _, plot_data = data_for_plots
            for data_type in data_types:
                data_for_plots_dict[data_type].append(plot_data[data_type])
            # Update 'min_len'
            data_len = len(plot_data[data_types[0]])
            if data_len < min_len:
                assert SL_finetuning is False # In SL finetuning all the experiments results should have the same length
                min_len = data_len

        ## Plot
        plt.figure()
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.title(plot_title)
        x_axis_ticks = [i + 1 for i in range(min_len)]
        for data_type in data_types:
            # Keep the minimum length of each experiment data based on the 'min_len' variable
            tmp_data_for_plots = [data_elem[:min_len] for data_elem in data_for_plots_dict[data_type]]
            # Compute mean and std
            mean_data = np.mean(tmp_data_for_plots, axis=0)
            std_data = np.std(tmp_data_for_plots, ddof=1, axis=0)
            # Plot the shadowed are based on std
            plt.fill_between(
                x_axis_ticks,
                (mean_data - std_data).squeeze(),
                (mean_data + std_data).squeeze(),
                alpha=0.5
            )
            # Plot the mean of data
            plt.plot(x_axis_ticks, mean_data.squeeze())
            # Keep the plot data to store them later
            data_for_plots_dict_to_store[plot_type][data_type] = {
                "mean": mean_data,
                "std": std_data,
                "original_data_for_plots": data_for_plots_dict[data_type],
                "min_len": min_len
            }
        plt.gca().legend(plot_legend, loc=plot_legend_loc)
        plt.savefig(os.path.join(train_plots_final_results_dir, plot_type + ".png"))
        plt.close()

    # Store the final train data for plot
    with open(os.path.join(train_tmp_final_results_dir, 'train_plots_data.pickle'), 'wb') as pickle_file:
        pickle.dump(data_for_plots_dict_to_store, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv):
    # get configuration
    config = get_config(argv[0])

    # Check is 'results' folder is empty or does not exist
    assert os.path.exists('results') is False or (os.path.exists('results') is True and len(os.listdir('results')) == 0)

    # Check consistency between 'algo' and 'SL_finetuning'. Currently, only SAC and PPO supports SL finetuning
    assert not (config['SL_finetuning'] is True and config['algo'] not in ['SAC', 'PPO'])

    # Currently only SAC and PPO is supported
    assert config['algo'] in ['SAC', 'PPO']

    # Check 'num_exp' consistency
    assert config['num_exp'] > 1

    n_exp = config['num_exp']
    algo = config['algo']
    SL_finetuning = config['SL_finetuning']
    test_seeds = config['test_seeds']
    exp_dir_name = None

    # Calculate all the different combinations of the provided datasets
    if SL_finetuning is True:
        SL_finetuning_datasets = config['SL_finetuning_datasets']
        SL_finetuning_datasets_combs = []
        for n_elems_for_combs in range(1, len(SL_finetuning_datasets) + 1):
            SL_finetuning_datasets_combs += combinations(SL_finetuning_datasets, n_elems_for_combs)

    data_for_plots_list = []
    ft_data_for_plots_list = []
    test_data_for_plots_list = []
    ft_test_data_for_plots_list = []
    for exp_id in range(n_exp):
        train_config_arg, test_config_arg, train_config, test_config = get_train_test_configs(algo.lower())
        # Check the consistency between 'train_config' and 'test_config' for
        # 'test_max_games' and 'test_max_timesteps_per_game'
        assert not (
                train_config['Experiment']['test_max_games'] !=
                test_config['Experiment']['test_max_games'] or
                train_config['Experiment']['test_max_timesteps_per_game'] !=
                test_config['Experiment']['test_max_timesteps_per_game']
        )
        # Check the consistency between 'train_config' and 'test_config' for
        # 'constraint_ball_only_at_the_right_side_wrt_hole', 'constraint_ball_only_at_the_up_side_wrt_hole',
        # 'constraint_ball_not_in_circle', 'constraint_ball_not_in_circle_circle_position',
        # 'constraint_ball_not_in_circle_circle_radius'
        assert not (
                train_config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] !=
                test_config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole'] or
                train_config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] !=
                test_config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole'] or
                train_config['Experiment']['constraint_ball_not_in_circle'] !=
                test_config['Experiment']['constraint_ball_not_in_circle'] or
                (
                        train_config['Experiment']['constraint_ball_not_in_circle'] is True and
                        (
                                train_config['Experiment']['constraint_ball_not_in_circle_circle_position'] !=
                                test_config['Experiment']['constraint_ball_not_in_circle_circle_position'] or
                                train_config['Experiment']['constraint_ball_not_in_circle_circle_radius'] !=
                                test_config['Experiment']['constraint_ball_not_in_circle_circle_radius']
                        )
                )
        )
        if algo == 'SAC':
            # Check the consistency between 'train_config' and 'test_config' for
            # 'layer1_size', 'layer2_size', 'normalize_features'
            assert not (
                    train_config['SAC']['layer1_size'] != test_config['SAC']['layer1_size'] or
                    train_config['SAC']['layer2_size'] != test_config['SAC']['layer2_size'] or
                    train_config['SAC']['normalize_features'] != test_config['SAC']['normalize_features']
            )
        elif algo == 'PPO':
            # Check the consistency between 'train_config' and 'test_config' for
            # 'layer1_size', 'layer2_size', and 'normalize_features'
            assert not (
                    train_config['PPO']['layer1_size'] != test_config['PPO']['layer1_size'] or
                    train_config['PPO']['layer2_size'] != test_config['PPO']['layer2_size'] or
                    train_config['PPO']['normalize_features'] != test_config['PPO']['normalize_features']
            )
            # Check the consistency between 'train_config' and 'test_config' for
            # 'ICRL', 'lagrangian', 'ICRL_constraint_net_layer1_size', and 'ICRL_constraint_net_layer2_size'
            assert not (
                    train_config['PPO']['ICRL'] != test_config['PPO']['ICRL'] or
                    (
                            train_config['PPO']['ICRL'] is True and
                            (
                                    train_config['PPO']['ICRL_constraint_net_layer1_size'] !=
                                    test_config['PPO']['ICRL_constraint_net_layer1_size'] or
                                    train_config['PPO']['ICRL_constraint_net_layer2_size'] !=
                                    test_config['PPO']['ICRL_constraint_net_layer2_size']
                            )
                    ) or
                    train_config['PPO']['lagrangian'] != test_config['PPO']['lagrangian']
            )
        else:
            raise NotImplementedError

        print("\n\n#############RUN EXP {}#############".format(exp_id))

        # Train
        exp_dir_name, cur_exp_dir_name, data_for_plots = training(
            train_config_arg,
            argv[1],
            exp_id=exp_id,
            exp_dir_name=exp_dir_name
        )
        data_for_plots_list.append(data_for_plots)

        # Test
        test_data_for_plots = testing(
            test_config,
            test_config_arg,
            os.path.join("results", cur_exp_dir_name, "train_results", "tmp"),
            cur_exp_dir_name,
            test_seeds,
            argv[1],
            os.path.join("results", cur_exp_dir_name)
        )
        test_data_for_plots_list.append(test_data_for_plots)

        if SL_finetuning is True:
            ## Finetuning with the different provided datasets
            print("\nFinetuning ...")

            # Create new directory "/results/<cur_exp_dir_name>/finetuning_results" to store finetuning results
            ft_results_target_dir = os.path.join("results", cur_exp_dir_name, "finetuning_results")
            assert os.path.exists(ft_results_target_dir) is False
            os.mkdir(ft_results_target_dir)

            ## Run a finetuning process for each combination, and then the tests with different seeds
            cur_exp_ft_data_for_plots_list = []
            cur_exp_ft_test_data_for_plots_list = []
            for comb in SL_finetuning_datasets_combs:
                print("\nDatasets combination: {}".format(comb))
                ## Copy train config, change the model path, 'load_checkpoint', 'X_Y_agent', 'X_Y_agent_pretrained',
                # 'save_all', and 'SL_finetuning' variables, dataset paths, and create a new ft train .yaml file
                # Copy
                train_config_copy = copy.deepcopy(train_config)
                # Change
                train_config_copy['game']['checkpoint_path'] = \
                    ['', os.path.join("results", cur_exp_dir_name, "train_results", "tmp")]
                train_config_copy['game']['checkpoint_name'] = ['', cur_exp_dir_name]
                train_config_copy['game']['load_checkpoint'] = True
                train_config_copy['game']['X_Y_agent'] = False
                train_config_copy['game']['X_Y_agent_pretrained'] = True
                train_config_copy['game']['save_all'] = True
                train_config_copy[algo]['SL_finetuning'] = True
                train_config_copy[algo]['expert_dataset_paths'] = \
                    [
                        os.path.join(
                            train_config[algo]['expert_dataset_paths'][0].split('results_human_alone_sub1')[0],
                            'results_human_alone_sub1_single_traj_' + data_name + '_1',
                            'tmp',
                            'human_alone_exp_test_1'
                        )
                        for data_name in list(comb)
                    ]
                if algo == 'PPO':
                    train_config_copy[algo]['ICRL'] = False
                    train_config_copy[algo]['lagrangian'] = False
                # Write
                ft_train_tmp_yaml_file = train_config_arg.split('.yaml')[0] + '_tmp.yaml'
                print("Write new ft train .yaml file to {}".format(ft_train_tmp_yaml_file))
                write_config(train_config_copy, ft_train_tmp_yaml_file)

                # Finetune
                ft_exp_name_arg = 'SL_data'
                for data_name in list(comb):
                    ft_exp_name_arg += '_' + data_name
                ft_exp_dir_name, ft_tmp_dir_mv_to, ft_data_for_plots = training(
                    ft_train_tmp_yaml_file,
                    ft_exp_name_arg,
                    exp_dir_name=exp_dir_name,
                    cur_exp_dir_name=cur_exp_dir_name,
                    SL_finetuning=True,
                    train_print_message="Running finetuning ..."
                )
                cur_exp_ft_data_for_plots_list.append(ft_data_for_plots)

                # Test using the fine-tuned models
                cur_ft_cur_exp_ft_test_data_for_plots_list = []
                SL_finetuning_epochs = len([
                    dir_elem for dir_elem in os.listdir(ft_tmp_dir_mv_to)
                    if os.path.isdir(os.path.join(ft_tmp_dir_mv_to, dir_elem))
                ])
                for ft_model_idx in range(SL_finetuning_epochs):
                    ft_test_data_for_plots = testing(
                        test_config,
                        test_config_arg,
                        os.path.join(
                            "results",
                            cur_exp_dir_name,
                            "finetuning_results",
                            ft_exp_dir_name,
                            "train_results",
                            "tmp",
                            ft_exp_dir_name
                        ),
                        str(ft_model_idx),
                        test_seeds,
                        ft_exp_name_arg,
                        os.path.join(
                            "results",
                            cur_exp_dir_name,
                            "finetuning_results",
                            ft_exp_dir_name
                        ),
                        str(ft_model_idx),
                        test_print_message="\nFinetuning testing ...",
                        SL_finetuning=True
                    )
                    cur_ft_cur_exp_ft_test_data_for_plots_list.append(ft_test_data_for_plots)
                cur_exp_ft_test_data_for_plots_list.append(cur_ft_cur_exp_ft_test_data_for_plots_list)
            ft_test_data_for_plots_list.append(cur_exp_ft_test_data_for_plots_list)
            ft_data_for_plots_list.append(cur_exp_ft_data_for_plots_list)

    # At the end of the training and testing processes,
    # remove the redundant (and empty) "results/plots" and "results/tmp" folders
    shutil.rmtree(os.path.join("results", "plots"))
    shutil.rmtree(os.path.join("results", "tmp"))

    ### Calculate and plot final results

    # Create the directory to store the final plots
    final_results_dir = os.path.join("results", "final_results")
    assert os.path.exists(final_results_dir) is False
    os.mkdir(final_results_dir)

    # Calculate and plot train results
    calculate_and_export_training_results(n_exp, data_for_plots_list, final_results_dir)

    # Calculate and plot test results
    calculate_and_export_testing_results(n_exp, test_seeds, test_data_for_plots_list, final_results_dir)

    if SL_finetuning is True:
        ## Calculate and plot train and test results for each finetuning combination

        # Create the directories to store the results of finetuning
        ft_final_results_dir = os.path.join(final_results_dir, "finetuning_results")
        assert os.path.exists(ft_final_results_dir) is False
        os.mkdir(ft_final_results_dir)

        for comb_idx, comb in enumerate(SL_finetuning_datasets_combs):

            # Create the directories to store the finetuning results of the current combination
            ft_exp_name_arg = 'SL_data'
            for data_name in list(comb):
                ft_exp_name_arg += '_' + data_name
            ft_cur_comb_final_results_dir = os.path.join(ft_final_results_dir, ft_exp_name_arg)
            assert os.path.exists(ft_cur_comb_final_results_dir) is False
            os.mkdir(ft_cur_comb_final_results_dir)

            # Calculate and plot finetuning results of the current combination
            calculate_and_export_training_results(
                n_exp,
                ft_data_for_plots_list,
                ft_cur_comb_final_results_dir,
                SL_finetuning=True,
                SL_comb_idx=comb_idx,
                training_results_print_message=
                "Plot finetuning results for '{}' combination".format(ft_exp_name_arg)
            )

            for ft_model_idx in range(SL_finetuning_epochs):
                calculate_and_export_testing_results(
                    n_exp,
                    test_seeds,
                    ft_test_data_for_plots_list,
                    ft_cur_comb_final_results_dir,
                    SL_finetuning=True,
                    SL_comb_idx=comb_idx,
                    SL_ft_model_idx=ft_model_idx,
                    testing_results_print_message=
                    "Plot finetuning test results for '{}' combination and epoch {}".format(
                        ft_exp_name_arg, ft_model_idx
                    )
                )

    print("\n###############################Multi train and test completed!###############################")


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
