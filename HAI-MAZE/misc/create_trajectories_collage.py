import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def create_trajectories_collage_func(training_dir, target_dir, ft):

    print(f"\nCreating trajectories collage for directory: {target_dir}")

    single_train_test = False
    if ft is True:
        # Get the test seeds of testing for training
        training_test_seeds = sorted([train_dir_name for train_dir_name in os.listdir(training_dir) if "test" in train_dir_name])
        # Get the test seeds of testing for finetuning
        finetuning_test_seeds = sorted([target_dir_name for target_dir_name in os.listdir(target_dir) if "test" in target_dir_name])
        # Check that all testing seeds for finetuning have been applied for the same epochs
        assert training_test_seeds == finetuning_test_seeds
        finetuning_epochs = os.listdir(os.path.join(target_dir, finetuning_test_seeds[0]))
        if len(finetuning_test_seeds) > 1:
            for test_seed_idx in range(1, len(finetuning_test_seeds)):
                assert os.listdir(os.path.join(target_dir, finetuning_test_seeds[test_seed_idx])) == finetuning_epochs
        nrows, ncols = len(training_test_seeds), len(finetuning_epochs)+1 # array of sub-plots. +1 for columns to include training results
        figsize = [int(21 * (nrows / 2)), int(6 * (ncols / 6))]  # figure size in inches
    else:
        if 'train_results' in os.listdir(training_dir):
            single_train_test = True
        if single_train_test is False:
            # Get the test seeds of testing for training
            training_dirs = os.listdir(training_dir)
            training_dirs = [training_dir_ for training_dir_ in training_dirs if os.path.isdir(os.path.join(training_dir, training_dir_))]
            training_dirs.remove('final_results')
            training_test_seeds = sorted([test_dir_name for test_dir_name in os.listdir(os.path.join(training_dir, training_dirs[0])) if "test" in test_dir_name])
            # Check that all training dirs have the same test dirs
            for training_dir_ in training_dirs:
                training_dir_ = os.path.join(training_dir, training_dir_)
                training_test_seeds_ = sorted([test_dir_name for test_dir_name in os.listdir(training_dir_) if "test" in test_dir_name])
                assert training_test_seeds == training_test_seeds_, \
                    ("Inconsistency between 'training_test_seeds' and 'training_test_seeds_': " +
                     "\n'training_test_seeds': {}, 'training_test_seeds_': {}".format(training_test_seeds, training_test_seeds_))
        else:
            training_dirs = [training_dir]
            training_test_seeds = sorted([test_dir_name for test_dir_name in os.listdir(training_dir) if "test" in test_dir_name])
            assert len(training_test_seeds) > 1, \
                f"'training_test_seeds': {training_test_seeds} must contain at least two test seeds."

        nrows, ncols = len(training_test_seeds), len(training_dirs)  # array of sub-plots
        figsize = [2.5*ncols, 2.5*nrows] # figure size in inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={'xticks': [], 'yticks': []}) # remove ticks

    # plot each image on a sub-plot
    for i, axi in enumerate(ax.flat):
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols

        # read image
        if ft is True:
            if colid == 0:
                img_path = os.path.join(training_dir, training_test_seeds[rowid], 'plots', 'ball_trajectories.png')
            else:
                img_path = os.path.join(target_dir, training_test_seeds[rowid], str(colid-1), 'plots', 'ball_trajectories.png')
        else:
            if single_train_test is False:
                img_path = os.path.join(training_dir, training_dirs[colid], training_test_seeds[rowid], 'plots', 'ball_trajectories.png')
            else:
                plots_dir = os.path.join(training_dir, training_dirs[colid], training_test_seeds[rowid], 'plots')
                plots_dir_ = os.listdir(plots_dir)
                assert len(plots_dir_) == 1, \
                    (f"Found {len(plots_dir_)} plots directories!" +
                     "\nThere should be just a single plots directory. \n'plots_dir_': '{plots_dir_}'")
                plots_dir_ = plots_dir_[0]
                img_path = os.path.join(plots_dir, plots_dir_, 'ball_trajectories.png')
        assert os.path.exists(img_path), "The provided image path does not exists: {}".format(img_path)
        img = plt.imread(img_path)

        # Crop image
        img = img[58:-55, 145:-130]

        # Plot image
        axi.imshow(img)

        # write row/col indices as axes' title for identification
        if rowid == 0:
            row_title = "Train seed"
            if ft is True:
                col_title = "Train" if colid == 0 else "FT epoch={}".format(colid-1)
            else:
                col_title = "Trial {}".format(colid + 1)
        elif rowid == 1:
            row_title = "Test seed"
            col_title = ""
        else:
            raise NotImplementedError
        axi.set_title(col_title, fontsize=23)
        if colid == 0:
            axi.set_ylabel(row_title, fontsize=23)

    plt.tight_layout(h_pad=1.0)

    # Save figure
    fig_path = os.path.join(target_dir, 'all_trajectories_fig.png')
    plt.savefig(fig_path)
    plt.close()

def create_trajectories_collage_multi(training_dir, ft):
    if ft is True:
        target_dir = os.path.join(training_dir, 'finetuning_results')
        assert os.path.exists(target_dir), f"The provided finetuning directory does not exists: {target_dir}"
        target_dirs = os.listdir(target_dir)
    else:
        target_dir = training_dir
        single_or_multi = 'multi-train-and-test' if 'multi_train_test' in target_dir else 'single-train-and-test'
        assert os.path.exists(target_dir), f"The provided {single_or_multi} directory does not exists: {target_dir}"
        target_dirs = [target_dir]

    for target_dir_ in tqdm(target_dirs, desc="Creating trajectories collage ..."):
        if ft is True:
            target_dir_ = os.path.join(target_dir, target_dir_)
        else:
            target_dir_ = target_dir
        create_trajectories_collage_func(training_dir, target_dir_, ft)


if __name__ == '__main__':

    # Provide the directory of which the sub-folders contain the ball trajectories
    # e.g.:
    # For multi-train-test:
    #   - For finetuning: './../experiments/multi_train_test/sac_single_agent_exp_multi_train_test_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_entr_coef_lr=0.0003_w_norm_reward_-1_0/sac_single_agent_exp_multi_train_test_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_entr_coef_lr=0.0003_w_norm_reward_-1_0_1'
    #   - For training: './../experiments/multi_train_test/sac_single_agent_exp_multi_train_test_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_entr_coef_lr=0.0003_w_norm_reward_-1_0'
    # For single-train-test:
    #   - For training: './../sac_single_agent_exp_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_norm_reward_range_-1_0_w_multimodes_1'
    #   - For finetuning: not supported
    train_dir = sys.argv[1] # Parse path

    # Choose to create trajectories collage for 'Finetuning' or 'Training'
    ft_flag = sys.argv[2] # Parse flag. When 'False', plot the multi-train-and-test ball trajectories, When 'True', plot the multi-finetuning ball trajectories
    if ft_flag not in ['True', 'False']:
        raise ValueError("Invalid flag value! Must be 'True' or 'False'. Got {}".format(ft_flag))
    ft_flag = eval(ft_flag)

    # Currently, 'create_trajectories_collage_multi' and 'create_trajectories_collage_func' have the same results for 'training' mode
    # target_dir_to_plot = '/home/georgepap/PycharmProjects/HAI-MAZE_master/experiments/multi_train_test/sac_single_agent_exp_multi_train_test_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_norm_reward_range_-1_0/sac_single_agent_exp_multi_train_test_w_upper_constraint_and_w_circle_constraint_w_entropy_in_lagrangian_w_norm_reward_range_-1_0_1/finetuning_results/sac_single_agent_exp_SL_data_layout_3_fifth_row_bottom_1/'
    # create_trajectories_collage_func(train_dir, target_dir_to_plot, ft_flag)

    create_trajectories_collage_multi(train_dir, ft_flag)
