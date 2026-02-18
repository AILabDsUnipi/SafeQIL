import os
import yaml
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


########################################################################
# 1) Parse stats_info.txt from test_seed={test_seed} to produce a table.
########################################################################


def parse_stats_info(file_path):
    """
    Reads stats_info.txt and extracts the numeric values for mean and std
    in the sections. Returns a dictionary with:
        {
            'Reward_mean': float,
            'Reward_std': float,
            'Steps_mean': float,
            'Steps_std': float,
            'Hazards_num_mean': float,
            'Hazards_num_std': float,
            'Hazards_freq_mean': float,
            'Hazards_freq_std': float,
            'Sum_num_mean': float,
            'Sum_num_std': float,
            'Sum_freq_mean': float,
            'Sum_freq_std': float
        }
    """
    stats = {
        'Reward_mean': None, 'Reward_std': None,
        'Steps_mean': None, 'Steps_std': None,
        'Hazards_num_mean': None, 'Hazards_num_std': None,
        'Hazards_freq_mean': None, 'Hazards_freq_std': None,
        'Sum_num_mean': None, 'Sum_num_std': None,
        'Sum_freq_mean': None, 'Sum_freq_std': None
    }

    with open(file_path, 'r') as f:
        content = f.read()

    sections = content.split('###################')
    for section in sections:
        if 'Reward:' in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Reward_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Reward_std'] = float(line.split(':')[-1].strip())

        elif 'Steps:' in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Steps_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Steps_std'] = float(line.split(':')[-1].strip())

        elif "Number of 'hazards' constraint violations:" in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Hazards_num_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Hazards_num_std'] = float(line.split(':')[-1].strip())

        elif "Frequency of 'hazards' constraint violations:" in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Hazards_freq_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Hazards_freq_std'] = float(line.split(':')[-1].strip())

        elif "Number of 'sum' constraint violations:" in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Sum_num_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Sum_num_std'] = float(line.split(':')[-1].strip())

        elif "Frequency of 'sum' constraint violations:" in section:
            lines = section.splitlines()
            for line in lines:
                if 'mean:' in line:
                    stats['Sum_freq_mean'] = float(line.split(':')[-1].strip())
                elif 'std:' in line:
                    stats['Sum_freq_std'] = float(line.split(':')[-1].strip())

    return stats


def aggregate_test_stats(log_dirs, legends, test_seed):
    """
    Iterates over the log_dirs (each labeled with a corresponding legend),
    finds `test_seed={test_seed}/tmp/stats_info.txt`, parses it, and returns
    a DataFrame with all the results. Each row is for one seed's experiment.
    """
    all_results = []

    for log_dir, legend in zip(log_dirs, legends):
        experiment_results = []
        for exp_folder in os.listdir(log_dir):
            exp_path = os.path.join(log_dir, exp_folder)
            if os.path.isdir(exp_path):

                test_dir = os.path.join(exp_path, f'test_results_seed={test_seed}')
                tmp_dir = os.path.join(test_dir, 'tmp')
                stats_file = os.path.join(tmp_dir, 'stats_info.txt')

                assert os.path.exists(stats_file), f'stats_info.txt not found: {stats_file}'
                stats = parse_stats_info(stats_file)
                stats['Experiment'] = legend + f"_{exp_folder}"
                stats['Legend'] = legend
                experiment_results.append(stats)

        if experiment_results:
            df = pd.DataFrame(experiment_results)
            all_results.append(df)

    if not all_results:
        # Return an empty DataFrame if nothing found
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def create_test_summary_csv(final_df, save_dir):
    """
    Given a DataFrame of test results (one row per seed/experiment),
    group by 'Legend' and compute mean/std. Saves the final summary
    to a CSV in save_dir.
    """
    # Calculate mean and std across seeds for each Legend
    summary_df = final_df.groupby('Legend').agg({
        'Reward_mean': ['mean', 'std'],
        'Reward_std': ['mean', 'std'],
        'Steps_mean': ['mean', 'std'],
        'Steps_std': ['mean', 'std'],
        'Hazards_num_mean': ['mean', 'std'],
        'Hazards_num_std': ['mean', 'std'],
        'Hazards_freq_mean': ['mean', 'std'],
        'Hazards_freq_std': ['mean', 'std'],
        'Sum_num_mean': ['mean', 'std'],
        'Sum_num_std': ['mean', 'std'],
        'Sum_freq_mean': ['mean', 'std'],
        'Sum_freq_std': ['mean', 'std']
    })
    summary_csv_path = os.path.join(save_dir, 'experiment_summary.csv')
    summary_df.to_csv(summary_csv_path)
    print(f"[INFO] Test summary CSV saved to: {summary_csv_path}")


########################################################################
# 2) Plot the training results (from train_results/tmp/*.csv)
########################################################################

def collect_train_data(log_dirs, legends):
    """
    Reads CSV files and the single YAML config in each log_dir's 'train_results/tmp' folder.
    For each CSV, renames columns <old_col> → <basename>_<old_col>,
    so that overlapping names (like 'Reward') from different files
    become distinct.
    Returns a nested dict:
    data_by_legend[legend][seed_folder] = {
          'dfs': [df1, df2, ...],
          'log_interval': int
      }
    We do NOT merge these DataFrames, so each CSV remains separate.
    """

    data_by_legend = {}

    for log_dir, legend in zip(log_dirs, legends):
        data_by_legend[legend] = {}
        for exp_folder in os.listdir(log_dir):
            exp_path = os.path.join(log_dir, exp_folder)
            if not os.path.isdir(exp_path):
                continue

            train_tmp_dir = os.path.join(exp_path, 'train_results', 'tmp')
            if os.path.isdir(train_tmp_dir):

                # 1. Find and parse the single .yaml file for log_interval
                yaml_files = [f for f in os.listdir(train_tmp_dir) if f.endswith('.yaml') or f.endswith('.yml')]
                assert len(yaml_files) == 1, f"Expected one YAML file in {train_tmp_dir}, found: {yaml_files}"
                config_path = os.path.join(train_tmp_dir, yaml_files[0])
                with open(config_path, 'r') as f:
                    conf = yaml.load(f, Loader=yaml.FullLoader)
                    log_interval = conf['Experiment']['log_interval']

                # 2. Collect CSVs
                csv_files = [
                    os.path.join(train_tmp_dir, f)
                    for f in os.listdir(train_tmp_dir)
                    if f.endswith('.csv')
                ]
                df_list = []
                for csv_f in csv_files:
                    df = pd.read_csv(csv_f)

                    # rename columns to disambiguate
                    base_name = os.path.splitext(os.path.basename(csv_f))[0]
                    rename_dict = {}
                    for c in df.columns:
                        # skip if you prefer not to rename e.g. 'Episode'
                        rename_dict[c] = f"{base_name}_{c}"
                    df.rename(columns=rename_dict, inplace=True)

                    df_list.append(df)

                if df_list:
                    # Store both the dataframes and the specific interval for this seed
                    data_by_legend[legend][exp_folder] = {
                        'dfs': df_list,
                        'log_interval': log_interval
                    }

    return data_by_legend


def get_episode_series_for_seed(df_list, y_col='Reward'):
    """
    Interprets each row in each DataFrame as a consecutive episode.
    We do NOT combine them into a single DataFrame. Instead, we just
    extract the y_col from each and concatenate them into a 1D array.

    This yields one continuous array of values for that seed,
    in the order they appear across the CSV files.
    """
    arrays = []
    for df in df_list:
        if y_col in df.columns:
            arrays.append(df[y_col].values)

    if arrays:
        # Concatenate them into a single 1D array for plotting
        return np.concatenate(arrays)
    else:
        return np.array([])


def plot_train_curves(data_by_legend, save_dir):
    """
    Produces ONE plot per numeric column, overlaid with multiple legends,
    only for columns that are COMMON across all legends (Intersection).
    Steps:
      1) Collect the union of all numeric columns across all legends' CSVs.
      2) For each numeric column:
         - Create a figure
         - For each legend:
           * Gather the seeds' data arrays
           * Truncate to min length across seeds
           * Compute mean ± std across seeds
           * Plot mean ± std
         - Save figure as 'train_<column>.png'
    Note that it scales x-axis if column name contains 'per_log_interval' using the
    log_interval found in the config.
    """

    ##############################
    mpl.rcParams.update({
        'axes.labelsize': 20,  # x/y label font. For paper: 20. Otherwise: 11
        'xtick.labelsize': 18,  # x tick font. For paper: 18. Otherwise: 9
        'ytick.labelsize': 18,  # y tick font. For paper: 18. Otherwise: 9
        'legend.fontsize': 20,  # legend font. For paper: 20. Otherwise: 11
    })
    ##############################

    # 1) Gather the INTERSECTION of numeric columns across all legends
    common_numeric_cols = None

    for legend, seeds_dict in data_by_legend.items():
        legend_cols = set()
        for seed, data_packet in seeds_dict.items():
            for df in data_packet['dfs']:
                # collect numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                legend_cols.update(numeric_cols)

        if common_numeric_cols is None:
            common_numeric_cols = legend_cols
        else:
            common_numeric_cols.intersection_update(legend_cols)

    if not common_numeric_cols:
        print("[WARNING] No common numeric columns found across all methods.")
        return

    sorted_cols = sorted(common_numeric_cols)  # sort for consistent ordering

    # 2) Generate one plot per common numeric column
    for col in sorted_cols:
        ##############################
        # For main body paper where wider figures are better: (9, 3). Otherwise: (8, 6).
        plt.figure(figsize=(8, 6))
        ##############################

        # check if this column requires scaling based on the log interval
        is_interval_metric = 'per_log_interval' in col

        # track if at least one legend had data for this column
        legend_plotted = False

        for legend, seeds_dict in data_by_legend.items():
            # For each legend, gather an array per seed
            all_seeds_arrays = []

            # We assume 'log_interval' is consistent across seeds of the same legend/method
            # Get interval from the first available seed
            current_interval = 1
            if len(seeds_dict) > 0:
                first_seed_key = next(iter(seeds_dict))
                current_interval = seeds_dict[first_seed_key]['log_interval']
            step_size = current_interval if is_interval_metric else 1

            for seed, data_packet in seeds_dict.items():
                seed_array = get_episode_series_for_seed(data_packet['dfs'], col)
                if seed_array.size > 0:
                    all_seeds_arrays.append(seed_array)

            if not all_seeds_arrays:
                # This legend has no data for `col`
                continue

            # Align all seeds by truncating to the shortest run
            min_len = min(len(arr) for arr in all_seeds_arrays)
            trimmed_arrays = [arr[:min_len] for arr in all_seeds_arrays]
            data_2d = np.vstack(trimmed_arrays)

            # Compute mean ± std across seeds
            mean_vals = data_2d.mean(axis=0)
            std_vals = data_2d.std(axis=0, ddof=1)  # sample std (ddof=1)

            # Construct X-axis based on the specific step size
            episodes = np.arange(min_len) * step_size

            # Plot
            plt.plot(episodes, mean_vals, label=legend)
            plt.fill_between(
                episodes,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2
            )
            legend_plotted = True

        # If no legend had data for this column, skip writing out a blank plot
        if not legend_plotted:
            plt.close()
            continue

        # Fix the y-axis label and the title
        col_label_to_use = col
        if col == 'test_rewards_avg_per_log_interval_Reward':
            col_label_to_use = 'Reward'
        elif col == 'test_num_constraint_sum_per_log_interval_Constraint sum':
            col_label_to_use = 'Cost'
            ##############################
            # For paper where we want to define the y-axis limits
            # plt.ylim(bottom=0, top=300)
            ##############################

        plt.xlabel("Episode")
        plt.ylabel(col_label_to_use)
        if col == col_label_to_use:
            plt.title(f"Training Curves (mean ± std) — {col}")
        ##############################
        # For paper where we want to place the legend below the x-axis:
        leg = plt.legend(
            loc='lower center',  # Position of the legend's anchor point
            bbox_to_anchor=(0.5, -0.1),  # Coordinates: (0.5=x-center, -0.1=below x-axis)
            ncol=3,  # Number of columns (make this equal to your # of labels)
            frameon=False  # Optional: Removes the box border
        )
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        # Otherwise:
        # plt.legend(loc='best')
        ##############################
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(save_dir, f"train_{col}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved plot: {plot_path}")


########################################################################
# Main routine combining everything.
########################################################################

def process_experiments(yaml_file_path):
    # 1) Load YAML config
    with open(yaml_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    log_dirs = config['log_dirs']
    legends = config['legends']
    test_seed = config['test_seed']
    save_dir = config['save_dir']

    os.makedirs(save_dir, exist_ok=True)

    ####################################################################
    # (A) Aggregate test statistics and save them
    ####################################################################
    final_df = aggregate_test_stats(log_dirs, legends, test_seed)
    if not final_df.empty:
        create_test_summary_csv(final_df, save_dir)
    else:
        print("[WARNING] No test stats found. Check your paths or test_seed directories.")

    ####################################################################
    # (B) Collect and plot training results
    ####################################################################
    data_by_legend = collect_train_data(log_dirs, legends)
    plot_train_curves(data_by_legend, save_dir)


########################################################################
# Usage example
########################################################################

if __name__ == "__main__":
    # Adjust the path to your actual YAML configuration file.
    yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'multi_exp_plot.yml')
    process_experiments(yaml_file)
