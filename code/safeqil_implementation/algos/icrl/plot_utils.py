import os
from typing import Optional, Dict

import pandas as pd

from safeqil_implementation.utils.plot_utils import group_metrics, plot_train_logs


def save_constraint_net_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots=False) -> Optional[Dict]:
    """
    Save the constraint net logs and plot them.
    :param experiment: The experiment object.
    :param files_dir: The directory where to save the logs.
    :param plot_dir: The directory where to save the plots.
    :param return_data_for_plots: If True, return the data for plotting.
    :return: None or data for plotting.
    """

    print("\nSaving constraint net logs and plots...")

    ## Save constraint net train logs in files
    # Remove items where the list is empty
    train_logs = {
        k: v for k, v in experiment.constraint_net_train_logs_dict.items() if len(v) > 0
    }
    # Create dataframes
    train_logs = pd.DataFrame(train_logs)
    # Save to csv
    train_logs.to_csv(
        os.path.join(files_dir, 'constraint_net_train_logs.csv'), index=False
    )

    # Plot for all steps.
    # The first element of each tuple is the ylabel,
    # the second is the plot title,
    # the third is the data to plot, and
    # the forth is the title of the file to be written.
    data_to_plot_train = group_metrics(experiment.constraint_net_train_logs_dict)
    x_axis = [(i + 1) for i in range(len(data_to_plot_train[0][2][0]))]
    data_for_plots_to_return = plot_train_logs(
        data_to_plot_train,
        x_axis,
        per_log_interval=False,
        plot_dir=plot_dir,
        return_data_for_plots=return_data_for_plots
    )

    print(f"Logs and plots for constraint net saved in '{files_dir}' and '{plot_dir}' successfully!")

    return data_for_plots_to_return
