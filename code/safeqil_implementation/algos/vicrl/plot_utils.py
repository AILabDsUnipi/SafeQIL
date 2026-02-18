from typing import Optional, Dict

from safeqil_implementation.algos.icrl.plot_utils import save_constraint_net_logs_and_plot as icrl_save_constraint_net_logs_and_plot


def save_constraint_net_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots=False) -> Optional[Dict]:
    """
    Save the constraint net logs and plot them.
    :param experiment: The experiment object.
    :param files_dir: The directory where to save the logs.
    :param plot_dir: The directory where to save the plots.
    :param return_data_for_plots: If True, return the data for plotting.
    :return: None or data for plotting.
    """

    data_for_plots_to_return = icrl_save_constraint_net_logs_and_plot(experiment, files_dir, plot_dir, return_data_for_plots)

    return data_for_plots_to_return
