import os
import pickle

import numpy as np

dir_path = "/home/georgepap/PycharmProjects/HAI-MAZE_master/experiments/multi_train_test/ppo_single_agent_exp_w_upper_constr_and_w_circle_constr_w_lagrangian_w_n_iters=40_w_norm_reward_range_-1_0_w_multimodes/"
exp_dirs = [exp_dir for exp_dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, exp_dir)) and exp_dir != 'final_results']

# Compute macro average results, i.e., first average over games and then over experiments
total_num_constraint_violations_all_exp = []
total_freq_constraint_violations_all_exp = []
for exp_dir in exp_dirs:

    pickle_dir = os.path.join(dir_path, exp_dir, "test_results", "tmp")

    # Load test data for plots
    test_path_to_data_for_plots = os.path.join(pickle_dir, 'plots_test_data.pickle')
    with open(test_path_to_data_for_plots, 'rb') as test_pickle_file:
        test_data_for_plots = pickle.load(test_pickle_file)

    # Sum all violations
    num_constraint_violations_keys = [key for key in test_data_for_plots.keys() if 'num_constraint_violations' in key]
    freq_constraint_violations_keys = [key for key in test_data_for_plots.keys() if 'freq_constraint_violations' in key]
    # Average over games
    total_num_constraint_violations = np.array([test_data_for_plots[key][5] for key in num_constraint_violations_keys]).sum(axis=0).mean()
    total_freq_constraint_violations = np.array([test_data_for_plots[key][5] for key in freq_constraint_violations_keys]).sum(axis=0).mean()
    # Store
    total_num_constraint_violations_all_exp.append(total_num_constraint_violations)
    total_freq_constraint_violations_all_exp.append(total_freq_constraint_violations)

# Average over games
mean_macro_avg_total_num_violations = np.array(total_num_constraint_violations_all_exp).mean()
std_macro_avg_total_num_violations = np.array(total_num_constraint_violations_all_exp).std(ddof=1)
mean_macro_avg_total_freq_violations = np.array(total_freq_constraint_violations_all_exp).mean()
std_macro_avg_total_freq_violations = np.array(total_freq_constraint_violations_all_exp).std(ddof=1)

# Append results to './final_results/test_results/tmp/stats_info.txt'
results_to_append = \
    ("###################" +
     "\nTotal number of constraint violations:" +
     "\n Macro avg:" +
     "\n  mean: {}".format(mean_macro_avg_total_num_violations) +
     "\n  std: {}".format(std_macro_avg_total_num_violations) +
     "\n\n###################"
     "\nTotal frequency of constraint violations:" +
     "\n Macro avg:" +
     "\n  mean: {}".format(mean_macro_avg_total_freq_violations) +
     "\n  std: {}".format(std_macro_avg_total_freq_violations))
path_to_txt = os.path.join(dir_path, 'final_results', 'test_results', 'tmp', 'stats_info.txt')
with open(path_to_txt, "a") as file:
    file.write(results_to_append)

print("\n##############################\n"
      "The macro average (mean and std) of the total number/frequency of violations were calculated and written successfully!"
      "\n##############################")
