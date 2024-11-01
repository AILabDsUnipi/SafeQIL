import os
import yaml
import shutil


def get_config(config_file='config_sac.yaml'):
    try:
        with open(config_file) as file:
            yaml_data = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file. Cannot find the specified path: {}'.format(config_file))

    return yaml_data


def write_config(dict_data, config_path):
    try:
        with open(config_path, 'w') as yaml_file:
            yaml.safe_dump(dict_data, yaml_file, sort_keys=False)
    except Exception as e:
        print('Error writing the config file. Cannot find the specified path: {}'.format(config_path))


def get_result_dirs(config, exp_id, config_file_name):

    if 'SAC' in list(config.items())[1]:
        results_dir_name = config['SAC']['chkpt_dir']
    elif 'PPO' in list(config.items())[1]:
        results_dir_name = config['PPO']['chkpt_dir']
    elif 'NO_ALGO' in list(config.items())[1]:
        results_dir_name = config['NO_ALGO']['chkpt_dir']
    else:
        raise not NotImplementedError

    files_dir = os.path.join('results', 'tmp', results_dir_name)
    plot_dir = os.path.join('results', 'plots', results_dir_name)

    i = 1
    while os.path.exists(files_dir + '_' + exp_id + '_' + str(i)):
        i += 1
    os.makedirs(files_dir + '_' + exp_id + '_' + str(i))
    files_dir = files_dir + '_' + exp_id + '_' + str(i)

    j = 1
    while os.path.exists(plot_dir + '_' + exp_id + '_' + str(j)):
        j += 1
    os.makedirs(plot_dir + '_' + exp_id + '_' + str(j))
    plot_dir = plot_dir + '_' + exp_id + '_' + str(j)

    shutil.copy(config_file_name, files_dir)

    return files_dir, plot_dir
