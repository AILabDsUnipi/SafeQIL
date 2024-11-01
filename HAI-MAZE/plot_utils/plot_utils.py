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


def get_plot_and_chkpt_dir(config, exp_ID, config_file_name):

    if 'SAC' in list(config.items())[1]:
        chkpt_dir_name = config['SAC']['chkpt_dir']
    elif 'coGAIL' in list(config.items())[1]:
        chkpt_dir_name = config['coGAIL']['chkpt_dir']
    elif 'PPO' in list(config.items())[1]:
        chkpt_dir_name = config['PPO']['chkpt_dir']
    elif 'NO_ALGO' in list(config.items())[1]:
        chkpt_dir_name = config['NO_ALGO']['chkpt_dir']
    else:
        raise not NotImplementedError

    chkpt_dir = 'results/tmp/' + chkpt_dir_name
    plot_dir = 'results/plots/' + chkpt_dir_name

    i = 1
    while os.path.exists(chkpt_dir + '_' + exp_ID + '_' + str(i)):
        i += 1
    os.makedirs(chkpt_dir + '_' + exp_ID + '_' + str(i))
    chkpt_dir = chkpt_dir + '_' + exp_ID + '_' + str(i)

    j = 1
    while os.path.exists(plot_dir + '_' + exp_ID + '_' + str(j)):
        j += 1
    os.makedirs(plot_dir + '_' + exp_ID + '_' + str(j))
    plot_dir = plot_dir + '_' + exp_ID + '_' + str(j)

    shutil.copy(config_file_name, chkpt_dir)

    return chkpt_dir, plot_dir


