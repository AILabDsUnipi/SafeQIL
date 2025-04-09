import os
import yaml
import shutil
import pathlib
import sys
import csv


def resolve_source_path():
    source_dir = pathlib.Path(__file__).absolute().parent.parent.parent
    sys.path.insert(0, str(source_dir))


def get_config(config_file='config_sac.yaml'):
    try:
        with open(config_file) as file:
            yaml_data = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file. Cannot find the specified path: {}'.format(config_file))
        print('Error: {}'.format(e))

    return yaml_data


def write_config(dict_data, config_path):
    try:
        with open(config_path, 'w') as yaml_file:
            yaml.safe_dump(dict_data, yaml_file, sort_keys=False)
    except Exception as e:
        print('Error writing the config file. Cannot find the specified path: {}'.format(config_path))
        print('Error: {}'.format(e))


def write_info_file(info, file_name, info_dir):
    try:
        file_path = os.path.join(info_dir, f'{file_name}.csv')
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 1) Write a row of keys
            keys = list(info.keys())
            writer.writerow(keys)

            # 2) Write a row of values in the same order
            values = [info[k] for k in keys]
            writer.writerow(values)
    except Exception as e:
        print('Error writing the info file: \n{}'.format(e))


def get_result_dirs(exp_id, config_file_name, test_only=False, seed=None, provided_exp_id=None):

    if test_only:
        assert seed is not None, "Seed must be provided for testing"

    intermediate_folder_name = 'train_results' if not test_only else f'test_results_seed={seed}'

    if provided_exp_id is None:
        i = 1
        while os.path.exists(exp_id + '_' + str(i)):
            i += 1
        exp_id = exp_id + '_' + str(i)
    else:
        exp_id = provided_exp_id

    files_dir = os.path.join(exp_id, intermediate_folder_name, 'tmp')
    os.makedirs(files_dir)

    plot_dir = os.path.join(exp_id, intermediate_folder_name, 'plots')
    os.makedirs(plot_dir)

    shutil.copy(config_file_name, files_dir)

    return files_dir, plot_dir, exp_id
