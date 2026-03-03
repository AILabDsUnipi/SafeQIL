import os
import pickle
import yaml

import numpy as np
import h5py


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def convert_dataset_to_hdf5(input_folder: str, output_folder: str):
    """
    Converts a directory of nested dictionary .pkl files into compressed HDF5 files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"No .pkl files found in {input_folder}")
        return

    print(f"Found {len(pkl_files)} files. Starting conversion...\n")

    total_original_size = 0
    total_new_size = 0

    for file_name in pkl_files:
        input_path = os.path.join(input_folder, file_name)

        # We'll name the output file the same, but with .h5 extension
        output_name = file_name.replace('.pkl', '.h5')
        output_path = os.path.join(output_folder, output_name)

        total_original_size += os.path.getsize(input_path)

        print(f"Processing: {file_name}...")

        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)

            # Open HDF5 file for writing
            with h5py.File(output_path, 'w') as h5f:

                # Iterate through episodes in the file
                for episode_key in data['actions'].keys():
                    ep_group = h5f.create_group(episode_key)

                    # 1. Sort the keys.
                    # all_keys includes the final step for obs/actions
                    all_keys = sorted(
                        data['actions'][episode_key].keys(),
                        key=lambda k: int(k.split('_')[-1])
                    )

                    # valid_keys only includes steps where a reward/done actually exists
                    valid_keys = sorted(
                        data['reward'][episode_key].keys(),
                        key=lambda k: int(k.split('_')[-1])
                    )

                    # 2. Extract, stack, and enforce dtypes
                    # Actions: usually floats
                    actions = np.stack([data['actions'][episode_key][k] for k in all_keys])

                    # Rewards: usually floats
                    rewards = np.array([data['reward'][episode_key][k] for k in valid_keys])

                    # Dones: booleans
                    dones = np.array([data['done'][episode_key][k] for k in valid_keys])

                    # Observations (Images and Vectors)
                    vision_obs = np.stack([data['vision_obs'][episode_key][k] for k in all_keys])
                    vector_obs = np.stack([data['vector_obs'][episode_key][k] for k in all_keys])

                    # 3. Save to HDF5 with gzip compression (level 4 is a great speed/size tradeoff)
                    ep_group.create_dataset('actions', data=actions, compression='gzip', compression_opts=4)
                    ep_group.create_dataset('reward', data=rewards, compression='gzip', compression_opts=4)
                    ep_group.create_dataset('done', data=dones, compression='gzip', compression_opts=4)
                    ep_group.create_dataset('vector_obs', data=vector_obs, compression='gzip', compression_opts=4)
                    ep_group.create_dataset('vision_obs', data=vision_obs, compression='gzip', compression_opts=4)

            total_new_size += os.path.getsize(output_path)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    # Print final compression stats
    orig_mb = total_original_size / (1024 * 1024)
    new_mb = total_new_size / (1024 * 1024)
    savings = (1 - (new_mb / orig_mb)) * 100 if orig_mb > 0 else 0

    print("\n--- Conversion Complete ---")
    print(f"Original Dataset Size: {orig_mb:.2f} MB")
    print(f"New Dataset Size:      {new_mb:.2f} MB")
    print(f"Space Saved:           {savings:.1f}%")


def convert_dataset_to_pickle(input_folder: str, output_folder: str):
    """
    Converts a directory of compressed HDF5 files back into nested dictionary .pkl files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    if not h5_files:
        print(f"No .h5 files found in {input_folder}")
        return

    print(f"Found {len(h5_files)} files. Starting conversion to Pickle...\n")

    total_original_size = 0
    total_new_size = 0

    for file_name in h5_files:
        input_path = os.path.join(input_folder, file_name)
        output_name = file_name.replace('.h5', '.pkl')
        output_path = os.path.join(output_folder, output_name)

        total_original_size += os.path.getsize(input_path)
        print(f"Processing: {file_name}...")

        try:
            # Initialize the base dictionary structure
            data = {
                'actions': {},
                'reward': {},
                'done': {},
                'vector_obs': {},
                'vision_obs': {}
            }

            with h5py.File(input_path, 'r') as h5f:
                for episode_key in h5f.keys():

                    # Initialize nested dicts for this specific episode
                    for key in data.keys():
                        data[key][episode_key] = {}

                    ep_group = h5f[episode_key]

                    # Load arrays into memory
                    actions = ep_group['actions'][:]
                    rewards = ep_group['reward'][:]
                    dones = ep_group['done'][:]
                    vector_obs = ep_group['vector_obs'][:]
                    vision_obs = ep_group['vision_obs'][:]

                    num_all_steps = len(actions)
                    num_valid_steps = len(rewards)

                    # Reconstruct the dictionary step-by-step
                    for i in range(num_all_steps):
                        step_key = f"step_{i}"

                        data['actions'][episode_key][step_key] = actions[i]
                        data['vector_obs'][episode_key][step_key] = vector_obs[i]
                        data['vision_obs'][episode_key][step_key] = vision_obs[i]

                        # The final step doesn't have a reward or done flag
                        if i < num_valid_steps:
                            # Converting back to standard Python float/bool
                            data['reward'][episode_key][step_key] = float(rewards[i])
                            data['done'][episode_key][step_key] = bool(dones[i])

            # Save back to Pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)

            total_new_size += os.path.getsize(output_path)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    # Print final decompression stats
    orig_mb = total_original_size / (1024 * 1024)
    new_mb = total_new_size / (1024 * 1024)
    growth = ((new_mb / orig_mb) - 1) * 100 if orig_mb > 0 else 0

    print("\n--- Conversion Complete ---")
    print(f"Original Dataset Size (H5):  {orig_mb:.2f} MB")
    print(f"New Dataset Size (Pickle):   {new_mb:.2f} MB")
    print(f"Size Increase:               {growth:.1f}%")


if __name__ == '__main__':

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'convert_dataset_config.yaml')
    config = load_config(config_path)

    input_dir = config['input_path']
    output_dir = config['output_path']
    mode = config['conversion_mode']

    if mode == 'to_h5':
        convert_dataset_to_hdf5(input_dir, output_dir)
    elif mode == 'to_pkl':
        convert_dataset_to_pickle(input_dir, output_dir)
    else:
        print(f"Unknown conversion mode: {mode}")
