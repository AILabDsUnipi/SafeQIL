import os
import pickle
import yaml
from tqdm import tqdm

import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_sorted_files(directory):
    """
    Returns a list of .pkl files sorted numerically.
    Standard os.listdir is arbitrary, and standard sort is lexicographical
    (1, 10, 2), so we sort by the integer value of the filename.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    try:
        # Assumes files are named "0.pkl", "1.pkl", etc.
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Fallback if filenames are not purely integers (e.g. "episode_0.pkl")
        files.sort()
        print(f"\nWarning: Could not sort files numerically in {directory}. Using alphanumeric sort.")
    return files


def unify_datasets():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'unify_datasets_config.yaml')
    config = load_config(config_path)

    output_dir = config['output_path']
    dataset_paths = config['datasets']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    global_episode_counter = 0

    # Lists to store totals per episode
    all_episode_rewards = []
    all_episode_costs = []

    print(f"\nTarget Output Directory: {output_dir}")
    print(f"Found {len(dataset_paths)} datasets to merge.\n")

    for d_idx, source_path in enumerate(dataset_paths):

        assert os.path.exists(source_path), f"Error: Path does not exist: {source_path}"

        files = get_sorted_files(source_path)
        print(f"\nProcessing Dataset {d_idx + 1}: {source_path} ({len(files)} episodes)")

        for filename in tqdm(files, desc=f"Merging Dataset {d_idx + 1}", unit="ep"):
            source_file_path = os.path.join(source_path, filename)

            # 1. Load the original data
            try:
                with open(source_file_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load {filename}: {e}")

            # 2. Create a new data structure with updated keys
            new_data = {}

            # The structure is: data['reward']['0'] = ...
            # We need to preserve the top keys (reward, actions, etc)
            # but change the inner key (episode ID) to match global_episode_counter

            # We extract the 'old' episode ID from the keys.
            # There is only 1 episode per file.

            # Let's inspect the keys to find the episode ID used in this file
            # We look at 'done' or 'reward' as they are guaranteed to exist
            sample_key = 'done' if 'done' in data else 'reward'
            if sample_key not in data:
                raise ValueError(f"Filename '{filename}': Structure unknown (no 'done' or 'reward' key).")

            old_episode_ids = list(data[sample_key].keys())

            # Usually len(old_episode_ids) is 1. If multiple, we handle them sequentially.
            for old_ep_id in old_episode_ids:

                # Calculate Stats for this episode
                # 1. Sum Rewards
                assert 'reward' in data and old_ep_id in data['reward'], (
                    f"Filename '{filename}': Missing reward data for episode ID '{old_ep_id}'."
                )
                # data['reward'][id] is a dict of {step: value}
                ep_rew_sum = sum(data['reward'][old_ep_id].values())
                all_episode_rewards.append(ep_rew_sum)

                # 2. Sum Costs
                assert 'cost' in data and old_ep_id in data['cost'], (
                    f"Filename '{filename}': Missing cost data for episode ID '{old_ep_id}'."
                )
                ep_cost_sum = sum(data['cost'][old_ep_id].values())
                all_episode_costs.append(ep_cost_sum)

                # Copy and Transform
                for top_level_key, content in data.items():
                    # Handle dictionary structures (actions, obs, rewards, done)
                    if isinstance(content, dict):
                        if top_level_key not in new_data:
                            new_data[top_level_key] = {}

                        # Copy the content from old ID to new global ID
                        if old_ep_id in content:
                            new_data[top_level_key][str(global_episode_counter)] = content[old_ep_id]

                    # Handle non-dict metadata (if any exists)
                    else:
                        new_data[top_level_key] = content

                # 3. Save to the new unified directory
                new_filename = f"{global_episode_counter}.pkl"
                dest_file_path = os.path.join(output_dir, new_filename)

                with open(dest_file_path, 'wb') as f:
                    pickle.dump(new_data, f)

                global_episode_counter += 1

    # Compute and Save Statistics
    stats_file_path = os.path.join(output_dir, 'stats.txt')

    with open(stats_file_path, 'w') as f:
        f.write(f"Total Episodes: {global_episode_counter}\n")
        f.write("------------------------------------------------\n")

        # Rewards Stats
        assert all_episode_rewards, "Error: No reward data collected from any episodes."
        mean_rew = np.mean(all_episode_rewards)
        std_rew = np.std(all_episode_rewards)
        f.write(f"Reward Mean: {mean_rew:.4f}\n")
        f.write(f"Reward Std:  {std_rew:.4f}\n")

        # Costs Stats
        assert all_episode_costs, "Error: No cost data collected from any episodes."
        mean_cost = np.mean(all_episode_costs)
        std_cost = np.std(all_episode_costs)
        f.write(f"Cost Mean:   {mean_cost:.4f}\n")
        f.write(f"Cost Std:    {std_cost:.4f}\n")

    print("\n------------------------------------------------")
    print(f"Unification Complete.")
    print(f"Total Episodes: {global_episode_counter}")
    print(f"Saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":

    unify_datasets()
