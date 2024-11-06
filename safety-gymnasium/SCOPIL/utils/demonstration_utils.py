import os
import pickle

import numpy as np
from torch.utils.data import Dataset
import torch as th  # Needed in __main__


def min_max_obs_values(env_id: str):

    max_obs = None
    min_obs = None
    if env_id == 'SafetyPointGoal1-v0':
        max_obs = np.array([
            3.24462421, 13.7011033, 9.81,       1.46503807, 0.64846094, 0.,
            0.,         0.,         3.01406257, 0.5,        0.49999999, 0.,
            0.89999832, 0.89981992, 0.899368,   0.8164237,  0.76146773, 0.7631361,
            0.76483541, 0.76579285, 0.76607739, 0.76598412, 0.76515357, 0.79237079,
            0.89992099, 0.89998301, 0.89999353, 0.89997067, 0.85350947, 0.87612177,
            0.90060856, 0.92215829, 0.92553072, 0.90321011, 0.88815048, 0.87873831,
            0.88828625, 0.90121941, 0.90411837, 0.91049685, 0.9098528,  0.89555713,
            0.86725519, 0.83502169, 0.85010174, 0.86352798, 0.89289803, 0.90427331,
            0.89711691, 0.88755617, 0.88579163, 0.86439939, 0.86757475, 0.87934745,
            0.90379567, 0.90750787, 0.90582002, 0.89787418, 0.85940369, 0.84051326
        ])
        min_obs = np.array([
            -2.71253937, -13.65950487, 9.81,        -0.19734316, -0.66497254, 0.,
            0.,          0.,           -3.01408301, -0.49999999, -0.5,        0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.,
            0.,          0.,           0.,          0.,          0.,          0.
        ])
    else:
        raise ValueError(f"Environment {env_id} not supported.")

    return max_obs, min_obs


def min_max_rew_values(env_id: str):

    max_reward = None
    min_reward = None
    if env_id == 'SafetyPointGoal1-v0':
        max_reward = 1.0278260468214948
        min_reward = -0.02666655807640672
    else:
        raise ValueError(f"Environment {env_id} not supported.")

    return max_reward, min_reward


def find_min_max_observation_reward(demonstrations_path: str) -> None:
    """
    Find the minimum and maximum observation values (separately for each element of the vector)
    and the minimum and maximum reward in the demonstrations.
    Args:
        demonstrations_path: Path to directory containing demonstrations in pickle files.
    """

    min_obs = None
    max_obs = None
    min_reward = float('inf')
    max_reward = float('-inf')

    for file_name in os.listdir(demonstrations_path):
        if file_name.endswith('.pkl'):
            with open(os.path.join(demonstrations_path, file_name), 'rb') as f:
                data = pickle.load(f)
                for episode_key in data['reward']:
                    rewards = data['reward'][episode_key]
                    observations = data['vector_obs'][episode_key]

                    for step_key in rewards:
                        reward = rewards[step_key]
                        obs = observations[step_key]

                        if min_obs is None:
                            min_obs = np.array(obs)
                            max_obs = np.array(obs)
                        else:
                            min_obs = np.minimum(min_obs, obs)
                            max_obs = np.maximum(max_obs, obs)

                        min_reward = min(min_reward, reward)
                        max_reward = max(max_reward, reward)

    print("Maximum observation values: ", max_obs)
    print("Minimum observation values: ", min_obs)
    print("Maximum reward: ", max_reward)
    print("Minimum reward: ", min_reward)


class ExpertDataset(Dataset):
    def __init__(
            self,
            directory,
            device="cpu",
            use_images=False,
            load_to_memory=False,
            env_id=None,
            normalize_features=False,
            smooth_actions=False,
            smooth_factor=0.9
    ):

        self.directory = directory
        self.device = device
        self.use_images = use_images
        self.load_to_memory = load_to_memory
        self.env_id = env_id
        self.normalize_features = normalize_features
        self.smooth_actions = smooth_actions
        self.smooth_factor = smooth_factor

        self.idx_to_file_and_step = []
        self.data_store = {}

        # Build an index that maps a flat list index to (filename, episode, step_key)
        print("Loading demonstrations ...")
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                episode = filename.split('.')[0]
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if self.load_to_memory is True:
                        # Store the data in memory
                        self.data_store[filepath] = {}
                        if self.use_images is True:
                            self.data_store[filepath]['vision_obs'] = data['vision_obs']
                            self.data_store[filepath]['actions'] = data['actions']
                        else:
                            # Normalize data here for more efficiency
                            self.data_store[filepath]['vector_obs'] = self.loaded_normalize_features_func(
                                data['vector_obs']
                            )
                            self.data_store[filepath]['actions'] = self.loaded_smooth_actions_func(
                                data['actions']
                            )
                    for step in data['actions'][episode].keys():  # Assuming all keys are the same across the dicts
                        self.idx_to_file_and_step.append((filepath, episode, step))
        print("{} samples were mapped successfully!".format(len(self.idx_to_file_and_step)))

    def smooth_actions_func(self, actions):
        if self.smooth_actions is True:
            actions = actions * self.smooth_factor

        return actions

    def loaded_smooth_actions_func(self, actions):
        if self.smooth_actions is True:
            for ep_key, ep_value in actions.items():  # Episode loop
                for st_key, st_value in ep_value.items():  # Step loop
                    actions[ep_key][st_key] = self.smooth_actions_func(st_value)

        return actions

    def loaded_normalize_features_func(self, features):
        if self.normalize_features is True:
            for ep_key, ep_value in features.items():  # Episode loop
                for st_key, st_value in ep_value.items():  # Step loop
                    features[ep_key][st_key] = self.normalize_features_func(st_value)

        return features

    def normalize_features_func(self, features):
        if self.normalize_features is True:
            max_obs, min_obs = min_max_obs_values(self.env_id)
            features = (features - min_obs) / (max_obs - min_obs + 1e-8)

        return features

    def __len__(self):
        return len(self.idx_to_file_and_step)

    def __getitem__(self, idx):
        filepath, episode, step_key = self.idx_to_file_and_step[idx]

        if self.load_to_memory is True:
            data = self.data_store[filepath]
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        actions = data['actions'][episode][step_key]
        if self.load_to_memory is False:
            # When data are not loaded in memory should be scaled
            actions = self.smooth_actions_func(actions)

        if self.use_images is True:
            observations = data['vision_obs'][episode][step_key]
        else:
            observations = data['vector_obs'][episode][step_key]
            if self.load_to_memory is False:
                # When data are not loaded in memory should be normalized
                observations = self.normalize_features_func(observations)

        return (
            th.from_numpy(actions).to(th.float32).to(self.device),
            th.from_numpy(observations).to(th.float32).to(self.device)
        )


if __name__ == '__main__':

    _demonstrations_path = '/home/georgepap/PycharmProjects/HAI-MAZE_master/experiments/safety_gymnasium/human_alone_exp_human_data_simple/tmp/human_alone_exp_human_data_simple_1/demonstrations'

    # find_min_max_observation_reward(_demonstrations_path)
    dataset = ExpertDataset(
        _demonstrations_path,
        use_images=False,
        load_to_memory=True,
        env_id="SafetyPointGoal1-v0",
        normalize_features=True,
        smooth_actions=False,
        smooth_factor=0.9
    )
    loader = th.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    import time
    start_time = time.time()
    for samples in enumerate(loader):
        acts, obs = samples
        print("\nactions: ", acts)
        print("obs: ", obs)
        end_time = time.time()
        total_time = end_time - start_time
        print("total time: ", total_time)
        exit(0)
