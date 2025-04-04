import os
import pickle

import numpy as np
from torch.utils.data import Dataset
import torch as th  # Needed in __main__


def min_max_obs_values(env_id: str):

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


def normalize_features_func(features, env_id):
    max_obs, min_obs = min_max_obs_values(env_id)
    features = (features - min_obs) / (max_obs - min_obs + 1e-8)  # avoid division by zero

    return features


def min_max_rew_values(env_id: str):

    if env_id == 'SafetyPointGoal1-v0':
        max_reward = 1.0278260468214948
        min_reward = -0.02666655807640672
    else:
        raise ValueError(f"Environment {env_id} not supported.")

    return max_reward, min_reward


def stats_discounted_rew_values(env_id: str, normalize_rewards: bool, demonstrations_path, gamma):

    # Normalize the path to the correct format for the OS
    norm_demonstrations_path = os.path.normpath(demonstrations_path)
    # Split the path into parts
    norm_demonstrations_path_parts = norm_demonstrations_path.split(os.sep)
    # Get the second last part
    demonstration_id = norm_demonstrations_path_parts[-2]

    if (
            env_id == 'SafetyPointGoal1-v0' and
            demonstration_id == 'human_alone_exp_human_data_simple_w_action_smooth=0.5_1' and
            gamma == 0.99
    ):

        if normalize_rewards is True:
            max_disc_reward = 5.316081952483461
            min_disc_reward = 0.015163055896130572
            median_disc_reward = 3.4690537858122976
            mean_disc_reward = 3.2823386148022387
            twenty_five_quant_disc_reward = 2.977138070490666
            seventy_five_quant_disc_reward = 3.8384858536377635
        else:
            max_disc_reward = 2.9406837539017032
            min_disc_reward = -0.5044503041748538
            median_disc_reward = 1.1027654302168344
            mean_disc_reward = 1.0585335490560195
            twenty_five_quant_disc_reward = 0.6779178411930585
            seventy_five_quant_disc_reward = 1.4327907224436196

    else:
        raise ValueError(
            f"Not supported combination!"
            f"\nEnvironment: {env_id}, gamma: {gamma}."
        )

    return (
        max_disc_reward,
        min_disc_reward,
        median_disc_reward,
        mean_disc_reward,
        twenty_five_quant_disc_reward,
        seventy_five_quant_disc_reward
    )


def normalize_reward_func(reward, env_id):
    max_rew, min_rew = min_max_rew_values(env_id)
    reward = (reward - min_rew) / (max_rew - min_rew + 1e-8)  # avoid division by zero

    return reward


def calculate_discounted_rewards_per_step(rewards, normalize_reward, env_id, gamma=0.99):
    """
    Calculate the discounted rewards for each step within an episode.
    Args:
        rewards: List of rewards for one episode.
        normalize_reward: Boolean determining whether to normalize the rewards or not.
        env_id: String of the environment id.
        gamma: Discount factor.
    Returns:
        np.array of discounted rewards for each step.
    """
    n = len(rewards)
    discounted_rewards = np.zeros(n)
    cumulative_reward = 0
    # Compute discounted rewards in reverse
    for i in reversed(range(n)):
        if normalize_reward is True:
            reward = normalize_reward_func(rewards[i], env_id)
        else:
            reward = rewards[i]
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    return discounted_rewards


def find_step_wise_discounted_rewards_and_statistics(demonstrations_path, normalize_reward, env_id, gamma=0.99):
    """
    Find the discounted rewards for each step of each episode and compute statistics:
    maximum, minimum, mean, median, 25th quantile, and 75th quantile.
    """

    print('Computing discounted rewards ...')

    all_discounted_rewards = {}
    discounted_rewards_list = []

    for file_name in os.listdir(demonstrations_path):
        if file_name.endswith('.pkl'):
            with open(os.path.join(demonstrations_path, file_name), 'rb') as f:
                data = pickle.load(f)
                for episode_key in data['reward']:
                    rewards = list(data['reward'][episode_key].values())

                    all_discounted_rewards[episode_key] = {}
                    discounted_rewards = calculate_discounted_rewards_per_step(rewards, normalize_reward, env_id, gamma)
                    discounted_rewards_list.extend(discounted_rewards)

                    for step_counter in range(discounted_rewards.shape[0]):
                        all_discounted_rewards[episode_key][f'step_{step_counter}'] = discounted_rewards[step_counter]

    # Calculate statistics after gathering all discounted rewards
    max_discounted_reward = np.max(discounted_rewards_list)
    min_discounted_reward = np.min(discounted_rewards_list)
    median_discounted_reward = np.median(discounted_rewards_list)
    mean_discounted_reward = np.mean(discounted_rewards_list)
    quantile_25 = np.quantile(discounted_rewards_list, 0.25)
    quantile_75 = np.quantile(discounted_rewards_list, 0.75)

    # Print statistics
    print("Maximum Discounted Reward: ", max_discounted_reward)
    print("Minimum Discounted Reward: ", min_discounted_reward)
    print("Median Discounted Reward: ", median_discounted_reward)
    print("Mean Discounted Reward: ", mean_discounted_reward)
    print("25th Quantile of Discounted Reward: ", quantile_25)
    print("75th Quantile of Discounted Reward: ", quantile_75)

    return (
        all_discounted_rewards,
        max_discounted_reward,
        min_discounted_reward,
        median_discounted_reward,
        mean_discounted_reward,
        quantile_25,
        quantile_75
    )


class ExpertDataset(Dataset):
    def __init__(
            self,
            directory,
            device="cpu",
            use_images=False,
            load_to_memory=False,
            env_id=None,
            normalize_features=False,
            normalize_rewards=False,
            smooth_actions=False,
            smooth_factor=0.9,
            gamma=0.99,
    ):

        print("\nLoading demonstrations ...")

        self.directory = directory
        self.device = device
        self.use_images = use_images
        self.load_to_memory = load_to_memory
        self.env_id = env_id
        self.normalize_features = normalize_features
        self.normalize_rewards = normalize_rewards
        self.smooth_actions = smooth_actions
        self.smooth_factor = smooth_factor

        self.idx_to_file_and_step = []
        self.data_store = {}

        # Get discounted rewards
        self.all_discounted_rewards, *_ = find_step_wise_discounted_rewards_and_statistics(
            directory,
            self.normalize_rewards,
            self.env_id,
            gamma
        )

        # Build an index that maps a flat list index to (filename, episode, step_key)
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                episode = filename.split('.')[0]
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if self.load_to_memory is True:
                        # Store the data in memory
                        self.data_store[filepath] = {}
                        self.data_store[filepath]['done'] = data['done']
                        if self.use_images is True:
                            self.data_store[filepath]['vision_obs'] = data['vision_obs']
                            self.data_store[filepath]['actions'] = data['actions']
                            self.data_store[filepath]['reward'] = data['reward']
                        else:
                            # Normalize data here for more efficiency
                            self.data_store[filepath]['vector_obs'] = self.loaded_normalize_features_func(
                                data['vector_obs']
                            )
                            self.data_store[filepath]['actions'] = self.loaded_smooth_actions_func(
                                data['actions']
                            )
                            self.data_store[filepath]['reward'] = self.loaded_normalize_reward_func(
                                data['reward']
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
            features = normalize_features_func(features, self.env_id)

        return features

    def loaded_normalize_reward_func(self, reward):
        if self.normalize_rewards is True:
            for ep_key, ep_value in reward.items():  # Episode loop
                for st_key, st_value in ep_value.items():  # Step loop
                    reward[ep_key][st_key] = self.normalize_reward_func(st_value)

        return reward

    def normalize_reward_func(self, reward):
        if self.normalize_rewards is True:
            reward = normalize_reward_func(reward, self.env_id)

        return reward

    def __len__(self):
        return len(self.idx_to_file_and_step)

    def __getitem__(self, idx):

        # Get a valid sample, that is, 'done' exists. At the last step, the actions and observations are only stored,
        # so those samples (steps) are not valid
        data = None
        done = None
        episode = None
        step_key = None
        next_step_key = None
        done_exists = False
        while done_exists is False:

            filepath, episode, step_key = self.idx_to_file_and_step[idx]
            next_step_key = f"{step_key.split('_')[0]}_{int(step_key.split('_')[1])+1}"

            if self.load_to_memory is True:
                data = self.data_store[filepath]
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            if step_key in list(data['done'][episode].keys()):
                done = data['done'][episode][step_key]
                done_exists = True
            else:
                if idx < len(self.idx_to_file_and_step) - 1:
                    idx += 1
                else:
                    idx = 0

        # Discounted reward
        disc_reward = self.all_discounted_rewards[episode][step_key]

        ## Reward
        reward = data['reward'][episode][step_key]
        if self.load_to_memory is False:
            # When data are not loaded in memory should be normalized
            reward = self.normalize_reward_func(reward)

        ## Actions
        actions = data['actions'][episode][step_key]
        next_actions = np.zeros_like(actions)
        if not done:
            next_actions = data['actions'][episode][next_step_key]
        if self.load_to_memory is False:
            # When data are not loaded in memory should be scaled
            actions = self.smooth_actions_func(actions)
            if not done:
                next_actions = self.smooth_actions_func(next_actions)

        if self.use_images is True:
            observations = data['vision_obs'][episode][step_key]
            next_observations = np.zeros_like(observations)
            if not done:
                next_observations = data['vision_obs'][episode][next_step_key]
        else:
            observations = data['vector_obs'][episode][step_key]
            next_observations = np.zeros_like(observations)
            if not done:
                next_observations = data['vector_obs'][episode][next_step_key]
            if self.load_to_memory is False:
                # When data are not loaded in memory should be normalized
                observations = self.normalize_features_func(observations)
                if not done:
                    next_observations = self.normalize_features_func(next_observations)

        return (
            th.from_numpy(actions).to(th.float32).to(self.device),
            th.from_numpy(observations).to(th.float32).to(self.device),
            th.from_numpy(np.array(done)).to(th.float32).to(self.device),
            th.from_numpy(np.array(reward, dtype=np.float32)).to(th.float32).to(self.device),
            th.from_numpy(np.array(disc_reward, dtype=np.float32)).to(th.float32).to(self.device),
            th.from_numpy(next_actions).to(th.float32).to(self.device),
            th.from_numpy(next_observations).to(th.float32).to(self.device),
        )

    def get_all_actions_and_observations(self):
        all_actions = []
        all_observations = []

        for idx in range(len(self)):
            actions, observations, _, _, _, _, _ = self[idx]

            # Convert tensors to numpy arrays and append
            all_actions.append(actions.cpu().numpy())
            all_observations.append(observations.cpu().numpy())

        # Stack arrays into single numpy arrays
        all_actions = np.stack(all_actions)
        all_observations = np.stack(all_observations)

        return all_actions, all_observations


if __name__ == '__main__':

    _demonstrations_path = '/home/georgepap/PycharmProjects/ModelFreeSafeIL/experiments/safety_gymnasium/demonstrations/human_alone_exp_human_data_simple_w_action_smooth=0.5/tmp/human_alone_exp_human_data_simple_w_action_smooth=0.5_1/demonstrations'

    # find_step_wise_discounted_rewards_and_statistics(_demonstrations_path, True, "SafetyPointGoal1-v0")

    # # Test 'ExpertDataset' class
    dataset = ExpertDataset(
        _demonstrations_path,
        use_images=False,
        load_to_memory=True,
        env_id="SafetyPointGoal1-v0",
        normalize_features=True,
        smooth_actions=False,
        smooth_factor=0.9,
        gamma=0.99
    )
    # loader = th.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    #
    # import time
    # start_time = time.time()
    # for iter_idx, samples in enumerate(loader):
    #     acts, obs, dones, rews, disc_rews, next_acts, next_obs = samples
    #     print("\nactions: ", acts)
    #     print("obs: ", obs)
    #     print("dones: ", dones)
    #     print("rewards: ", rews)
    #     print("disc rewards: ", disc_rews)
    #     print("next_acts: ", next_acts)
    #     print("next_obs: ", next_obs)
    #     end_time = time.time()
    #     total_time = end_time - start_time
    #     print("total time: ", total_time)
    #     exit(0)

    # Test 'get_all_actions_and_observations' function
    _all_actions, _all_observations = dataset.get_all_actions_and_observations()
    print("actions: ", _all_actions)
    print("observations: ", _all_observations)
