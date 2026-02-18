import os
import pickle

import numpy as np
from torch.utils.data import Dataset
import torch as th  # Needed in __main__


def min_max_obs_values(env_id: str):
    # Note that these are computed based on the first dataset collected for each environment.
    # If unified datasets are used, these values might need to be updated.
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
    elif env_id == 'SafetyCarButton1-v0':
        max_obs = np.array([
            18.950063, 88.287286, 31.456283, 0.239206, 0.826327, 0.158559,  1.873314,
            1.945219,  3.020701,  0.500000,  0.500000, 0.035761, 10.704111, 11.985283,
            10.782499, 1.000000,  0.999991,  0.999818, 0.999992, 1.000000,  0.999845,
            0.999918,  0.999923,  1.000000,  0.919120, 0.906118, 0.900989,  0.900116,
            0.901042,  0.896516,  0.906431,  0.914928, 0.921210, 0.918017,  0.919927,
            0.908138,  0.908125,  0.916103,  0.920162, 0.921683, 0.898303,  0.890069,
            0.900989,  0.891175,  0.874480,  0.874066, 0.882910, 0.887538,  0.921210,
            0.918017,  0.920370,  0.909961,  0.909833, 0.920665, 0.920162,  0.900917,
            0.900305,  0.890964,  0.882231,  0.879215, 0.877057, 0.891933,  0.898728,
            0.907476,  0.907252,  0.899392,  0.889444, 0.890110, 0.888730,  0.898193,
            0.902458,  0.900724,  0.899589,  0.899043, 0.899946, 0.891608,  0.891504,
            0.900636,  0.908889,  0.907578,  0.896147, 0.881942, 0.888887,  0.867632,
            0.879379,  0.901447,  0.905185,  0.907771
        ])
        min_obs = np.array([
            -17.538469, -15.168458, -1.632738, -0.274202,  -0.968059, -0.182820,
            -1.880636,  -1.852639,  -3.341597, -0.500000,  -0.500000, -0.027144,
            -10.579345, -10.693819, -11.291259, -0.999989, -0.999702, -0.999895,
            -0.999930,  -0.999971,  -0.999974,  -0.999983, -0.999957, -0.999986,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000
        ])
    elif env_id == 'SafetyCarPush2-v0':
        max_obs = np.array([
            10.909340, 13.426229, 27.658043, 0.246056, 0.801182, 0.151610,  1.282954,
            2.006658,  2.747225,  0.500000,  0.500000, 0.036330, 11.340370, 10.444683,
            10.842868, 1.000000,  0.999645,  0.999958, 0.999982, 1.000000,  0.999956,
            0.999972,  0.999942,  1.000000,  0.759506, 0.791573, 0.788071,  0.835299,
            0.941747,  0.969066,  0.975719,  0.978838, 0.936292, 0.976964,  0.963351,
            0.939128,  0.818401,  0.843582,  0.837189, 0.767193, 0.863871,  0.868866,
            0.865288,  0.834815,  0.826935,  0.831620, 0.872696, 0.884135,  0.883703,
            0.872624,  0.848386,  0.829026,  0.822974, 0.814039, 0.851456,  0.863545,
            0.875463,  0.863680,  0.839042,  0.833072, 0.841572, 0.845014,  0.863378,
            0.874059,  0.874109,  0.870528,  0.867158, 0.813504, 0.812466,  0.846738,
            0.875504,  0.876775,  0.891719,  0.892428, 0.863205, 0.863975,  0.858261,
            0.857469,  0.867579,  0.884836,  0.896737, 0.896213, 0.893071,  0.900906,
            0.900914,  0.893626,  0.897450,  0.896683
        ])
        min_obs = np.array([
            -9.355898,  -13.822393, -0.268258,  -0.242027, -0.875333, -0.211282,
            -1.840608,  -2.402588,  -2.754489,  -0.500000, -0.500000, -0.041602,
            -10.645706, -10.860123, -10.314421, -0.999969, -0.999958, -0.999843,
            -0.999986,  -0.999910,  -0.999993,  -0.999937, -0.999991, -0.999906,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000,  0.000000,  0.000000,
            0.000000,   0.000000,   0.000000,   0.000000
        ])
    elif env_id == 'SafetyPointCircle2-v0':
        max_obs = np.array([
            2.892850, 19.349019, 9.810000, 1.338129, 0.034625, 0.000000, 0.000000,
            0.000000, 1.908738,  0.500000, 0.500000, 0.000000, 0.972775, 0.985324,
            0.988337, 0.988668,  0.987757, 0.984874, 0.972781, 0.949011, 0.966433,
            0.978545, 0.985076,  0.986158, 0.986492, 0.985007, 0.979528, 0.964325
        ])
        min_obs = np.array([
            -2.454631, -17.907113, 9.810000,  -0.041311, -0.742849, 0.000000,
            0.000000,  0.000000,   -1.910734, -0.500000, -0.500000, 0.000000,
            0.000000,  0.000000,   0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,   0.000000,  0.000000,  0.000000,  0.000000,
            0.000000,  0.000000,   0.000000,  0.000000
        ])
    else:
        raise ValueError(f"Environment {env_id} not supported.")

    return max_obs, min_obs


def normalize_features_func(features, env_id):
    max_obs, min_obs = min_max_obs_values(env_id)
    features = (features - min_obs) / (max_obs - min_obs + 1e-8)  # avoid division by zero

    return features


def min_max_rew_values(env_id: str):

    # Note that these are computed based on the first dataset collected for each environment.
    # If unified datasets are used, these values might need to be updated.
    if env_id == 'SafetyPointGoal1-v0':
        max_reward = 1.0278260468214948
        min_reward = -0.02666655807640672
    elif env_id == 'SafetyCarButton1-v0':
        max_reward = 1.0319995150405856
        min_reward = -0.032288244904923324
    elif env_id == 'SafetyCarPush2-v0':
        max_reward = 1.0321157327258565
        min_reward = -0.05873555213240278
    elif env_id == 'SafetyPointCircle2-v0':
        max_reward = 0.09170403341013715
        min_reward = -0.03984981189322493
    else:
        raise ValueError(f"Environment {env_id} not supported.")

    return max_reward, min_reward


def stats_discounted_rew_values(env_id: str, normalize_rewards: bool, demonstrations_path, gamma):

    # Normalize the path to the correct format for the OS
    norm_demonstrations_path = os.path.normpath(demonstrations_path)
    # Split the path into parts
    norm_demonstrations_path_parts = norm_demonstrations_path.split(os.sep)
    # Get the 4th last part
    demonstration_id = norm_demonstrations_path_parts[-4]

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
    elif (
            env_id == 'SafetyCarButton1-v0' and
            demonstration_id == 'human_alone_exp_human_data_simple_w_action_smooth=0.5_1' and
            gamma == 0.99
    ):
        if normalize_rewards is True:
            max_disc_reward = 8.104383849620868
            min_disc_reward = 0.014595633142959556
            median_disc_reward = 4.086479683815325
            mean_disc_reward = 3.9538680549431344
            twenty_five_quant_disc_reward = 3.364820086460876
            seventy_five_quant_disc_reward = 4.701916345872244
        else:
            max_disc_reward = 5.397722521742743
            min_disc_reward = -1.109909808873561
            median_disc_reward = 1.224493244591033
            mean_disc_reward = 1.2988687490749151
            twenty_five_quant_disc_reward = 0.6499148732253948
            seventy_five_quant_disc_reward = 1.8816637732090966
    elif (
            env_id == 'SafetyCarPush2-v0' and
            demonstration_id == 'human_alone_exp_human_data_simple_w_action_smooth=0.5_1' and
            gamma == 0.99
    ):
        if normalize_rewards is True:
            max_disc_reward = 8.088654149484768
            min_disc_reward = 0.030821119439550054
            median_disc_reward = 5.825079172550234
            mean_disc_reward = 5.476375972720478
            twenty_five_quant_disc_reward = 5.276501957155365
            seventy_five_quant_disc_reward = 6.240300288163425
        else:
            max_disc_reward = 2.9576927494180563
            min_disc_reward = -0.7150746124634568
            median_disc_reward = 0.6468967137421253
            mean_disc_reward = 0.6818134705414836
            twenty_five_quant_disc_reward = 0.27392176128999646
            seventy_five_quant_disc_reward = 1.043470033214057
    elif (
            env_id == 'SafetyPointCircle2-v0' and
            demonstration_id == 'human_alone_exp_human_data_simple_w_action_smooth=0.5_1' and
            gamma == 0.99
    ):
        if normalize_rewards is True:
            max_disc_reward = 87.27018204855494
            min_disc_reward = 0.4795425045825609
            median_disc_reward = 62.10992832047728
            mean_disc_reward = 57.66473553601139
            twenty_five_quant_disc_reward = 49.28007689482096
            seventy_five_quant_disc_reward = 69.8689903550011
        else:
            max_disc_reward = 7.757593354856117
            min_disc_reward = 0.023235853366441135
            median_disc_reward = 4.648938875844119
            mean_disc_reward = 4.384879077211347
            twenty_five_quant_disc_reward = 3.304774879489737
            seventy_five_quant_disc_reward = 5.517668643477849
    else:
        raise ValueError(
            f"Not supported combination!"
            f"\nEnvironment: {env_id}, demonstration id: {demonstration_id}, gamma: {gamma}."
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


def find_min_max_obs_from_demos(demonstrations_path: str, use_images: bool = False):
    """
    Scan demonstrations, compute and print per-feature min and max for observations.

    Returns:
        None
    """

    print('Computing max and min of observations features ...')

    obs_key = 'vision_obs' if use_images else 'vector_obs'
    min_obs = None
    max_obs = None
    num_files = 0
    num_obs = 0

    for file_name in os.listdir(demonstrations_path):
        if not file_name.endswith('.pkl'):
            continue
        with open(os.path.join(demonstrations_path, file_name), 'rb') as f:
            data = pickle.load(f)
            num_files += 1
            if obs_key not in data:
                raise KeyError(f"Key '{obs_key}' not found in {file_name}")

            for _, step_dict in data[obs_key].items():  # per episode
                for _, obs in step_dict.items():        # per step
                    obs = np.asarray(obs)
                    if min_obs is None:
                        # Initialize on the first observation we see
                        min_obs = obs.astype(np.float64).copy()
                        max_obs = obs.astype(np.float64).copy()
                    else:
                        if obs.shape != min_obs.shape:
                            raise ValueError(
                                f"Inconsistent obs shape: got {obs.shape}, expected {min_obs.shape}."
                            )
                        np.minimum(min_obs, obs, out=min_obs)
                        np.maximum(max_obs, obs, out=max_obs)
                    num_obs += 1

    if min_obs is None:
        raise RuntimeError(f"No observations found under: {demonstrations_path}")

    # Print statistics
    np.set_printoptions(suppress=True, floatmode='fixed', precision=6)  # For better readability
    print("max obs: ", max_obs)
    print("min obs: ", min_obs)


def find_min_max_reward_from_demos(demonstrations_path: str):
    """
    Scan demonstrations, compute and print global min and max over all rewards.

    Returns:
        None
    """

    print('Computing max and min of rewards ...')

    min_rew = np.inf
    max_rew = -np.inf
    found_any = False

    for file_name in os.listdir(demonstrations_path):
        if not file_name.endswith('.pkl'):
            continue
        with open(os.path.join(demonstrations_path, file_name), 'rb') as f:
            data = pickle.load(f)
            if 'reward' not in data:
                raise KeyError(f"Key 'reward' not found in {file_name}")

            for _, step_dict in data['reward'].items():  # per episode
                for _, r in step_dict.items():           # per step
                    r = float(r)
                    if r < min_rew: min_rew = r
                    if r > max_rew: max_rew = r
                    found_any = True

    if not found_any:
        raise RuntimeError(f"No rewards found under: {demonstrations_path}")

    print("max reward: ", float(max_rew))
    print("min reward: ", float(min_rew))


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
            compute_discounted_rewards=False,
            build_search_memory=False,
            search_func=None
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
        self.compute_discounted_rewards = compute_discounted_rewards
        self.build_search_memory = build_search_memory
        self.search_func = search_func

        self.idx_to_file_and_step = []
        self.data_store = {}

        # Get discounted rewards
        if self.compute_discounted_rewards is True:
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

        # If load_to_memory is True, we can build a memory for searching the closest states
        self.memory_built = False
        if self.build_search_memory is True:
            self._build_search_memory()
            assert self.search_func is not None, "Search function must be defined when building search memory."
            if self.search_func == "cosine_sim_w_discr_embed":
                # Placeholder for Discriminator
                self.discriminator = None

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
        disc_reward = (
            None if self.compute_discounted_rewards is False else self.all_discounted_rewards[episode][step_key]
        )

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
            th.from_numpy(actions).to(th.float32),
            th.from_numpy(observations).to(th.float32),
            th.from_numpy(np.array(done)).to(th.float32),
            th.from_numpy(np.array(reward, dtype=np.float32)).to(th.float32),
            th.from_numpy(np.array(disc_reward, dtype=np.float32)).to(th.float32),
            th.from_numpy(next_actions).to(th.float32),
            th.from_numpy(next_observations).to(th.float32),
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

    def _build_search_memory(self):
        """
        Build arrays in memory for quick retrieval of:
            (state, action, reward, done, next_state, next_action)
        This allows fast nearest-neighbor lookups (by cosine similarity).
        """

        if not self.load_to_memory:
            raise ValueError(
                "You must have 'load_to_memory=True' to allow in-memory searching. "
            )
        if self.search_func is None:
            raise ValueError("Search function must be defined when building search memory.")

        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_next_states = []
        all_next_actions = []

        # We'll also store the norm of each state for quick search in case of cosine similarity on raw observations
        all_states_norms = []

        # We iterate once through the entire dataset to build this memory
        for idx in range(len(self)):
            (
                actions,  # shape (act_dim,)
                observations,  # shape (obs_dim,)
                dones,
                rewards,
                disc_rewards,
                next_actions,
                next_observations,
            ) = self[idx]

            # Move them to CPU NumPy arrays (if they're still on GPU)
            obs_np = observations.cpu().numpy()
            act_np = actions.cpu().numpy()
            rew_np = rewards.cpu().numpy()
            done_np = dones.cpu().numpy()
            next_obs_np = next_observations.cpu().numpy()
            next_act_np = next_actions.cpu().numpy()

            all_states.append(obs_np)
            all_actions.append(act_np)
            all_rewards.append(rew_np)
            all_dones.append(done_np)
            all_next_states.append(next_obs_np)
            all_next_actions.append(next_act_np)

            if self.search_func == "cosine_sim":
                # Precompute norm of the state
                # Add a small epsilon to avoid division by zero
                norm_val = np.linalg.norm(obs_np) + 1e-8
                all_states_norms.append(norm_val)

        # Convert lists to NumPy arrays
        self._all_states = np.array(all_states)  # shape (N, obs_dim)
        self._all_actions = np.array(all_actions)  # shape (N, act_dim)
        self._all_rewards = np.array(all_rewards)  # shape (N,)
        self._all_dones = np.array(all_dones)  # shape (N,)
        self._all_next_states = np.array(all_next_states)  # shape (N, obs_dim)
        self._all_next_actions = np.array(all_next_actions)  # shape (N, act_dim)
        self._all_states_norms = np.array(all_states_norms)  # shape (N,) or ()

        self.memory_built = True
        print(f"Built in-memory search with {len(self._all_states)} samples.")

    def find_closest_states_batch(self, query_states: th.Tensor):
        """
        Given a batch of query states (shape: (batch_size, obs_dim)),
        find the closest demonstrated state for each query by cosine similarity.

        Returns:
            closest_states: shape (batch_size, obs_dim)
            closest_actions: shape (batch_size, act_dim)
            closest_rewards: shape (batch_size,)
            closest_dones: shape (batch_size,)
            closest_next_states: shape (batch_size, obs_dim)
            closest_next_actions: shape (batch_size, act_dim)
        """

        if self.use_images:
            raise ValueError(
                "Currently, find_closest_states_batch only supports vector observations. "
                "If you have images, flatten or embed them first."
            )
        if not self.memory_built:
            raise ValueError(
                "Memory not built. Ensure _build_search_memory() was called."
            )
        if self.search_func is None:
            raise ValueError("Search function must be defined when searching the closest state.")

        # We'll compute cosine similarity in a vectorized manner:
        # dot(Q, X) / (||Q|| * ||X||)
        #  - Q: shape (B, obs_dim)
        #  - X: shape (N, obs_dim)

        # 0) Get all the states in a vector format according to the 'search_func'
        if self.search_func == "cosine_sim":
            # Use the raw states (and precalculated norms for the demonstrated states)
            query_states_vector = query_states
            dem_states_vector = th.as_tensor(self._all_states, dtype=th.float32, device=self.device)
            dem_states_norms = th.as_tensor(self._all_states_norms, dtype=th.float32, device=self.device)
        elif self.search_func == "cosine_sim_w_discr_embed":
            query_states_vector = self.discriminator.embeddings(query_states)
            dem_states_vector = self.discriminator.embeddings(
                th.as_tensor(self._all_states, dtype=th.float32, device=self.device)
            )
            dem_states_norms = None
        else:
            raise ValueError(f"Search function '{self.search_func}' not supported.")

        # 1) Dot products ⇒ shape (B, N)
        dot_products = th.mm(query_states_vector, dem_states_vector.T)  # matrix multiplication

        # 2) Norm of query states ⇒ shape (B,)
        query_states_norms = th.linalg.norm(query_states_vector, ord=2, dim=1) + 1e-8

        # 3) Norm of dataset states ⇒ shape (N,)
        if dem_states_norms is None:
            dem_states_norms = th.linalg.norm(dem_states_vector, ord=2, axis=1) + 1e-8
        # Otherwise, already stored in self._all_states_norms

        # 4) Cosine similarity ⇒ shape (B, N)
        #    We expand dimensions so shapes line up in broadcast:
        #      query_states_norms ⇒ shape (B, 1)
        #      dem_states_norms ⇒ shape (1, N)
        cos_sim = dot_products / (query_states_norms.unsqueeze(1) * dem_states_norms.unsqueeze(0))

        # 5) Argmax along axis=1 => shape (B,)
        best_indices = th.argmax(cos_sim, dim=1)
        # Move to CPU for NumPy indexing
        best_indices = best_indices.cpu().numpy()

        # 6) Gather the best for each query
        closest_states = self._all_states[best_indices]  # (B, obs_dim)
        closest_actions = self._all_actions[best_indices]  # (B, act_dim)
        closest_rewards = self._all_rewards[best_indices]  # (B,)
        closest_dones = self._all_dones[best_indices]  # (B,)
        closest_next_states = self._all_next_states[best_indices]  # (B, obs_dim)
        closest_next_actions = self._all_next_actions[best_indices]  # (B, act_dim)

        return (
            th.as_tensor(closest_states, dtype=th.float32, device=self.device),
            th.as_tensor(closest_actions, dtype=th.float32, device=self.device),
            th.as_tensor(closest_rewards, dtype=th.float32, device=self.device),
            th.as_tensor(closest_dones, dtype=th.float32, device=self.device),
            th.as_tensor(closest_next_states, dtype=th.float32, device=self.device),
            th.as_tensor(closest_next_actions, dtype=th.float32, device=self.device),
        )


if __name__ == '__main__':

    _demonstrations_path = '/home/georgepap/PycharmProjects/ModelFreeSafeIL/experiments/safety_gymnasium/demonstrations/SafetyPointCircle2-v0/human_alone_exp_human_data_simple_w_action_smooth=0.5_1/test_results_seed=2023/tmp/demonstrations'

    # find_min_max_obs_from_demos(_demonstrations_path, use_images=False)
    # find_min_max_reward_from_demos(_demonstrations_path)
    find_step_wise_discounted_rewards_and_statistics(
        _demonstrations_path,
        normalize_reward=False,
        env_id="SafetyPointCircle2-v0",
        gamma=0.99
    )

    # # Test 'ExpertDataset' class
    # dataset = ExpertDataset(
    #     _demonstrations_path,
    #     use_images=False,
    #     load_to_memory=True,
    #     env_id="SafetyPointGoal1-v0",
    #     normalize_features=True,
    #     smooth_actions=False,
    #     smooth_factor=0.9,
    #     gamma=0.99
    # )
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
    # _all_actions, _all_observations = dataset.get_all_actions_and_observations()
    # print("actions: ", _all_actions)
    # print("observations: ", _all_observations)
