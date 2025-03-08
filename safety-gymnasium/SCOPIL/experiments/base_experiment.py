import os
from abc import abstractmethod

import torch as th


class BaseExperiment:
    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):

        self.file_results_dir = file_results_dir
        self.agent = agent
        self.algo = None
        self.seed = seed

        # Set torch threads
        if 'device' in list(config['Experiment'].keys()):
            self.device = config['Experiment']['device']
        else:
            self.device = 'cpu'
        if self.device == 'cpu' and 'torch_threads' in list(config['Experiment'].keys()):
            th.set_num_threads(config['Experiment']['torch_threads'])
        else:
            th.set_num_threads(1)

        # Retrieve parameters
        self.config = config  # configuration file dictionary
        self.env = environment  # environment to play in
        self.test_model = config['Experiment']['test_model']  # check if only test
        if self.agent is not None:
            self.load_checkpoint = config['Experiment']['load_checkpoint']
            self.checkpoint_path = config['Experiment']['checkpoint_path']
            self.checkpoint_prefix = config['Experiment']['checkpoint_prefix']

        # Retrieve information from the config file only for test
        self.debug_ = config['Experiment']['debug_']
        self.test_window_size_moving_avg = config['Experiment']['test_window_size_moving_avg']
        self.test_max_games = config['Experiment']['test_max_games']
        self.test_max_timesteps_per_game = config['Experiment']['test_max_timesteps_per_game']

        # Dummy env reset and step just to get the constraint types
        _ = self.env.reset()
        _, _, _, _, _, info = self.env.step([0., 0.])
        self.constraint_types = [  # NOTE: 'cost_sum' doesn't sum all costs, it's just 0 or 1
            constraint_type for constraint_type in info.keys() if 'cost' in constraint_type
        ]

        # Set the env seed
        self.env.set_seed(self.seed)

        # Initialize lists to keep track of information and variables during testing
        self.total_steps = 0
        self.test_game_number = 0
        self.test_reward_list = []
        self.test_num_constraint_violation_list = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.test_freq_constraint_violation_list = {
            constraint_type: [] for constraint_type in self.constraint_types
        }
        self.test_step_list = []
        if self.test_model is True or self.debug_ is True:
            # Each element has a dictionary in the form: {'episode_<i>': {'step_<j>': [...], ...}, ...}
            self.test_history = {
                'actions': {},
                'vector_obs': {},
                'vision_obs': {},
                "reward": {},
                "cost": {},
                "done": {},
                "fixed_done": {},  # 0 if time-out else float(done)
                "truncated": {},
                "info": {}
            }

    @abstractmethod
    def save_info(self, chkpt_dir):
        """
        Saves experiment additional information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        """

        raise NotImplementedError
