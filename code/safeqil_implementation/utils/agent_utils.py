import os
import copy

import numpy as np
import torch as th
from gymnasium import spaces

from safeqil_implementation.algos.sac.sac_continuous_agent import SAC
from safeqil_implementation.algos.icrl.ppo_continuous_agent import ContinuousPPOAgent
from safeqil_implementation.algos.icrl.constraint_net import ConstraintNet
from safeqil_implementation.algos.vicrl.variational_constraint_net import VariationalConstraintNet
from safeqil_implementation.utils.exp_utils import set_random_seed


def get_agent_info(config, env):
    device = config['Experiment']['device']
    device = th.device("cuda:0" if th.cuda.is_available() and device == "cuda" else "cpu")

    action_space = env.action_space
    if config['game']['use_image_obs'] is True:
        observation_space = env.observation_space.spaces['image']
    else:
        observation_space = env.observation_space

    return device, action_space, observation_space


def get_sac_agent(config, env, only_test=False, seed=None):

    device, action_space, observation_space = get_agent_info(config, env)

    sac = SAC(
        config=config,
        action_space=action_space,
        observation_space=observation_space,
        device=device,
        only_test=only_test,
        seed=seed
    )

    # In case of testing, set the eval mode
    if only_test is True:
        sac.policy.set_training_mode(False)

    return sac


def get_icrl_agent(config, env, only_test=False, seed=None):

    device, action_space, observation_space = get_agent_info(config, env)

    icrl = ICRLAg(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
        only_test=only_test,
        device=device,
        seed=seed
    )

    return icrl


class ICRLAg(object):

    def __init__(
            self,
            config,
            observation_space,
            action_space,
            only_test,
            device,
            seed
    ):
        # Set the randomness here before creating the networks
        set_random_seed(seed, using_cuda=device.type == th.device("cuda").type)

        # Define the PPO agent
        self.ppo = ContinuousPPOAgent(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            only_test=only_test,
            device=device,
        )

        # Define the constraint net
        self.constraint_net = self.define_constraint_net(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            only_test=only_test,
            device=device,
        )

        # In case of testing, set the eval mode
        if only_test is True:
            self.ppo.policy.eval()
            self.constraint_net.eval()

    @staticmethod
    def define_constraint_net(
            config,
            observation_space,
            action_space,
            only_test,
            device
    ):
        constraint_net = ConstraintNet(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            only_test=only_test,
            device=device,
        )

        return constraint_net

    def save(self, prefix, path):
        """
        Save the agent models
        :param prefix: str, prefix name for the models
        :param path: str, path to the directory where the models will be stored
        """

        # Create the folder if needed
        path = os.path.join(path, 'chkpts')
        if os.path.exists(path) is False:
            os.mkdir(path)

        self.ppo.save_models(prefix, path)
        self.constraint_net.save_models(prefix, path)

    def load(self, prefix, path):
        """
        Save the agent models
        :param prefix: str, prefix name for the models
        :param path: str, path to the directory from where the models will be loaded
        """
        self.ppo.load_models(prefix, path)
        self.constraint_net.load_models(prefix, path)

    def clip_action(self, action):
        clipped_actions = action
        # Clip the actions to avoid out of bound error
        if isinstance(self.ppo.action_space, spaces.Box):
            clipped_actions = np.clip(
                action,
                self.ppo.action_space.low,
                self.ppo.action_space.high
            )
        return clipped_actions

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action = self.ppo.policy.predict(obs, deterministic=deterministic)
        action = self.clip_action(action)

        return action


def get_vicrl_agent(config, env, only_test=False, seed=None):

    device, action_space, observation_space = get_agent_info(config, env)

    vicrl = VICRLAg(
        config=config,
        observation_space=observation_space,
        action_space=action_space,
        only_test=only_test,
        device=device,
        seed=seed
    )

    return vicrl


class VICRLAg(ICRLAg):

    def __init__(
            self,
            config,
            observation_space,
            action_space,
            only_test,
            device,
            seed
    ):

        # Create an 'ICRL' field and copy the 'VICRL' since 'ContinuousPPOAgent'
        # gets the hyperparameters from 'ICRL' field
        config['ICRL'] = copy.deepcopy(config['VICRL'])
        super(VICRLAg, self).__init__(
            config,
            observation_space,
            action_space,
            only_test,
            device,
            seed
        )

    @staticmethod
    def define_constraint_net(
            config,
            observation_space,
            action_space,
            only_test,
            device
    ):
        constraint_net = VariationalConstraintNet(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            only_test=only_test,
            device=device,
        )

        return constraint_net
