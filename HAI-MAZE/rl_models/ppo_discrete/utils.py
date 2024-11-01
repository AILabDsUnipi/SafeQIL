import numpy as np

from rl_models.ppo_discrete.ppo_discrete_agent import DiscretePPOAgent
from rl_models.ppo_discrete.constraint_net import ConstraintNet
from rl_models.utils.expert_dataset import exprt_dataset
from rl_models.utils.algo_utils import transform_actions_to_action

import torch

def get_ppo_agent(config, env, chkpt_dir=None, axis_agent='X', only_test=False):

    ppo = PPOAg(config, env, chkpt_dir, axis_agent, only_test)
    return ppo

class PPOAg(object):

    def __init__(self, config, env, chkpt_dir, axis_agent, only_test):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        action_dim = env.action_space.actions_number
        if axis_agent == 'X_Y':
            # In this case, there is a single agent which move both axes.
            action_dim **= 2

        self.axis_agent = axis_agent
        self.ppo = DiscretePPOAgent(config=config,
                                    observation_space=env.observation_shape[0],
                                    action_space=action_dim,
                                    chkpt_dir=chkpt_dir,
                                    axis_agent=axis_agent,
                                    only_test=only_test,
                                    device=self.device)

        if self.ppo.icrl is True or self.ppo.lagrangian is True or self.ppo.SL_finetune is True:

            allInputs = None
            allActions = None
            if only_test is False:
                # Define torch dataset for demonstrations
                expert_dataset = exprt_dataset(config['PPO']['expert_dataset_paths'])
                # Get all samples
                allInputs, allActions = expert_dataset.getAll()
                if axis_agent == 'X':
                    # Keep only first dimension of actions which corresponds to X-axis
                    allActions = allActions[:, 0]
                elif axis_agent == 'Y':
                    # Keep only second dimension of actions which corresponds to Y-axis
                    allActions = allActions[:, 1]
                elif axis_agent == 'X_Y':
                    # Convert two-dimensional actions to one-dimensional actions
                    allActions = transform_actions_to_action(allActions, np.sqrt(action_dim)).squeeze(1)

            if self.ppo.icrl is True:
                self.constraint_net = ConstraintNet(config=config,
                                                    obs_space=env.observation_shape[0],
                                                    n_actions=action_dim,
                                                    expert_obs=allInputs,
                                                    expert_acs=allActions,
                                                    device=self.device,
                                                    chkpt_dir=chkpt_dir,
                                                    axis_agent=axis_agent,
                                                    only_test=only_test)
            elif (self.ppo.lagrangian is True or self.ppo.SL_finetune is True) and only_test is False:
                self.ppo.expert_obs = allInputs
                self.ppo.expert_acts = allActions

    def save_models(self, override):
        self.ppo.save_models(override)
        if self.ppo.icrl is True:
            assert override is True, "Not overriding option when saving the models is only applicable when in 'SL_finetuning' mode."
            self.constraint_net.save_models()

    def load_models(self, load_checkpoint_path_name):
        self.ppo.load_models(load_checkpoint_path_name)
        if self.ppo.icrl is True:
            self.constraint_net.load_models(load_checkpoint_path_name)
