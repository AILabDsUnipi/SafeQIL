from rl_models.sac_discrete.sac_discrete_agent import DiscreteSACAgent
from rl_models.utils.algo_utils import transform_actions_to_action

import torch


def get_sac_agent(config, env, chkpt_dir=None, axis_agent='X', only_test=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    action_dim = env.action_space.actions_number
    if axis_agent == 'X_Y':
        # In this case, there is a single agent that moves both axes.
        action_dim **= 2

    sac = DiscreteSACAgent(config=config,
                           input_dims=env.observation_shape,
                           n_actions=action_dim,
                           chkpt_dir=chkpt_dir,
                           axis_agent=axis_agent,
                           only_test=only_test,
                           device=device,
                           transform_actions_to_action_func=transform_actions_to_action)

    return sac
