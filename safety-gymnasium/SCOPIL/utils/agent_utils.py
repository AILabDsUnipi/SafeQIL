from SCOPIL.rl_models.sac.sac_continuous_agent import SAC

import torch as th


def get_sac_agent(config, env, chkpt_dir=None, only_test=False, seed=None):

    device = config['Experiment']['device']
    device = th.device("cuda:0" if th.cuda.is_available() and device == "cuda" else "cpu")

    action_space = env.action_space
    if config['game']['use_image_obs'] is True:
        observation_space = env.observation_space.spaces['image']
    else:
        observation_space = env.observation_space

    sac = SAC(
        config=config,
        action_space=action_space,
        observation_space=observation_space,
        device=device,
        only_test=only_test,
        chkpt_dir=chkpt_dir,
        seed=seed
    )

    return sac
