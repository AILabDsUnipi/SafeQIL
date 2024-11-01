import numpy as np
import torch

def transform_action_to_actions(action, n_actions):
    """
    Transforms a single agent's discrete action (index) to two agents' discrete actions (two indices).
    The first agent's action is the div of the single action while the second agent's action is the mod
    of the single action.
    Actions will be represented as follows:
    0: [0, 0]
    1: [0, 1]
    2: [0, 2]
    3: [1, 0]
    4: [1, 1]
    5: [1, 2]
    6: [2, 0]
    7: [2, 1]
    8: [2, 2]
    @param action: A single int.
    @param n_actions: Total number of action of each agent (same for both).
    @return: A list with two int.
    """

    return [action // n_actions, action % n_actions]

def transform_actions_to_action(actions, n_actions):
    """
    Transforms two agents' discrete actions (two indices) to a single agent's discrete action (one index).
    The single action is calculated as: num_actions*first_action + second_action.
    Actions will be represented as follows:
    [0, 0]: 0
    [0, 1]: 1
    [0, 2]: 2
    [1, 0]: 3
    [1, 1]: 4
    [1, 2]: 5
    [2, 0]: 6
    [2, 1]: 7
    [2, 2]: 8
    @param actions: A list with two int, or a torch.Tensor of size=[batch_size, 2],
                    or a numpy.array with shape=[batch_size, 2].
    @param n_actions: Total number of actions of each agent (same for both).
    @return: A single int, or a torch.Tensor of size=[batch_size, 1] and dtype=torch.int64,
             or a numpy.array with shape=[batch_size, 1] and dtype=np.int.
    """

    if isinstance(actions[0], int) and isinstance(actions[1], int):
        res = (n_actions * actions[0]) + actions[1]
    elif isinstance(actions, torch.Tensor) and len(actions.size()) == 2 and actions.size(1) == 2:
        res = ((n_actions * actions[:, 0]) + actions[:, 1]).to(dtype=torch.int64).unsqueeze(dim=1)
    elif isinstance(actions, np.ndarray) and len(actions.shape) == 2 and actions.shape[1] == 2:
        res = np.expand_dims(np.array((n_actions * actions[:, 0]) + actions[:, 1], dtype=np.int), axis=1)
    else:
        raise NotImplementedError

    return res
