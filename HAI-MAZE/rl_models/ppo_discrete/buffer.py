from typing import NamedTuple, Tuple, Generator, Optional
import numpy as np
import torch as th


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    old_reward_values: th.Tensor
    reward_advantages: th.Tensor
    reward_returns: th.Tensor
    old_cost_values: Optional[th.Tensor] = None
    cost_advantages: Optional[th.Tensor] = None
    cost_returns: Optional[th.Tensor] = None


class RolloutBuffer(object):

    """
    Code based on:
    https://github.com/shehryar-malik/icrl/blob/master/stable_baselines3/common/buffers.py
    """

    def __init__(self,
                 buffer_size: int,
                 obs_shape: int,
                 n_actions: int,
                 device: th.device = th.device("cpu"),
                 reward_gamma: float = 0.99,
                 reward_gae_lambda: float = 1,
                 cost_gamma: float = 0.99,
                 cost_gae_lambda: float = 1,
                 icrl: bool = False):

        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.icrl = icrl

        # Get Parameters
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        if self.icrl is True:
            self.cost_gamma = cost_gamma
            self.cost_gae_lambda = cost_gae_lambda

        self.observations = None
        self.actions = None
        self.advantages = None
        self.dones = None
        self.values = None
        self.log_probs = None

        # Declare Rewards
        self.rewards = None
        self.reward_returns = None
        self.reward_values = None
        self.reward_advantages = None

        if self.icrl is True:
            # Declare Costs
            self.cost_returns = None
            self.costs = None
            self.cost_values = None
            self.cost_advantages = None

        # Initialize vars
        self.pos = 0
        self.full = False

        # Initialize all
        self.reset()

    def reset(self) -> None:

        # Initialize with nans to identify bugs and fix them easier.

        self.observations = np.array([[np.nan]*self.obs_shape]*self.buffer_size, dtype=np.float32)
        self.actions = np.array([[np.nan]]*self.buffer_size, dtype=np.float32)
        self.dones = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.log_probs = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        # Rewards
        self.rewards = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_returns = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_values = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_advantages = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        if self.icrl is True:
            # Costs
            self.costs = np.array([np.nan]*self.buffer_size, dtype=np.float32)
            self.cost_returns = np.array([np.nan]*self.buffer_size, dtype=np.float32)
            self.cost_values = np.array([np.nan]*self.buffer_size, dtype=np.float32)
            self.cost_advantages = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        # Vars
        self.pos = 0
        self.full = False

    def _compute_returns_and_advantage(self,
                                       rewards: np.ndarray,
                                       values: np.ndarray,
                                       dones: np.ndarray,
                                       gamma: float,
                                       gae_lambda: float,
                                       advantages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.
        """

        last_gae_lam = 0
        for step in reversed(range(self.pos)):

            if step == self.pos - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step]
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]

            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values

        return returns, advantages

    def compute_returns_and_advantage(self) -> None:

        self.reward_returns, self.reward_advantages = \
            self._compute_returns_and_advantage(self.rewards,
                                                self.reward_values,
                                                self.dones,
                                                self.reward_gamma,
                                                self.reward_gae_lambda,
                                                self.reward_advantages)

        if self.icrl is True:
            self.cost_returns, self.cost_advantages = \
                self._compute_returns_and_advantage(self.costs,
                                                    self.cost_values,
                                                    self.dones,
                                                    self.cost_gamma,
                                                    self.cost_gae_lambda,
                                                    self.cost_advantages)

    def add(self,
            obs: th.Tensor,
            action: th.Tensor,
            reward: np.ndarray,
            cost: Optional[np.ndarray],
            done: np.ndarray,
            reward_value: th.Tensor,
            cost_value: Optional[th.Tensor],
            log_prob: th.Tensor) -> None:

        """
        :param obs: Observation
        :param action: Action
        :param reward: Environment Reward
        :param cost: Constraint-net Cost
        :param done: End of episode signal.
        :param reward_value: estimated reward value of the current state following the current policy.
        :param cost_value: estimated cost value of the current state following the current policy.
        :param log_prob: log probability of the action following the current policy.
        """

        assert self.full is False, ""
        assert obs.shape == (self.obs_shape,), ""
        assert action.shape == (1, 1), ""
        assert reward.shape == (1,), ""
        assert cost is None or cost.shape == (1,), ""
        assert done.shape == (1,), ""
        assert reward_value.shape == (1,), ""
        assert cost_value is None or cost_value.shape == (1,), ""
        assert log_prob.shape == (1,), ""

        self.observations[self.pos] = obs.clone().cpu().numpy()
        self.actions[self.pos] = action.clone().cpu().numpy()
        self.dones[self.pos] = done.copy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.rewards[self.pos] = reward.copy()
        self.reward_values[self.pos] = reward_value.clone().cpu().numpy()
        if self.icrl is True:
            self.costs[self.pos] = cost.copy()
            self.cost_values[self.pos] = cost_value.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int = None) -> Generator[RolloutBufferSamples, None, None]:

        indices = np.random.permutation(self.pos)

        start_idx = 0
        while start_idx + batch_size <= self.pos:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_idxs: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_idxs],
            self.actions[batch_idxs],
            self.log_probs[batch_idxs],
            self.reward_values[batch_idxs],
            self.reward_advantages[batch_idxs],
            self.reward_returns[batch_idxs],
               ) \
               + \
               (() if self.icrl is False
                   else
                (
                self.cost_values[batch_idxs],
                self.cost_advantages[batch_idxs],
                self.cost_returns[batch_idxs],
                ))
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
                     (might be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

