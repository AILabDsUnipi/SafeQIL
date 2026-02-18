from typing import NamedTuple, Tuple, Generator, Union
import numpy as np
import torch as th
from gymnasium import spaces

from safeqil_implementation.algos.icrl.utils import get_action_dim, get_obs_shape


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    old_reward_values: th.Tensor
    reward_advantages: th.Tensor
    reward_returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class RolloutBuffer(object):

    """
    Code based on:
    https://github.com/shehryar-malik/icrl/blob/master/stable_baselines3/common/buffers.py
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: th.device = th.device("cpu"),
            reward_gamma: float = 0.99,
            reward_gae_lambda: float = 1,
            cost_gamma: float = 0.99,
            cost_gae_lambda: float = 1
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.device = device

        # Get Parameters
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda

        self.observations = None
        self.actions = None
        self.advantages = None
        self.dones = None
        self.values = None
        self.log_probs = None

        # Rewards placeholders
        self.rewards = None
        self.reward_returns = None
        self.reward_values = None
        self.reward_advantages = None

        # Costs placeholders
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
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32) * np.nan
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32) * np.nan
        self.dones = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.log_probs = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        # Rewards
        self.rewards = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_returns = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_values = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.reward_advantages = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        # Costs
        self.costs = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.cost_returns = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.cost_values = np.array([np.nan]*self.buffer_size, dtype=np.float32)
        self.cost_advantages = np.array([np.nan]*self.buffer_size, dtype=np.float32)

        # Vars
        self.pos = 0
        self.full = False

    def _compute_returns_and_advantage(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
            gamma: float,
            gae_lambda: float,
            advantages: np.ndarray,
            last_value: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

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

            # 'dones' refers to the next states; however, 'values' refers to the current states.
            # So, in case of the last step during collecting the samples for multiple episodes,
            # we use 'dones[step]' and 'last_value', where the last value is estimated based on the
            # last observation which is not stored in the buffer.
            # Otherwise, we use 'dones[step]' and 'values[step + 1]'.
            if step == self.pos - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]

            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values

        return returns, advantages

    def compute_returns_and_advantage(
            self,
            last_reward_value: np.ndarray,
            last_cost_value: np.ndarray
    ) -> None:

        # Check the last reward value
        assert isinstance(last_reward_value, np.ndarray) and last_reward_value.shape == (1,)

        # Check the last cost value
        assert isinstance(last_cost_value, np.ndarray) and last_cost_value.shape == (1,)

        self.reward_returns, self.reward_advantages = self._compute_returns_and_advantage(
            self.rewards,
            self.reward_values,
            self.dones,
            self.reward_gamma,
            self.reward_gae_lambda,
            self.reward_advantages,
            last_reward_value
        )

        self.cost_returns, self.cost_advantages = self._compute_returns_and_advantage(
            self.costs,
            self.cost_values,
            self.dones,
            self.cost_gamma,
            self.cost_gae_lambda,
            self.cost_advantages,
            last_cost_value
        )

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: Union[np.float32, np.float64, float],
            cost: np.ndarray,
            done: bool,
            reward_value: np.ndarray,
            cost_value: np.ndarray,
            log_prob: np.ndarray
    ) -> None:

        """
        :param obs: Observation.
        :param action: Action.
        :param reward: Environment Reward.
        :param cost: Constraint-net Cost.
        :param done: End of the episode signal.
        :param reward_value: Estimated reward value of the current state following the current policy.
        :param cost_value: Estimated cost value of the current state following the current policy.
        :param log_prob: Log probability of the action following the current policy.
        """

        # Do some checks
        assert self.full is False
        assert isinstance(obs, np.ndarray) and obs.shape == self.obs_shape
        assert isinstance(action, np.ndarray) and action.shape == (self.action_dim,)
        assert isinstance(reward, (np.float32, np.float64, float))
        assert isinstance(cost, np.ndarray) and cost.shape == (1,)
        assert isinstance(done, bool)
        assert isinstance(reward_value, np.ndarray) and reward_value.shape == (1,)
        assert isinstance(cost_value, np.ndarray) and cost_value.shape == (1,)
        assert isinstance(log_prob, np.ndarray) and log_prob.shape == tuple()

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.dones[self.pos] = np.array([done])
        self.log_probs[self.pos] = log_prob[None, ...].copy()
        self.rewards[self.pos] = np.array([reward])
        self.reward_values[self.pos] = reward_value.copy()
        self.costs[self.pos] = cost.copy()
        self.cost_values[self.pos] = cost_value.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:

        assert isinstance(batch_size, int) and batch_size > 0
        indices = np.random.permutation(self.pos)

        start_idx = 0
        while start_idx + batch_size <= self.pos:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_idxs: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_idxs],
            self.actions[batch_idxs],
            self.log_probs[batch_idxs].flatten(),
            self.reward_values[batch_idxs].flatten(),
            self.reward_advantages[batch_idxs].flatten(),
            self.reward_returns[batch_idxs].flatten(),
            self.cost_values[batch_idxs].flatten(),
            self.cost_advantages[batch_idxs].flatten(),
            self.cost_returns[batch_idxs].flatten(),
        )
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

