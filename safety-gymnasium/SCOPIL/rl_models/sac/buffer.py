from abc import ABC
from typing import Any, Dict, Union, Tuple, NamedTuple

import numpy as np
import torch as th
import psutil
from gymnasium import spaces

from .utils import get_device


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ReplayBuffer(ABC):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as an infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        handle_timeout_termination: bool = True,
    ):

        super().__init__()

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape
        self.action_dim = int(np.prod(action_space.shape))
        self.device = get_device(device)

        self.pos = 0
        self.full = False

        # Adjust buffer size
        self.buffer_size = max(buffer_size, 1)

        # Check that the replay buffer can fit into the memory
        mem_available = None
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros(
            (
                self.buffer_size,
                1,  # for a single env
                *self.obs_shape
            ),
            dtype=observation_space.dtype
        )

        self.next_observations = np.zeros(
            (
                self.buffer_size,
                1,  # for a single env
                *self.obs_shape
            ),
            dtype=observation_space.dtype
        )

        self.actions = np.zeros(
            (self.buffer_size, 1, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        # Handle timeouts' termination properly if needed.
        # See https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, 1), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            total_memory_usage += self.next_observations.nbytes

            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            assert total_memory_usage < mem_available, (
                "This system does not have apparently enough memory to store the complete "
                f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (might be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.float64,
            fixed_done: float,
            done: Union[bool, np.ndarray],
            truncated: bool,
            info: Dict[str, Any],
    ) -> None:

        ## Reshape as necessary
        # actions
        if len(action.shape) == 1:
            action = action.reshape((1, self.action_dim))
        else:
            assert action.shape == (1, self.action_dim), f"Invalid action shape: {action.shape}"
        # obs
        if len(obs.shape) in [1, 3]:  # 3 for images
            obs = obs.reshape((1, *self.obs_shape))
        else:
            assert obs.shape == (1, *self.obs_shape), f"Invalid obs shape: {obs.shape}"
        # next obs
        if len(next_obs.shape) in [1, 3]:  # 3 for images
            next_obs = next_obs.reshape((1, *self.obs_shape))
        else:
            assert next_obs.shape == (1, *self.obs_shape), f"Invalid next_obs shape: {next_obs.shape}"
        # reward
        if len(reward.shape) == 0:
            reward = np.array([reward])
        else:
            assert reward.shape == (1,), f"Invalid reward shape: {reward.shape}"
        # done
        if isinstance(done, bool):
            done = np.array([done])
        else:
            assert done.shape == (1,), f"Invalid done shape: {done.shape}"

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            # fixed_done is 0. only when the time horizon is reached
            # truncated is True when the time horizon is reached
            assert (fixed_done == 0. and done.item() is True) == truncated, \
                f"'not fixed_done' and 'truncated' are not equal: 'fixed_done': {fixed_done}, 'truncated': {truncated}"
            self.timeouts[self.pos] = np.array([
                any([
                    truncated,
                    info.get("TimeLimit.truncated", False)
                ])
            ])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of elements to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=1, size=(len(batch_inds),))

        next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts are initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
