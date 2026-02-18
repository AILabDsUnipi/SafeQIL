from typing import Optional, Tuple, Type, Dict, Any, Union
from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym

from safeqil_implementation.algos.icrl.utils import get_flattened_obs_dim, is_image_space, preprocess_obs
from safeqil_implementation.algos.icrl.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    Distribution,
    make_proba_distribution
)


class Nu(nn.Module):
    """
    Class for Lagrangian multiplier.

    :param lambda_init: The value with which to initialize the Lagrange multiplier with
    """

    def __init__(self, lambda_init: float = 1.):

        super(Nu, self).__init__()

        # When lambda_init=0.1, then λ=0.1 (approximately)
        self.lambda_init = lambda_init
        lambda_init = np.log(max(np.exp(lambda_init)-1, 1e-8))
        self.log_nu = nn.Parameter(lambda_init*th.ones(1))
        self.clamp_at = lambda_init  # lower bound

    def forward(self):
        return F.softplus(self.log_nu)

    def clamp(self):
        self.log_nu.data.clamp_(min=np.log(max(np.exp(self.clamp_at)-1, 1e-8)))


class DualVariable(object):
    """
    Class for handling the Lagrangian multiplier.

    :param alpha: The budget size
    :param learning_rate: Learning rate for the Lagrange multiplier.
    :param lambda_init: The value with which to initialize the Lagrange multiplier with
    """

    def __init__(
            self,
            alpha: float = 0,
            learning_rate: float = 0.05,
            lambda_init: float = 0.1,
            device: th.device = th.device('cpu'),
            only_test: bool = False
    ):

        self.device = device
        self.alpha = alpha
        self.loss = None

        self.nu = Nu(lambda_init)
        self.nu = self.nu.to(self.device)

        if only_test is False:
            self.optimizer = optim.Adam(self.nu.parameters(), lr=learning_rate)

    def update_parameter(self, cost: th.Tensor):
        # Compute loss
        self.loss = - self.nu() * (cost-self.alpha)

        # Update
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Clamp
        self.nu.clamp()


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features' extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flattens the input.
    Used as a placeholder when feature extraction is unnecessary.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space} "
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MLPBase(nn.Module):
    def __init__(
            self,
            observation_space: spaces.Space,
            first_layer_units: int,
            second_layer_units: int,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(MLPBase, self).__init__()

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.first_layer_units = first_layer_units
        self.second_layer_units = second_layer_units
        self.features_extractor = features_extractor_class(self.observation_space, **features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        # Actor
        self.actor_latent = nn.Sequential(
            nn.Linear(self.features_dim, self.first_layer_units),
            nn.Tanh(),
            nn.Linear(self.first_layer_units, self.second_layer_units),
            nn.Tanh()
        )

        ## Critic
        # For Rewards
        self.value_latent = nn.Sequential(
            nn.Linear(self.features_dim, self.first_layer_units),
            nn.Tanh(),
            nn.Linear(self.first_layer_units, self.second_layer_units),
            nn.Tanh()
        )
        # For Costs
        self.cost_value_latent = nn.Sequential(
            nn.Linear(self.features_dim, self.first_layer_units),
            nn.Tanh(),
            nn.Linear(self.first_layer_units, self.second_layer_units),
            nn.Tanh()
        )

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No feature extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=False)
        return self.features_extractor(preprocessed_obs)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)

        actor_latent = self.actor_latent(features)
        value_latent = self.value_latent(features)
        cost_value_latent = self.cost_value_latent(features)

        return actor_latent, value_latent, cost_value_latent


class PPONetworks(nn.Module):
    """
    Implements the actor critic algorithm.
    Two value networks (for reward and cost) are used.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr: float,
            first_layer_units: int,
            second_layer_units: int,
            device: th.device,
            only_test: bool,
            log_std_init: float = 0.0,
    ):

        super(PPONetworks, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.first_layer_units = first_layer_units
        self.second_layer_units = second_layer_units
        self.only_test = only_test
        self.device = device
        self.log_std_init = log_std_init

        # Define 'n_actions'
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, "Error: the action space must be a vector"
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            self.n_actions = 1
        else:
            raise NotImplementedError(
                "Error: constraint net, not implemented for action space"
                f"of type {type(action_space)}."
                " Must be of type Gym Spaces: Box, Discrete."
            )

        self._build()

    def _build(self) -> None:
        """
        Create the networks and the optimizer.
        """

        # Latent feature extractor
        self.mlp_base = MLPBase(self.observation_space, self.first_layer_units, self.second_layer_units)

        ## Actor
        # Action distribution
        self.action_dist = make_proba_distribution(self.action_space)
        # Action network
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.second_layer_units, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.second_layer_units)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.second_layer_units)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.second_layer_units)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        ## Critic
        # For reward
        self.value_net = nn.Linear(self.second_layer_units, 1)
        # For cost
        self.cost_value_net = nn.Linear(self.second_layer_units, 1)

        if self.only_test is False:
            # Init weights: use orthogonal initialization
            # with small initial weight for the output
            # Values from stable-baselines.
            module_gains = {
                self.mlp_base: np.sqrt(2),  # Includes features' extractor
                self.action_net: 0.01,
                self.value_net: 1,
                self.cost_value_net: 1
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

            # Setup optimizer with initial learning rate
            self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr, eps=1e-05)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def maybe_convert_obs_from_numpy_to_torch(self, obs: Union[np.ndarray, th.Tensor]) -> Tuple[th.Tensor, bool, bool]:

        # Check batch dimension
        has_batch_dim = True
        if len(obs.shape) == len(self.observation_space.shape):
            # Add batch dimension
            obs = obs[None, ...]
            has_batch_dim = False
        else:
            assert len(obs.shape) == len(self.observation_space.shape) + 1

        # Check the type
        is_numpy = False
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).to(self.device)
            is_numpy = True
        else:
            assert isinstance(obs, th.Tensor)

        return obs, has_batch_dim, is_numpy

    def maybe_convert_acts_from_numpy_to_torch(self, acts: Union[np.ndarray, th.Tensor]) -> Tuple[th.Tensor, bool, bool]:

        # Check batch dimension
        has_batch_dim = True
        if isinstance(self.action_space, spaces.Box):
            if len(acts.shape) == 1 and acts.shape[0] == self.n_actions:
                # Add batch dimension
                acts = acts[None, ...]
                has_batch_dim = False
            else:
                assert len(acts.shape) == 2 and acts.shape[1] == self.n_actions
        elif isinstance(self.action_space, spaces.Discrete):
            if len(acts.shape) == 1:
                # Add batch dimension
                acts = acts[None, ...]
                has_batch_dim = False
            else:
                assert len(acts.shape) == 2 and acts.shape[1] == self.n_actions
        else:
            raise NotImplementedError(
                "Error: networks continuous, not implemented for action space"
                f"of type {type(self.action_space)}."
                " Must be of type Gym Spaces: Box, Discrete."
            )

        # Check the type
        is_numpy = False
        if isinstance(acts, np.ndarray):
            acts = th.from_numpy(acts).to(self.device)
            is_numpy = True
        else:
            assert isinstance(acts, th.Tensor)

        return acts, has_batch_dim, is_numpy

    def forward(
            self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False
    ) -> Tuple[
        th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor
    ]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        obs, has_batch_dim, is_numpy = self.maybe_convert_obs_from_numpy_to_torch(obs)
        latent_pi, latent_vf, latent_cvf = self.mlp_base(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cvf)

        if has_batch_dim is False:
            # Remove batch dimension
            actions = actions[0]
            values = values[0]
            cost_values = cost_values[0]
            log_prob = log_prob[0]

        if is_numpy:
            # Convert to numpy
            actions = actions.detach().cpu().numpy()
            values = values.detach().cpu().numpy()
            cost_values = cost_values.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()

        return actions, values, cost_values, log_prob

    def evaluate_actions(
            self, obs: Union[th.Tensor, np.ndarray], actions: Union[th.Tensor, np.ndarray]
    ) -> Tuple[
        Union[th.Tensor, np.ndarray],
        Union[th.Tensor, np.ndarray],
        Union[th.Tensor, np.ndarray],
        Union[th.Tensor, np.ndarray]
    ]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observations.
        :param actions: Actions to be evaluated.
        :return: estimated value, cost value, log likelihood of taking those actions
                 and entropy of the action distribution.
        """
        obs, *_ = self.maybe_convert_obs_from_numpy_to_torch(obs)
        latent_pi, latent_vf, latent_cvf = self.mlp_base(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions, *_ = self.maybe_convert_acts_from_numpy_to_torch(actions)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cvf)

        return values, cost_values, log_prob, distribution.entropy()

    def predict(self, obs: Union[th.Tensor, np.ndarray], deterministic: bool = False) -> Union[th.Tensor, np.ndarray]:
        """
        Get the action according to the policy for a given observation.

        :param obs: Observations
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

        obs, has_batch_dim, is_numpy = self.maybe_convert_obs_from_numpy_to_torch(obs)
        latent_pi, _, _ = self.mlp_base(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        if has_batch_dim is False:
            # Remove batch dimension
            actions = actions[0]

        if is_numpy:
            # Convert to numpy
            actions = actions.detach().cpu().numpy()

        return actions
