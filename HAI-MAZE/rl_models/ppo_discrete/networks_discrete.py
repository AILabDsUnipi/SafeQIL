from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
        self.clamp_at = lambda_init # lower bound

    def forward(self):
        return F.softplus(self.log_nu)

    def clamp(self):
        self.log_nu.data.clamp_(min=np.log(max(np.exp(self.clamp_at)-1, 1e-8)))

class DualVariable(object):
    """
    Class for handling the Lagrangian multiplier.

    :param alpha: The budget size
    :param learning_rate: Learning rate for the Lagrange multiplier
    :param lambda_init: The value with which to initialize the Lagrange multiplier with
    """

    def __init__(self,
                 alpha: float = 0,
                 learning_rate: float = 0.05,
                 lambda_init: float = 0.1,
                 device: th.device = th.device('cpu'),
                 only_test: bool = False):

        self.device = device
        self.alpha = alpha
        self.loss = None

        self.nu = Nu(lambda_init)
        self.nu = self.nu.to(self.device)

        if only_test is False:
            self.optimizer = optim.Adam(self.nu.parameters(), lr=learning_rate)

    def update_parameter(self, cost: th.Tensor):
        # Compute loss.
        self.loss = - self.nu() * (cost-self.alpha)

        # Update.
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Clamp.
        self.nu.clamp()

class CategoricalDistribution(object):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MLPBase(nn.Module):
    def __init__(self,
                 observation_space: int,
                 first_layer_units: int,
                 second_layer_units: int,
                 icrl: bool):

        super(MLPBase, self).__init__()

        self.observation_space = observation_space
        self.first_layer_units = first_layer_units
        self.second_layer_units = second_layer_units
        self.icrl = icrl

        # Actor
        self.actor_latent = nn.Sequential(nn.Linear(self.observation_space, self.first_layer_units),
                                          nn.Tanh(),
                                          nn.Linear(self.first_layer_units, self.second_layer_units),
                                          nn.Tanh())

        ## Critic
        # For Rewards
        self.value_latent = nn.Sequential(nn.Linear(self.observation_space, self.first_layer_units),
                                          nn.Tanh(),
                                          nn.Linear(self.first_layer_units, self.second_layer_units),
                                          nn.Tanh())
        if self.icrl:
            # For Costs
            self.cost_value_latent = nn.Sequential(nn.Linear(self.observation_space, self.first_layer_units),
                                                   nn.Tanh(),
                                                   nn.Linear(self.first_layer_units, self.second_layer_units),
                                                   nn.Tanh())

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        actor_latent = self.actor_latent(obs)
        value_latent = self.value_latent(obs)
        cost_value_latent = None
        if self.icrl is True:
            cost_value_latent = self.cost_value_latent(obs)

        return actor_latent, value_latent, cost_value_latent


class PPONetworks(nn.Module):
    """Implements the actor critic algorithm.
       When icrl is True two value networks (for reward and cost) are used."""

    def __init__(self,
                 observation_space: int,
                 action_space: int,
                 lr: float,
                 first_layer_units: int,
                 second_layer_units: int,
                 icrl: bool,
                 only_test: bool):

        super(PPONetworks, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.first_layer_units = first_layer_units
        self.second_layer_units = second_layer_units
        self.icrl = icrl
        self.only_test = only_test

        self._build()

    def _build(self) -> None:
        """
        Create the networks and the optimizer.
        """

        # Latent feature extractor
        self.mlp_base = MLPBase(self.observation_space, self.first_layer_units, self.second_layer_units, self.icrl)

        # Actor
        self.action_dist = CategoricalDistribution(self.action_space)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.second_layer_units)

        ## Critic
        # For reward
        self.value_net = nn.Linear(self.second_layer_units, 1)
        if self.icrl is True:
            # For cost
            self.cost_value_net = nn.Linear(self.second_layer_units, 1)

        if self.only_test is False:
            # Init weights: use orthogonal initialization
            # with small initial weight for the output
            # Values from stable-baselines.
            module_gains = {
                self.mlp_base: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1
            }
            if self.icrl is True:
                module_gains[self.cost_value_net] = 1
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

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        latent_pi, latent_vf, latent_cvf = self.mlp_base(obs)

        action_logits = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)

        cost_values = None
        if self.icrl is True:
            cost_values = self.cost_value_net(latent_cvf)

        return actions, values, cost_values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) \
            -> Tuple[th.Tensor, Optional[th.Tensor], th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observations
        :param actions: Actions to be evaluated
        :return: estimated value, cost value, log likelihood of taking those actions
                 and entropy of the action distribution.
        """

        latent_pi, latent_vf, latent_cvf = self.mlp_base(obs)

        action_logits = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)

        cost_values = None
        if self.icrl is True:
            cost_values = self.cost_value_net(latent_cvf)

        return values, cost_values, log_prob, distribution.entropy()

    def predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param obs: Observations
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

        latent_pi, _, _ = self.mlp_base(obs)

        action_logits = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution.get_actions(deterministic=deterministic)
