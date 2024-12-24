from statistics import mean
from typing import Dict

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from gymnasium import spaces

from .buffer import ReplayBuffer


class Discriminator(nn.Module):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            w_actions: bool,
            lr: float,
            batch_size: int,
            gradient_steps: int,
            hidden_dim_layer1: int,
            hidden_dim_layer2: int,
            rollout_buffer: ReplayBuffer,
            expert_dataloader: DataLoader,
            w_icrl_regularization: bool,
            icrl_regularization_coef: float,
            w_dac_regularization: bool,
            dac_regularization_coef: float,
            activation_fn: th.nn.Module = nn.ReLU,
            device: th.device = th.device("cpu"),
    ):
        super(Discriminator, self).__init__()

        assert isinstance(observation_space, spaces.Box), f'observation_space: {observation_space}'
        assert isinstance(action_space, spaces.Box), f'action_space: {action_space}'

        self.w_actions = w_actions
        self.device = device

        # Model specifications
        self.hidden_dim_layer1 = hidden_dim_layer1
        self.hidden_dim_layer2 = hidden_dim_layer2
        self.activation_fn = activation_fn
        self.input_dim = observation_space.shape[0]
        if self.w_actions is True:
            self.input_dim += action_space.shape[0]

        # Training specification
        self.rollout_buffer = rollout_buffer
        self.expert_dataloader = expert_dataloader
        self.lr = lr
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.eps = 1e-6
        self.w_icrl_regularization = w_icrl_regularization
        self.icrl_regularization_coef = icrl_regularization_coef
        self.w_dac_regularization = w_dac_regularization
        self.dac_regularization_coef = dac_regularization_coef

        self._build_model()

    def _build_model(self):

        # NN architecture
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim_layer1),
            self.activation_fn(),
            nn.Linear(self.hidden_dim_layer1, self.hidden_dim_layer2),
            self.activation_fn(),
            nn.Linear(self.hidden_dim_layer2, 1),
        )
        self.network = self.network.to(self.device)

        # Optimizer
        self.optimizer = th.optim.Adam(
            list(self.network.parameters()),
            lr=self.lr
        )

    def forward(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:

        x = states.to(th.float32)
        if self.w_actions is True:
            x = th.cat([x, actions.to(th.float32)], dim=1)

        x = self.network(x)
        return x

    @th.no_grad()
    def predict(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return th.sigmoid(self.forward(states, actions))

    def update(self) -> Dict[str, float]:

        # Initialize log lists
        log = {
            "discriminator_total_loss": [],
            "discriminator_rollout_loss": [],
            "discriminator_expert_loss": []
        }
        if self.w_icrl_regularization is True:
            log.update({"discriminator_icrl_reg_loss": []})
        if self.w_dac_regularization is True:
            log.update({"discriminator_dac_reg_loss": []})

        for itr in range(self.gradient_steps):

            ## Get samples
            # From replay buffer (rollouts)
            rollout_data = self.rollout_buffer.sample(self.batch_size)
            rollout_observations = rollout_data.observations
            rollout_actions = rollout_data.actions
            # From expert data
            expert_actions, expert_observations = next(iter(self.expert_dataloader))

            # Discriminator loss
            rollout_preds = th.sigmoid(self.forward(rollout_observations, rollout_actions))
            rollout_loss = -th.log(1 - rollout_preds + self.eps)
            expert_preds = th.sigmoid(self.forward(expert_observations, expert_actions))
            expert_loss = -th.log(expert_preds + self.eps)
            loss = th.mean(rollout_loss + expert_loss)
            if self.w_icrl_regularization is True:
                # ICRL regularizer loss is R(θ) = -δ Σ|1-ζ_θ(τ)|
                # regularizer_loss = self.regularizer_coeff * (th.mean(1 - expert_preds) + th.mean(1 - nominal_preds))
                icrl_reg_term = (th.mean(1 - expert_preds) + th.mean(1 - rollout_preds))
                loss += self.icrl_regularization_coef * icrl_reg_term
            if self.w_dac_regularization is True:
                # DAC regularization is the "Gradient Penalty" technique
                dac_reg_term = self.gradient_penalty(
                    rollout_observations, rollout_actions, expert_observations, expert_actions
                )
                loss += self.dac_regularization_coef * dac_reg_term

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log
            log["discriminator_total_loss"].append(loss.item())
            log["discriminator_rollout_loss"].append(th.mean(rollout_loss).item())
            log["discriminator_expert_loss"].append(th.mean(expert_loss).item())
            if self.w_icrl_regularization is True:
                log["discriminator_icrl_reg_loss"].append(icrl_reg_term.item())
            if self.w_dac_regularization is True:
                log["discriminator_dac_reg_loss"].append(dac_reg_term.item())

        # Average the logs
        for key in list(log.keys()):
            log[key] = mean(log[key])

        return log

    def gradient_penalty(self, rollout_observations, rollout_actions, expert_observations, expert_actions):
        """
        Compute the gradient penalty for the current update.
        Code based on: https://github.com/vluzko/dac-iclr-reproducibility/blob/master/dac/adversary.py
        """

        assert rollout_observations.size()[0] == expert_observations.size()[0], \
            (
                f'rollout_observations.size(): {rollout_observations.size()}, '
                f'expert_observations.size(): {expert_observations.size()}'
            )
        batch_size = rollout_observations.size()[0]

        ## Calculate interpolation
        alpha = th.rand(batch_size, 1)
        # For observations
        alpha_obs = alpha.expand_as(rollout_observations)
        alpha_obs = alpha_obs.to(self.device)
        interpolated_obs = alpha_obs * rollout_observations + (1 - alpha_obs) * expert_observations
        interpolated_obs = Variable(interpolated_obs, requires_grad=True)
        interpolated_obs = interpolated_obs.to(self.device).to(th.float32)
        # For actions
        alpha_acts = alpha.expand_as(rollout_actions)
        interpolated_acts = alpha_acts * rollout_actions + (1 - alpha_acts) * expert_actions
        interpolated_acts = Variable(interpolated_acts, requires_grad=True)
        interpolated_acts = interpolated_acts.to(self.device).to(th.float32)

        # Calculate the probability of interpolated examples
        prob_interpolated = self.forward(interpolated_obs, interpolated_acts)

        ## Calculate gradients of probabilities with respect to examples
        # Inputs
        interpolated_inputs = [interpolated_obs]
        if self.w_actions is True:
            interpolated_inputs.append(interpolated_acts)
        # Outputs
        grad_outputs = th.ones(prob_interpolated.size()).to(self.device)
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated_inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )

        # Calculate norm
        norm_gradients = th.cat([grad.view(batch_size, -1).norm(2, dim=1) for grad in gradients])

        # Gradients penalty
        gradients_norm = th.mean((norm_gradients - 1) ** 2)

        # Return gradient penalty
        return gradients_norm

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict()}, path)

    def load_model(self, path: str) -> None:
        """
        Load model from the path.

        :param path:
        :return:
        """

        saved_variables = th.load(path, map_location=self.device, weights_only=False)

        # Load the NN weights
        self.load_state_dict(saved_variables["state_dict"])
