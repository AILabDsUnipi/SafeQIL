from typing import Dict, Type, Optional, Any, Union
import copy
from tqdm import tqdm

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces

from SCOPIL.algos.icrl.networks_continuous import BaseFeaturesExtractor, FlattenExtractor
from SCOPIL.algos.icrl.constraint_net import ConstraintNet
from SCOPIL.algos.vicrl.utils import dirichlet_kl_divergence_loss


class VariationalConstraintNet(ConstraintNet):
    """
    Code based on:
    https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/constraint_models/constraint_net/variational_constraint_net.py
    """
    def __init__(
            self,
            config: Dict,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            device: th.device = th.device("cpu"),
            only_test: bool = False,
    ):

        self.dir_prior = config['VICRL']['constraint_net_di_prior']
        self.mode = config['VICRL']['constraint_net_mode']

        # Create an 'ICRL' field and copy the 'VICRL' since the superclass (ConstraintNet)
        # gets the hyperparameters from 'ICRL' field
        config['ICRL'] = copy.deepcopy(config['VICRL'])

        super(VariationalConstraintNet, self).__init__(
            config,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            device,
            only_test
        )

    def _build(self) -> None:

        # Create network and add softplus at the end
        if isinstance(self.n_hidden_units_layer2, int):
            self.network = nn.Sequential(
                nn.Linear(self.input_dims, self.n_hidden_units_layer1),
                nn.ReLU(),
                nn.Linear(self.n_hidden_units_layer1, self.n_hidden_units_layer2),
                nn.ReLU(),
                nn.Linear(self.n_hidden_units_layer2, 2),
                nn.Softplus()
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(self.input_dims, self.n_hidden_units_layer1),
                nn.ReLU(),
                nn.Linear(self.n_hidden_units_layer1, 2),
                nn.Softplus()
            )
        self.network.to(self.device)

        if self.only_test is False:
            # Build optimizer
            self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr_schedule(1), eps=1e-05)

    def forward(self, obs: Union[th.Tensor, np.ndarray], acts: Union[th.Tensor, np.ndarray]) -> th.tensor:
        alpha_beta = super().forward(obs, acts)

        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        pred = th.distributions.Beta(alpha, beta).rsample().unsqueeze(-1)

        return pred

    def cost_function(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:

        assert self.mode == 'sample', f"'mode' other than 'sample' is not supported! mode: {self.mode}"

        return super().cost_function(obs, acts)

    def train_network(
            self,
            iterations: int,
            nominal_obs: np.ndarray,
            nominal_acts: np.ndarray,
            episode_lengths: np.ndarray,
            current_progress_remaining: float = 1
    ):

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Save current network predictions since we use importance sampling
        with th.no_grad():
            start_preds = self.forward(nominal_obs, nominal_acts).detach()

        # Variables initialization
        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        expert_loss = th.tensor(np.inf)
        nominal_loss = th.tensor(np.inf)
        regularizer_loss = th.tensor(np.inf)
        is_weights = th.tensor(0.0)
        nominal_preds = th.tensor(np.inf)
        expert_preds = th.tensor(np.inf)
        kl_old_new = th.tensor(0.0)
        kl_new_old = th.tensor(0.0)

        # Train loop
        for itr in tqdm(range(iterations), desc='Constraint-net training iteration: '):

            # Break if kl is very large
            if self.constraint_net_importance_sampling is True:
                # Compute IS weights
                with th.no_grad():
                    current_preds = self.forward(nominal_obs, nominal_acts).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(
                    start_preds.clone(),
                    current_preds.clone(),
                    episode_lengths
                )
                if (
                        (self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)
                ):
                    early_stop_itr = itr
                    break

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_obs.shape[0], self.expert_obs.shape[0]):

                # Get batch data
                nominal_obs_batch = nominal_obs[nom_batch_indices]
                nominal_acts_batch = nominal_acts[nom_batch_indices]
                expert_obs_batch = self.expert_obs[exp_batch_indices]
                expert_acts_batch = self.expert_acts[exp_batch_indices]
                if self.constraint_net_importance_sampling is False:
                    is_batch = th.ones(nominal_obs_batch.shape[0], dtype=th.float32, device=self.device)
                else:
                    is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_alpha_beta = super().forward(nominal_obs_batch, nominal_acts_batch)
                nominal_alpha = nominal_alpha_beta[:, 0]
                nominal_beta = nominal_alpha_beta[:, 1]
                nominal_preds = th.distributions.Beta(nominal_alpha, nominal_beta).rsample()
                expert_alpha_beta = super().forward(expert_obs_batch, expert_acts_batch)
                expert_alpha = expert_alpha_beta[:, 0]
                expert_beta = expert_alpha_beta[:, 1]
                expert_preds = th.distributions.Beta(expert_alpha, expert_beta).rsample()

                ## Calculate loss
                # Expert loss
                expert_preds = th.clip(expert_preds, min=self.eps, max=1)
                expert_loss = th.mean(th.log(expert_preds))
                # Nominal loss
                nominal_preds = th.clip(nominal_preds, min=self.eps, max=1)
                nominal_loss = th.mean(is_batch * th.log(nominal_preds))
                # Regularizer loss
                nominal_batch_size = nominal_preds.shape[0]
                expert_batch_size = expert_preds.shape[0]
                regularizer_loss = self.regularizer_coeff * (
                        self.kl_regularizer_loss(
                            batch_size=nominal_batch_size,
                            alpha=nominal_alpha,
                            beta=nominal_beta,
                        ) +
                        self.kl_regularizer_loss(
                            batch_size=expert_batch_size,
                            alpha=expert_alpha,
                            beta=expert_beta,
                        )
                )

                # Total loss
                loss = (-expert_loss + nominal_loss) + regularizer_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        training_returns = {
            "constraint_net_total_loss": loss.item(),
            "constraint_net_expert_loss": expert_loss.item(),
            "constraint_net_nominal_loss": nominal_loss.item(),
            "constraint_net_nominal_preds": th.mean(th.log(nominal_preds + self.eps)).item(),
            "constraint_net_regularizer_loss": regularizer_loss.item(),
            "mean_constraint_net_is_weight": th.mean(is_weights).detach().item(),
            "min_constraint_net_is_weight": th.min(is_weights).detach().item(),
            "max_constraint_net_is_weight": th.max(is_weights).detach().item(),
            "constraint_net_log_nominal_preds": th.mean(th.log(nominal_preds + self.eps)).item(),
            "mean_constraint_net_nominal_preds": th.mean(nominal_preds).item(),
            "min_constraint_net_nominal_preds": th.min(nominal_preds).item(),
            "max_constraint_net_nominal_preds": th.max(nominal_preds).item(),
            "constraint_net_log_expert_preds": th.mean(th.log(expert_preds + self.eps)).item(),
            "mean_constraint_net_expert_preds": th.mean(expert_preds).item(),
            "min_constraint_net_expert_preds": th.min(expert_preds).item(),
            "max_constraint_net_expert_preds": th.max(expert_preds).item(),
            "constraint_net_kl_old_new": kl_old_new.item(),
            "constraint_net_kl_new_old": kl_new_old.item(),
            "constraint_net_early_stop_iterations": early_stop_itr
        }
        return training_returns

    def kl_regularizer_loss(self, batch_size: int, alpha: th.Tensor, beta: th.Tensor) -> th.tensor:
        prior = th.from_numpy(np.array(batch_size * [self.dir_prior], dtype=np.float32)).to(self.device)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=th.stack([alpha, beta], dim=1),
            prior=prior
        ).mean()

        return analytical_kld_loss
