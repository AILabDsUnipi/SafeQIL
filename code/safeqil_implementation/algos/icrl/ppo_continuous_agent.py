from typing import Any, Dict, Optional
import os
import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from safeqil_implementation.algos.icrl.buffer import RolloutBuffer
from safeqil_implementation.algos.icrl.networks_continuous import DualVariable, PPONetworks


class ContinuousPPOAgent(object):
    """
    Proximal Policy Optimization algorithm (PPO) used both with ICRL
    for cost optimization.

    Paper: https://arxiv.org/abs/1707.06347
    Code based on: https://github.com/shehryar-malik/icrl/blob/master/stable_baselines3/ppo_lag/ppo_lag.py
                   https://github.com/openai/spinningup/
                   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
                   https://github.com/hill-a/stable-baselines
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            device: th.device,
            config: Dict[str, Any],
            only_test: bool = False,
    ):

        # Get parameters
        self.only_test = only_test
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.first_layer_units = config['ICRL']['layer1_size']
        self.second_layer_units = config['ICRL']['layer2_size']
        self.lr = None
        self.lambda_initial_value: float = 1.0
        self.alpha_cost = None
        self.lambda_lr = None
        self.w_constraint_optimization: bool = config['ICRL']['w_constraint_optimization']

        if self.only_test is False:
            self.lr: float = config['ICRL']['lr']
            self.n_steps: int = config['ICRL']['n_steps'] + 500  # add a constant to ensure the free space
            self.reward_gamma: float = config['ICRL']['reward_gamma']
            self.reward_gae_lambda: float = config['ICRL']['reward_gae_lambda']
            self.reward_vf_coef: float = config['ICRL']['reward_vf_coef']
            self.max_grad_norm: float = config['ICRL']['max_grad_norm']
            self.batch_size: Optional[int] = config['ICRL']['batch_size']
            self.n_epochs: int = config['ICRL']['epochs']
            self.clip_range: float = config['ICRL']['clip_range']
            self.target_kl: float = config['ICRL']['target_kl']
            self.rollout_buffer = None
            self.cost_gamma: float = config['ICRL']['cost_gamma']
            self.cost_gae_lambda: float = config['ICRL']['cost_gae_lambda']
            self.cost_vf_coef: float = config['ICRL']['cost_vf_coef']
            self.lambda_initial_value: float = config['ICRL']['lambda_initial_value']
            self.lambda_lr: float = config['ICRL']['lambda_lr']
            self.alpha_cost: float = config['ICRL']['alpha_cost']

        self._setup_model()

    def _setup_model(self) -> None:

        self.policy = PPONetworks(
            self.observation_space,
            self.action_space,
            self.lr,
            self.first_layer_units,
            self.second_layer_units,
            self.device,
            self.only_test
        )
        self.policy = self.policy.to(self.device)

        self.dual = DualVariable(
            self.alpha_cost,
            self.lambda_lr,
            self.lambda_initial_value,
            self.device,
            self.only_test
        )

        if self.only_test is False:
            self.rollout_buffer = RolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                reward_gamma=self.reward_gamma,
                reward_gae_lambda=self.reward_gae_lambda,
                cost_gamma=self.cost_gamma,
                cost_gae_lambda=self.cost_gae_lambda
            )

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """

        all_entropy_losses, all_kl_divs = [], []
        all_policy_losses, all_reward_value_losses = [], []
        all_total_losses = []
        all_clip_fractions = []
        all_total_policy_losses = []
        if self.w_constraint_optimization is True:
            all_cost_advantages_ratio_terms = []
            all_cost_losses = []
            all_cost_value_losses = []

        # Perform 'n_epochs' gradient steps
        early_stop_epoch = self.n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            entropy_losses = []
            policy_losses = []
            reward_value_losses = []
            total_losses = []
            clip_fractions = []
            total_policy_losses = []
            if self.w_constraint_optimization is True:
                cost_advantages_ratio_terms = []
                cost_losses = []
                cost_value_losses = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions
                )

                # Normalize reward advantages
                reward_advantages = rollout_data.reward_advantages - rollout_data.reward_advantages.mean()
                reward_advantages /= (rollout_data.reward_advantages.std() + 1e-8)
                if self.w_constraint_optimization is True:
                    # Center but NOT rescale cost advantages
                    cost_advantages = rollout_data.cost_advantages - rollout_data.cost_advantages.mean()

                # The ratio between the old and the new policy should be 1 at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                assert len(reward_advantages.shape) == 1 and reward_advantages.shape[0] == self.batch_size
                assert len(ratio.shape) == 1 and ratio.shape[0] == self.batch_size
                policy_loss_1 = reward_advantages * ratio
                policy_loss_2 = reward_advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                policy_loss_for_logs = policy_loss.item()
                total_policy_loss = policy_loss

                # Add cost to loss
                if self.w_constraint_optimization is True:
                    assert len(cost_advantages.shape) == 1 and cost_advantages.shape[0] == self.batch_size
                    current_lambda = self.dual.nu().item()
                    cost_advantages_ratio_term = th.mean(cost_advantages * ratio)
                    cost_loss = current_lambda * cost_advantages_ratio_term
                    total_policy_loss += cost_loss
                    total_policy_loss /= (1 + current_lambda)

                # Policy Logs
                policy_losses.append(policy_loss_for_logs)
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                total_policy_losses.append(total_policy_loss.item())
                if self.w_constraint_optimization is True:
                    cost_advantages_ratio_terms.append(cost_advantages_ratio_term.item())
                    cost_losses.append(cost_loss.item())

                ## Value loss using the TD(gae_lambda) target
                # Rewards
                assert (
                        len(reward_values.shape) == 2 and
                        reward_values.shape[0] == self.batch_size and
                        reward_values.shape[1] == 1
                )
                assert (
                        len(rollout_data.reward_returns.size()) == 1 and
                        rollout_data.reward_returns.size(0) == self.batch_size
                )
                reward_values = reward_values.squeeze(1)
                reward_value_loss = F.mse_loss(rollout_data.reward_returns, reward_values)
                reward_value_losses.append(reward_value_loss.item())
                # Costs
                if self.w_constraint_optimization is True:
                    assert (
                            len(cost_values.shape) == 2 and
                            cost_values.shape[0] == self.batch_size and
                            cost_values.shape[1] == 1
                    )
                    assert len(rollout_data.cost_returns.size()) == 1 and \
                           rollout_data.cost_returns.size(0) == self.batch_size
                    cost_values = cost_values.squeeze(1)
                    cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                    cost_value_losses.append(cost_value_loss.item())

                # Entropy loss favors exploration
                assert entropy is not None
                entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                        total_policy_loss +
                        (self.reward_vf_coef * reward_value_loss)
                )
                if self.w_constraint_optimization is True:
                    loss += self.cost_vf_coef * cost_value_loss

                # Compute gradients
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # Perform optimizer step
                self.policy.optimizer.step()

                # Store policy loss and KL-Divergence
                total_losses.append(loss.item())
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            # Store epoch mean of metrics
            all_kl_divs.append(np.mean(approx_kl_divs))
            all_entropy_losses.append(np.mean(entropy_losses))
            all_policy_losses.append(np.mean(policy_losses))
            all_reward_value_losses.append(np.mean(reward_value_losses))
            all_total_losses.append(total_losses)
            all_clip_fractions.append(clip_fractions)
            all_total_policy_losses.append(total_policy_losses)
            if self.w_constraint_optimization is True:
                all_cost_advantages_ratio_terms.append(cost_advantages_ratio_terms)
                all_cost_losses.append(cost_losses)
                all_cost_value_losses.append(cost_value_losses)

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                early_stop_epoch = epoch
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        ## End of policy train
        # Update dual variable using original (unnormalized) cost
        if self.w_constraint_optimization is True:
            average_cost = np.mean(self.rollout_buffer.costs[:self.rollout_buffer.pos])
            all_cost_per_step = self.rollout_buffer.costs[:self.rollout_buffer.pos].tolist()  # for logs
            self.dual.update_parameter(torch.from_numpy(np.array([average_cost])).to(self.device))

        # Extra logs
        mean_reward_advantages = np.mean(self.rollout_buffer.reward_advantages[:self.rollout_buffer.pos])
        assert (np.isnan(self.rollout_buffer.reward_returns[self.rollout_buffer.pos:])).all()
        assert (np.isnan(self.rollout_buffer.reward_values[self.rollout_buffer.pos:])).all()
        explained_reward_var = self.explained_variance(
            self.rollout_buffer.reward_returns[:self.rollout_buffer.pos],
            self.rollout_buffer.reward_values[:self.rollout_buffer.pos]
        )

        # Extra cost logs
        if self.w_constraint_optimization is True:
            mean_cost_advantages = np.mean(self.rollout_buffer.cost_advantages[:self.rollout_buffer.pos])
            assert (np.isnan(self.rollout_buffer.cost_returns[self.rollout_buffer.pos:])).all()
            assert (np.isnan(self.rollout_buffer.cost_values[self.rollout_buffer.pos:])).all()
            explained_cost_var = self.explained_variance(
                self.rollout_buffer.cost_returns[:self.rollout_buffer.pos],
                self.rollout_buffer.cost_values[:self.rollout_buffer.pos]
            )
            dual_nu = self.dual.nu().item()
            dual_loss = self.dual.loss.item()

        # Define the returns
        training_returns = {
            "ppo_total_loss": np.mean(all_total_losses).item(),
            "ppo_policy_loss": np.mean(all_policy_losses).item(),
            "ppo_reward_value_loss": np.mean(all_reward_value_losses).item(),
            "ppo_kl_div": np.mean(all_kl_divs).item(),
            "ppo_entropy_loss": np.mean(all_entropy_losses).item(),
            "ppo_clip_fraction": np.mean(all_clip_fractions).item(),
            "ppo_reward_advantage": mean_reward_advantages.item(),
            "ppo_explained_reward_var": explained_reward_var.item(),
            "ppo_early_stop_epoch": early_stop_epoch,
            "ppo_total_policy_loss": np.mean(all_total_policy_losses).item(),
        }
        if self.w_constraint_optimization is True:
            training_returns["ppo_dual_nu"] = dual_nu
            training_returns["ppo_dual_loss"] = dual_loss
            training_returns["ppo_cost_value_loss"] = np.mean(all_cost_value_losses).item()
            training_returns["ppo_cost_advantage"] = mean_cost_advantages.item()
            training_returns["ppo_explained_cost_var"] = explained_cost_var.item()
            training_returns["mean_ppo_cost_per_step"] = np.mean(all_cost_per_step).item()
            training_returns["min_ppo_cost_per_step"] = np.min(all_cost_per_step).item()
            training_returns["max_ppo_cost_per_step"] = np.max(all_cost_per_step).item()
            training_returns["ppo_cost_advantage_ratio_term"] = np.mean(all_cost_advantages_ratio_terms).item()
            training_returns["ppo_cost_loss"] = np.mean(all_cost_losses).item()
        return training_returns

    @staticmethod
    def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Code from stable baselines.
        Computes the fraction of variance that y_pred explains about y.
        Returns 1 - Var[y-y_pred] / Var[y]

        interpretation:
            ev=0 ⇒ might as well have predicted zero
            ev=1 ⇒ perfect prediction
            ev<0 ⇒ worse than just predicting zero

        :param y_pred: The prediction.
        :param y_true: The expected value.
        :return: explained variance of y_pred and y.
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1

        var_y = np.var(y_true)
        eps = 1e-05  # To avoid division by zero
        return 1 - np.var(y_true - y_pred) / (var_y + eps)

    def save_models(self, prefix_model_name, path):

        path_to_save_ppo_nets = os.path.join(path, f'{prefix_model_name}_ppo_nets.pt')
        print(
            'Saving {} model to {} ...'.format(
                'ppo_nets',
                path_to_save_ppo_nets
            )
        )
        torch.save(self.policy.state_dict(), path_to_save_ppo_nets)

        if self.w_constraint_optimization is True:
            path_to_save_ppo_dualvar = os.path.join(path, f'{prefix_model_name}_ppo_dualVar.pt')
            print(
                'Saving {} to {} ...'.format(
                    'ppo_dualVar',
                    path_to_save_ppo_dualvar
                )
            )
            torch.save(self.dual.nu.state_dict(), path_to_save_ppo_dualvar)

    def load_models(self, prefix_model_name, path):

        # Load "ppo_nets"
        path_to_load_ppo_nets = os.path.join(path, f'{prefix_model_name}_ppo_nets.pt')
        print(
            'Loading {} models from {} ...'.format(
                'ppo_nets',
                path_to_load_ppo_nets
            )
        )
        self.policy.load_state_dict(
            torch.load(
                path_to_load_ppo_nets,
                map_location=self.device,
                weights_only=False
            )
        )

        # Load "ppo_dualVar"
        if self.w_constraint_optimization is True:
            path_to_load_ppo_dualvar = os.path.join(path, f'{prefix_model_name}_ppo_dualVar.pt')
            print(
                'Loading {} from {} ...'.format(
                    'ppo_dualVar',
                    path_to_load_ppo_dualvar
                )
            )
            self.dual.nu.load_state_dict(
                torch.load(
                    path_to_load_ppo_dualvar,
                    map_location=self.device,
                    weights_only=False
                )
            )
