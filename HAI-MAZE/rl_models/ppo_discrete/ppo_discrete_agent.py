from typing import Any, Callable, Dict, Optional, Tuple
import os
import numpy as np
import torch
import torch as th
from torch import Tensor
from torch.nn import functional as F

from rl_models.ppo_discrete.buffer import RolloutBuffer
from rl_models.ppo_discrete.networks_discrete import DualVariable, PPONetworks


class DiscretePPOAgent(object):
    """
    Proximal Policy Optimization algorithm (PPO) used both with and without lagrangian augmentation
    for cost optimization.

    Paper: https://arxiv.org/abs/1707.06347
    Code based on: https://github.com/shehryar-malik/icrl/blob/master/stable_baselines3/ppo_lag/ppo_lag.py
                   https://github.com/openai/spinningup/
                   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
                   https://github.com/hill-a/stable-baselines)
    """

    def __init__(self,
                 observation_space: int,
                 action_space: int,
                 device: th.device,
                 config: Dict[str, Any],
                 only_test: bool = False,
                 chkpt_dir: str = "PPO",
                 axis_agent: str = "X",
                 expert_obs: Optional[np.ndarray] = None,
                 expert_acts: Optional[np.ndarray] = None):

        # Get parameters
        self.axis_agent = axis_agent
        self.checkpoint_dir = chkpt_dir
        self.only_test = only_test
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.first_layer_units = config['PPO']['layer1_size']
        self.second_layer_units = config['PPO']['layer2_size']
        self.icrl = config['PPO']['ICRL']
        self.lagrangian = config['PPO']['lagrangian']
        self.SL_finetune = False if 'SL_finetuning' not in config['PPO'].keys() else config['PPO']['SL_finetuning']
        self.lr = None
        if self.icrl is True or self.lagrangian is True:
            self.lambda_initial_value = 1.0
            if self.icrl is True:
                self.alpha_cost = None
                self.lambda_lr = None
            elif self.lagrangian is True:
                self.constraint_lambda = None
        if self.only_test is False:
            self.lr = config['PPO']['lr']
            self.n_steps = config['PPO']['n_steps'] + 500  # add a constant to ensure the free space
            self.reward_gamma = config['PPO']['reward_gamma']
            self.reward_gae_lambda = config['PPO']['reward_gae_lambda']
            self.reward_vf_coef = config['PPO']['reward_vf_coef']
            self.max_grad_norm = config['PPO']['max_grad_norm']
            self.batch_size = config['PPO']['batch_size']
            self.n_epochs = config['PPO']['epochs']
            self.clip_range = config['PPO']['clip_range']
            self.target_kl = config['PPO']['target_kl']
            self.SL_finetuning_batch_size = config['PPO']['SL_finetuning_batch_size']
            self.SL_finetuning_lr = config['PPO']['SL_finetuning_lr']
            self.rollout_buffer = None
            if self.icrl is True:
                self.cost_gamma = config['PPO']['ICRL_cost_gamma']
                self.cost_gae_lambda = config['PPO']['ICRL_cost_gae_lambda']
                self.cost_vf_coef = config['PPO']['ICRL_cost_vf_coef']
                self.lambda_initial_value = config['PPO']['ICRL_lambda_initial_value']
                self.lambda_lr = config['PPO']['ICRL_lambda_lr']
                self.alpha_cost = config['PPO']['ICRL_alpha_cost']
            elif self.lagrangian is True or self.SL_finetune is True:
                if self.lagrangian is True:
                    self.lambda_initial_value = config['PPO']['lagrangian_lambda_initial_value']
                    self.lambda_lr = config['PPO']['lagrangian_lambda_lr']
                self.expert_obs = expert_obs
                self.expert_acts = expert_acts

        self._setup_model()

    def _setup_model(self) -> None:

        self.policy = PPONetworks(
            self.observation_space,
            self.action_space,
            self.lr,
            self.first_layer_units,
            self.second_layer_units,
            self.icrl,
            self.only_test
        )
        self.policy = self.policy.to(self.device)

        if self.icrl is True:
            self.dual = DualVariable(
                self.alpha_cost,
                self.lambda_lr,
                self.lambda_initial_value,
                self.device,
                self.only_test
            )
        elif self.lagrangian is True:
            self.constraint_lambda = torch.tensor(
                [self.lambda_initial_value],
                dtype=torch.float32,
                requires_grad=True,
                device=self.device
            )

        if self.only_test is False:
            self.rollout_buffer = RolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                reward_gamma=self.reward_gamma,
                reward_gae_lambda=self.reward_gae_lambda,
                icrl=self.icrl,
                cost_gamma=None if self.icrl is False else self.cost_gamma,
                cost_gae_lambda=None if self.icrl is False else self.cost_gae_lambda
            )
            if self.lagrangian is True:
                # Define the optimizer for Lambda of Lagrangian
                self.constraint_lambda_optimizer = torch.optim.Adam(
                    [self.constraint_lambda],
                    lr=self.lambda_lr,
                    eps=1e-05
                )

    def train_(self):
        """
        Update policy using the currently gathered rollout buffer.
        """

        all_entropy_losses, all_kl_divs = [], []
        all_policy_losses, all_reward_value_losses = [], []
        all_total_losses = []
        all_clip_fractions = []

        if self.icrl is True or self.lagrangian is True:
            all_total_policy_losses = []
            if self.icrl is True:
                # Logs for ICRL
                all_cost_advantages_ratio_terms = []
                all_cost_losses = []
                all_cost_value_losses = []
            elif self.lagrangian is True:
                # Logs for Lagrangian
                all_lagrangian_constraint_policy_term_losses = []
                all_lagrangian_constraint_lambda_losses = []

        # Perform 'n_epochs' gradient steps
        early_stop_epoch = self.n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            entropy_losses = []
            policy_losses = []
            reward_value_losses = []
            total_losses = []
            clip_fractions = []
            if self.icrl is True or self.lagrangian is True:
                total_policy_losses = []
                if self.icrl is True:
                    cost_advantages_ratio_terms = []
                    cost_losses = []
                    cost_value_losses = []
                elif self.lagrangian is True:
                    lagrangian_constraint_policy_term_losses = []
                    lagrangian_constraint_lambda_losses = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # Convert discrete action from float to long
                assert len(rollout_data.actions.size()) == 2 and \
                       rollout_data.actions.size(0) == self.batch_size and \
                       rollout_data.actions.size(1) == 1, \
                    ""
                actions = rollout_data.actions.long().squeeze(1)

                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions
                )

                # Normalize reward advantages
                reward_advantages = rollout_data.reward_advantages - rollout_data.reward_advantages.mean()
                reward_advantages /= (rollout_data.reward_advantages.std() + 1e-8)
                if self.icrl is True:
                    # Center but NOT rescale cost advantages
                    cost_advantages = rollout_data.cost_advantages - rollout_data.cost_advantages.mean()

                # Ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                assert len(reward_advantages.size()) == 1 and reward_advantages.size(0) == self.batch_size, ""
                assert len(ratio.size()) == 1 and ratio.size(0) == self.batch_size, ""
                policy_loss_1 = reward_advantages * ratio
                policy_loss_2 = reward_advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                policy_loss_for_logs = policy_loss.item()
                total_policy_loss = policy_loss

                if self.icrl is True:
                    # Add cost to loss
                    assert len(cost_advantages.size()) == 1 and cost_advantages.size(0) == self.batch_size, ""
                    current_lambda = self.dual.nu().item()
                    cost_advantages_ratio_term = th.mean(cost_advantages * ratio)
                    cost_loss = current_lambda * cost_advantages_ratio_term

                    total_policy_loss += cost_loss
                    total_policy_loss /= (1 + current_lambda)
                elif self.lagrangian is True:
                    # Sample from expert demonstrations
                    sample_expert_obs, sample_expert_acts = self.get()
                    assert len(sample_expert_acts.size()) == 1, ""

                    lagrangian_constraint_policy_loss_term = self.calc_lagrangian_constraint_policy_loss_term(
                        sample_expert_obs,
                        sample_expert_acts
                    )
                    total_policy_loss += self.constraint_lambda.item()*lagrangian_constraint_policy_loss_term

                # Policy Logs
                policy_losses.append(policy_loss_for_logs)
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.icrl is True or self.lagrangian:
                    total_policy_losses.append(total_policy_loss.item())
                    if self.icrl is True:
                        cost_advantages_ratio_terms.append(cost_advantages_ratio_term.item())
                        cost_losses.append(cost_loss.item())
                    elif self.lagrangian is True:
                        lagrangian_constraint_policy_term_losses.append(lagrangian_constraint_policy_loss_term.item())

                ## Value loss using the TD(gae_lambda) target
                # Rewards
                assert len(reward_values.size()) == 2 and \
                       reward_values.size(0) == self.batch_size and \
                       reward_values.size(1) == 1, \
                    ""
                assert len(rollout_data.reward_returns.size()) == 1 and \
                       rollout_data.reward_returns.size(0) == self.batch_size, \
                    ""
                reward_values = reward_values.squeeze(1)
                reward_value_loss = F.mse_loss(rollout_data.reward_returns, reward_values)
                reward_value_losses.append(reward_value_loss.item())
                if self.icrl is True:
                    # Costs
                    assert len(cost_values.size()) == 2 and \
                           cost_values.size(0) == self.batch_size and \
                           cost_values.size(1) == 1, \
                        ""
                    assert len(rollout_data.cost_returns.size()) == 1 and \
                           rollout_data.cost_returns.size(0) == self.batch_size, \
                        ""
                    cost_values = cost_values.squeeze(1)
                    cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                    cost_value_losses.append(cost_value_loss.item())

                # Entropy loss favors exploration
                entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (total_policy_loss
                        + (self.reward_vf_coef * reward_value_loss)
                        + (0 if self.icrl is False else (self.cost_vf_coef * cost_value_loss)))

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

                if self.lagrangian is True:
                    ## Update Lambda
                    # Calculate Lambda loss
                    lagrangian_constraint_lambda_loss = self.calc_lagrangian_constraint_lambda_loss(
                        sample_expert_obs,
                        sample_expert_acts
                    )
                    # Lambda Optimizer step
                    self.constraint_lambda_optimizer.zero_grad()
                    lagrangian_constraint_lambda_loss.backward()
                    self.constraint_lambda_optimizer.step()
                    # Store Lambda loss
                    lagrangian_constraint_lambda_losses.append(lagrangian_constraint_lambda_loss.item())

            # Store epoch mean of metrics
            all_kl_divs.append(np.mean(approx_kl_divs))
            all_entropy_losses.append(np.mean(entropy_losses))
            all_policy_losses.append(np.mean(policy_losses))
            all_reward_value_losses.append(np.mean(reward_value_losses))
            all_total_losses.append(total_losses)
            all_clip_fractions.append(clip_fractions)
            if self.icrl is True or self.lagrangian is True:
                all_total_policy_losses.append(total_policy_losses)
                if self.icrl is True:
                    all_cost_advantages_ratio_terms.append(cost_advantages_ratio_terms)
                    all_cost_losses.append(cost_losses)
                    all_cost_value_losses.append(cost_value_losses)
                elif self.lagrangian is True:
                    all_lagrangian_constraint_policy_term_losses.append(lagrangian_constraint_policy_term_losses)
                    all_lagrangian_constraint_lambda_losses.append(lagrangian_constraint_lambda_losses)

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                early_stop_epoch = epoch
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        ## End of policy train

        if self.icrl is True:
            # Update dual variable using original (unnormalized) cost
            average_cost = np.mean(self.rollout_buffer.costs[:self.rollout_buffer.pos])
            all_cost_per_step = self.rollout_buffer.costs[:self.rollout_buffer.pos].tolist() # for logs
            self.dual.update_parameter(torch.from_numpy(np.array([average_cost])).to(self.device))

        # Extra logs
        mean_reward_advantages = np.mean(self.rollout_buffer.reward_advantages[:self.rollout_buffer.pos])
        assert (np.isnan(self.rollout_buffer.reward_returns[self.rollout_buffer.pos:])).all(), ""
        assert (np.isnan(self.rollout_buffer.reward_values[self.rollout_buffer.pos:])).all(), ""
        explained_reward_var = self.explained_variance(
            self.rollout_buffer.reward_returns[:self.rollout_buffer.pos],
            self.rollout_buffer.reward_values[:self.rollout_buffer.pos]
        )

        if self.icrl is True:
            # Extra cost logs
            mean_cost_advantages = np.mean(self.rollout_buffer.cost_advantages[:self.rollout_buffer.pos])
            assert (np.isnan(self.rollout_buffer.cost_returns[self.rollout_buffer.pos:])).all(), ""
            assert (np.isnan(self.rollout_buffer.cost_values[self.rollout_buffer.pos:])).all(), ""
            explained_cost_var = self.explained_variance(
                self.rollout_buffer.cost_returns[:self.rollout_buffer.pos],
                self.rollout_buffer.cost_values[:self.rollout_buffer.pos]
            )
            dual_nu = self.dual.nu().item()
            dual_loss = self.dual.loss.item()

        # Define the returns
        returns = [
            np.mean(all_total_losses).tolist(),
            np.mean(all_policy_losses).tolist(),
            np.mean(all_reward_value_losses).tolist(),
            np.mean(all_kl_divs).tolist(),
            np.mean(all_entropy_losses).tolist(),
            np.mean(all_clip_fractions).tolist(),
            mean_reward_advantages.tolist(),
            explained_reward_var.tolist(),
            early_stop_epoch
        ]
        if self.icrl is True or self.lagrangian is True:
            returns += [np.mean(all_total_policy_losses).tolist()]
            if self.icrl is True:
                returns += [
                    dual_nu,
                    dual_loss,
                    np.mean(all_cost_value_losses).tolist(),
                    mean_cost_advantages.tolist(),
                    explained_cost_var.tolist(),
                    all_cost_per_step,
                    np.mean(all_cost_advantages_ratio_terms).tolist(),
                    np.mean(all_cost_losses).tolist()
                ]
            elif self.lagrangian is True:
                returns += [
                    self.constraint_lambda.item(),
                    np.mean(all_lagrangian_constraint_lambda_losses).tolist(),
                    np.mean(all_lagrangian_constraint_policy_term_losses).tolist()
                ]

        return returns

    def calc_lagrangian_constraint_lambda_loss(self, expert_obs, expert_acts) -> Tensor:
        lagrangian_constraint_policy_loss_term = self.calc_lagrangian_constraint_policy_loss_term(
            expert_obs,
            expert_acts
        )
        lagrangian_constraint_lambda_loss = - self.constraint_lambda.squeeze(0) * lagrangian_constraint_policy_loss_term.item()

        return lagrangian_constraint_lambda_loss

    def calc_lagrangian_constraint_policy_loss_term(self, expert_obs, expert_acts) -> Tensor:
        # Get action probabilities for demonstrations
        _, _, expert_log_probs, _ = self.policy.evaluate_actions(expert_obs, expert_acts)

        # Compute loss
        negative_log_probs = -expert_log_probs.mean()

        return negative_log_probs

    def get(self, batch_size=None) -> Tuple[Tensor, Tensor]:

        batch_size = batch_size if batch_size is not None else self.batch_size
        indices = np.random.permutation(batch_size)

        assert len(self.expert_acts.shape) == 1, ""
        sample_expert_acts = self.expert_acts[indices[:batch_size]]
        sample_expert_acts_tensor = torch.from_numpy(sample_expert_acts).long().to(self.device)

        sample_expert_obs = self.expert_obs[indices[:batch_size]]
        sample_expert_obs_tensor = torch.from_numpy(sample_expert_obs).to(self.device)

        return sample_expert_obs_tensor, sample_expert_acts_tensor

    def SL_finetuning(self):
        assert not self.SL_finetuning_batch_size < self.expert_acts.shape[0], \
            "Batch size is smaller that the number of samples."
        self.SL_finetuning_batch_size = self.expert_acts.shape[0]

        # Get samples
        expert_states, expert_actions = self.get(self.SL_finetuning_batch_size)
        # Get log probabilities
        _, _, log_prob, _ = self.policy.evaluate_actions(expert_states, expert_actions)
        # Compute negative log-likelihood
        constraint_policy_loss_term = (-log_prob).mean()

        # Compute gradients
        self.policy.optimizer.zero_grad()
        constraint_policy_loss_term.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        # Perform optimizer step
        self.policy.optimizer.step()

        return constraint_policy_loss_term.item()

    # From stable baselines
    @staticmethod
    def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes fraction of variance that y_pred explains about y.
        Returns 1 - Var[y-y_pred] / Var[y]

        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero

        :param y_pred: the prediction
        :param y_true: the expected value
        :return: explained variance of y_pred and y
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1

        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    def save_models(self, override=False):

        new_chkpt_dir = self.checkpoint_dir

        # If all models should be saved, the corresponding directory should be created
        if override is False:
            checkpoint_dir_suffix = 0
            while os.path.exists(os.path.join(self.checkpoint_dir, str(checkpoint_dir_suffix))):
                checkpoint_dir_suffix += 1
            new_chkpt_dir = os.path.join(self.checkpoint_dir, str(checkpoint_dir_suffix))
            os.mkdir(new_chkpt_dir)

        path_to_save_ppo_nets = os.path.join(new_chkpt_dir, self.axis_agent + '_ppo_nets')
        print(
            'Saving {} models to {} ...'.format(
                os.path.basename(path_to_save_ppo_nets),
                path_to_save_ppo_nets
            )
        )
        torch.save(self.policy.state_dict(), path_to_save_ppo_nets)
        if self.icrl is True:
            path_to_save_ppo_dualvar = os.path.join(new_chkpt_dir, self.axis_agent + '_ppo_dualVar')
            print(
                'Saving {} to {} ...'.format(
                    os.path.basename(path_to_save_ppo_dualvar),
                    path_to_save_ppo_dualvar
                )
            )
            torch.save(self.dual.nu.state_dict(), path_to_save_ppo_dualvar)
        elif self.lagrangian is True:
            path_to_save_ppo_lagr_lambda = os.path.join(new_chkpt_dir, self.axis_agent + '_ppo_lagr_lambda')
            print(
                'Saving {} to {} ...'.format(
                    os.path.basename(path_to_save_ppo_lagr_lambda),
                    path_to_save_ppo_lagr_lambda
                )
            )
            torch.save(self.constraint_lambda, path_to_save_ppo_lagr_lambda)

    def load_models(self, load_checkpoint_path_name):

        # Load "ppo_nets"
        print(
            'Loading {} models from {} ...'.format(
                self.axis_agent + '_ppo_nets',
                os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_nets')
            )
        )

        # Load only the desired part of the pretrained model.
        # For instance, when SL finetuning is used but the pretrained model has trained with ICRL,
        # the cost model should not be loaded.
        # Code obtained from: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        original_pretrained_dict = torch.load(
            os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_nets'),
            map_location=self.device
        )
        policy_dict = self.policy.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in original_pretrained_dict.items() if k in policy_dict}
        unnecessary_keys = [k for k in original_pretrained_dict.keys() if k not in policy_dict]
        if len(unnecessary_keys) > 0:
            print(
                "\nThe following parameters are parts of the "
                "pretrained model '{}' but they will not be loaded: \n{}".format(
                    self.axis_agent + '_ppo_nets',
                    unnecessary_keys
                )
            )
        # 2. overwrite entries in the existing state dict
        policy_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.policy.load_state_dict(policy_dict)

        # Load "ppo_dualVar"
        if self.icrl is True:
            print(
                'Loading {} from {} ...'.format(
                    self.axis_agent + '_ppo_dualVar',
                    os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_dualVar')
                )
            )
            self.dual.nu.load_state_dict(
                torch.load(
                    os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_dualVar'),
                    map_location=self.device
                )
            )

        # Load "ppo_lagr_lambda"
        elif self.lagrangian is True:
            print(
                'Loading {} from {} ...'.format(
                    self.axis_agent + '_ppo_lagr_lambda',
                    os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_lagr_lambda')
                )
            )
            self.constraint_lambda = torch.load(
                os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_lagr_lambda'),
                map_location=self.device
            )
