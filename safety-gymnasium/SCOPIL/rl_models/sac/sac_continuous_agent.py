# Code based on: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac

import os
import pathlib
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from abc import ABC
import io
from statistics import mean

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from .buffer import ReplayBuffer
from .discriminator import Discriminator
from .networks_continuous import Actor, CnnPolicy, MlpPolicy, SACPolicy, ContinuousCritic
from .utils import (
    get_device,
    set_random_seed,
    recursive_getattr,
    get_parameters_by_name,
    polyak_update
)
from SCOPIL.utils.demonstration_utils import ExpertDataset

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(ABC):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param config: Dictionary with hyperparameters.
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    """

    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            config: dict,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            use_image_obs: bool = False,
            device: Union[th.device, str] = "auto",
            only_test: bool = False,
            chkpt_dir: bool = None,
            seed: Optional[int] = None
    ):

        self.config = config
        self.device = get_device(device)
        self.only_test = only_test
        self.chkpt_dir = chkpt_dir
        self.seed = seed

        ## Hyperparameters, variables and optimizers
        if self.only_test is False:
            self.actor_lr: float = self.config['SAC']['alpha']
            self.critic_lr: float = self.config['SAC']['beta']
            self.entr_coef_lr: float = self.config['SAC']['entropy_coefficient_lr']
            self.buffer_size: int = self.config['SAC']['buffer_memory_size']
            self.batch_size: int = self.config['SAC']['batch_size']
            self.learning_starts: int = self.config['SAC']['start_steps']
            self.tau: float = self.config['SAC']['tau']
            self.gamma: float = self.config['SAC']['gamma']
            self.gradient_steps: int = self.config['SAC']['gradient_steps']
            self.train_freq: int = self.config['SAC']['update_every_steps']
            # Used for gSDE only
            self.sde_sample_freq: int = self.config['SAC']['sde_sample_freq']
            self.use_sde_at_warmup: bool = self.config['SAC']['use_sde_at_warmup']
            # Entropy coefficient / Entropy temperature
            # Inverse of the reward scale
            self.target_entropy: Union[float, str] = self.config['SAC']['target_entropy']
            self.log_ent_coef: Optional[th.Tensor] = None
            self.ent_coef_tensor: Optional[th.Tensor] = None
            self.ent_coef: Union[float, str] = "auto"
            self.ent_coef_optimizer: Optional[th.optim.Adam] = None
            self.target_update_interval = self.config['SAC']['target_update_interval']
            # Constraints' optimization
            self.w_constraint_optimization = self.config['SAC']['w_constraint_optimization']
            self.constraint_lambda: Optional[th.Tensor] = None
            self.clip_grad_norm: bool = self.config['SAC']['clip_grad_norm']
            self.max_grad_norm: int = self.config['SAC']['max_grad_norm']
            self.adjust_entropy: bool = self.config['SAC']['adjust_entropy']
            self.w_mse: bool = self.config['SAC']['w_mse']
            self.mse_factor: float = self.config['SAC']['mse_factor']
            self.nll_factor: float = self.config['SAC']['nll_factor']
            self.pretrain: bool = self.config['SAC']['pretrain']
            self.pretrain_epochs: int = self.config['SAC']['pretrain_epochs']
            self.pretrain_mse_factor: float = self.config['SAC']['pretrain_mse_factor']
            self.pretrain_nll_factor: float = self.config['SAC']['pretrain_nll_factor']
            self.w_std_grads: bool = self.config['SAC']['w_std_grads']
            self.w_kl_div: bool = self.config['SAC']['w_kl_div']
            self.w_q_values: bool = self.config['SAC']['w_q_values']
            self.w_max_min: bool = self.config['SAC']['w_max_min']
            self.w_lower_bound: bool = self.config['SAC']['w_lower_bound']
            self.w_use_target_critic: bool = self.config['SAC']['w_use_target_critic']
            self.w_discriminator: bool = self.config['SAC']['w_discriminator']
            # Buffer placeholder
            self.replay_buffer: Optional[ReplayBuffer] = None
        # Define policy keyword arguments
        self.use_sde: bool = self.config['SAC']['use_sde']
        self.policy_kwargs: dict = {
            'net_arch': [self.config['SAC']['layer1_size'], self.config['SAC']['layer2_size']],
            'log_std_init': self.config['SAC']['log_std_init'],
            'use_sde': self.use_sde
        }

        # Define spaces
        self.use_image_obs = use_image_obs
        self.observation_space = observation_space
        self.action_space = action_space

        # Define the type of police according to the input
        if use_image_obs is True:
            self.policy_class = CnnPolicy
        else:
            self.policy_class = MlpPolicy

        # Do checks
        self.check_args()

        # Create the NNs
        self._setup_model()

    def check_args(self) -> None:
        supported_action_spaces = (spaces.Box,)
        assert isinstance(self.action_space, supported_action_spaces), (
            f"The algorithm only supports {supported_action_spaces} as action spaces "
            f"but {self.action_space} was provided"
        )
        assert np.all(
            np.isfinite(np.array([self.action_space.low, self.action_space.high]))
        ), "Continuous action space must have a finite lower and upper bound"

        if self.only_test is False:
            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError(
                    "generalized State-Dependent Exploration (gSDE) can only be used with continuous actions."
                )

    def _setup_model(self) -> None:

        self.set_random_seed(self.seed)

        if self.only_test is False:
            self.replay_buffer = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
            )
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.actor_lr,
                self.critic_lr,
                **self.policy_kwargs,
            )
        else:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                only_test=True,
                **self.policy_kwargs,
            )
        self.policy = self.policy.to(self.device)

        self._create_aliases()

        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        if self.only_test is False:
            # Target entropy is used when learning the entropy coefficient
            if self.target_entropy == "auto":
                # automatically set target entropy if needed
                self.target_entropy = float(-np.prod(self.action_space.shape).astype(np.float32))
            else:
                # Force conversion
                # this will also throw an error for unexpected string
                self.target_entropy = float(self.target_entropy)

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
                # Default initial value of ent_coef when learned
                init_value = 1.0
                if "_" in self.ent_coef:
                    init_value = float(self.ent_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff, which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
                self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.entr_coef_lr)
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

            if self.w_constraint_optimization is True:
                self.initial_lambda_constraint = self.config['SAC']['initial_lambda_constraint']
                self.constraint_lambda_lr = self.config['SAC']['lambda_constraint_lr']
                self.w_dual_grad_desc = self.config['SAC']['w_dual_grad_desc']
                # Define torch dataset for demonstrations
                expert_dataset = ExpertDataset(
                    self.config['SAC']['expert_dataset_path'],
                    device=self.device,
                    use_images=self.use_image_obs,
                    load_to_memory=self.config['SAC']['load_demos_in_memory'],
                    env_id=self.config['game']['env_id'],
                    normalize_features=self.config['Experiment']['normalize_features'],
                    smooth_actions=self.config['SAC']['smooth_actions'],
                    smooth_factor=self.config['SAC']['smooth_factor']
                )
                # Define torch loader based on torch dataset for training the policy wrt the constraints
                drop_last = len(expert_dataset) > self.batch_size
                self.expert_train_loader = th.utils.data.DataLoader(
                    dataset=expert_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=drop_last
                )
                if self.w_discriminator is True:
                    self.discriminator = Discriminator(
                        self.observation_space,
                        self.action_space,
                        self.config['SAC']['w_actions_in_discriminator'],
                        self.config['SAC']['discriminator_lr'],
                        self.config['SAC']['discriminator_batch_size'],
                        self.config['SAC']['discriminator_gradient_steps'],
                        self.config['SAC']['discriminator_layer1_size'],
                        self.config['SAC']['discriminator_layer2_size'],
                        self.replay_buffer,
                        self.expert_train_loader,
                        self.config['SAC']['w_discriminator_icrl_regularization'],
                        self.config['SAC']['discriminator_icrl_regularization_coef'],
                        self.config['SAC']['w_discriminator_dac_regularization'],
                        self.config['SAC']['discriminator_dac_regularization_coef'],
                        device=self.device,
                    )
                else:
                    # Define 'constraint_lambda'
                    self.constraint_lambda = th.tensor(
                        [self.initial_lambda_constraint],
                        dtype=th.float32,
                        requires_grad=self.w_dual_grad_desc,
                        device=self.device
                    )
                    # Define optimizer for 'constraint_lambda'
                    self.constraint_lambda_optimizer = None
                    if self.w_dual_grad_desc is True:
                        self.constraint_lambda_optimizer = th.optim.Adam(
                            [self.constraint_lambda],
                            lr=self.constraint_lambda_lr,
                            eps=1e-4
                        )
                if self.w_kl_div is True:
                    self.w_entropy_in_constraint_policy_loss_term = \
                        self.config['SAC']['w_entropy_in_constraint_policy_loss_term']
                    # Define the MSE loss function
                    if self.pretrain is True or self.w_mse is True:
                        self.mse_loss_func = th.nn.MSELoss()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self) -> dict:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Basic SAC logs
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        grad_norms_clipped = []
        mean_qvals = []
        min_qvals = []
        max_qvals = []
        mean_cur_qvals = []
        min_cur_qvals = []
        max_cur_qvals = []
        mean_next_qvals = []
        min_next_qvals = []
        max_next_qvals = []
        mean_target_qvals = []
        min_target_qvals = []
        max_target_qvals = []
        mean_logprobs = []
        min_logprobs = []
        max_logprobs = []
        mean_probs = []
        min_probs = []
        max_probs = []

        # Logs for constraint optimization
        constraint_policy_loss_term_values = []
        constraint_policy_loss_nll_term_values = []
        constraint_policy_loss_mse_term_values = []
        constraint_lambda_loss_values = []
        policy_loss_value_wo_constraint_terms = []
        constraint_lambdas = []
        mean_dem_qvals = []
        min_dem_qvals = []
        max_dem_qvals = []
        mean_dem_logprobs = []
        min_dem_logprobs = []
        max_dem_logprobs = []
        mean_dem_probs = []
        min_dem_probs = []
        max_dem_probs = []
        mean_cur_dem_qvals = []
        min_cur_dem_qvals = []
        max_cur_dem_qvals = []
        mean_cur_dem_logprobs = []
        min_cur_dem_logprobs = []
        max_cur_dem_logprobs = []
        mean_cur_dem_probs = []
        min_cur_dem_probs = []
        max_cur_dem_probs = []
        constraint_critic_loss_term_values = []
        critic_loss_values_wo_constraint_term = []
        lower_bound_constraint_critic_loss_term_values = []
        rollout_discr_preds_values = []
        expert_discr_preds_values = []
        mean_cur_dem_logprobs_value = None
        min_cur_dem_logprobs_value = None
        max_cur_dem_logprobs_value = None
        mean_cur_dem_probs_value = None
        min_cur_dem_probs_value = None
        max_cur_dem_probs_value = None
        mean_cur_dem_qvals_value = None
        min_cur_dem_qvals_value = None
        max_cur_dem_qvals_value = None
        mean_dem_logprobs_value = None
        min_dem_logprobs_value = None
        max_dem_logprobs_value = None
        mean_dem_probs_value = None
        min_dem_probs_value = None
        max_dem_probs_value = None
        mean_dem_qvals_value = None
        min_dem_qvals_value = None
        max_dem_qvals_value = None

        # First train the discriminator
        discriminator_train_logs = {}
        if self.w_discriminator is True:
            discriminator_train_logs = self.discriminator.update()

        for gradient_step in range(self.gradient_steps):

            # Sample replay buffer, that is, samples from agent-environment interactions
            replay_data = self.replay_buffer.sample(self.batch_size)

            # Samples from demonstrations
            expert_observations = None
            expert_actions = None
            if self.w_constraint_optimization is True:
                expert_actions, expert_observations = self.get_samples_from_demonstrations()

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Keep policy probs logs
            mean_logprobs.append(th.mean(log_prob).item())
            min_logprobs.append(log_prob.min().item())
            max_logprobs.append(log_prob.max().item())
            mean_probs.append(th.mean(log_prob).exp().item())
            min_probs.append(log_prob.min().exp().item())
            max_probs.append(log_prob.max().exp().item())

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph,
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -th.mean(self.log_ent_coef * (log_prob + self.target_entropy).detach())
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics' targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # Keep next and target Q-values for logs
                mean_next_qvals.append(th.mean(next_q_values).item())
                min_next_qvals.append(next_q_values.min().item())
                max_next_qvals.append(next_q_values.max().item())
                mean_target_qvals.append(th.mean(target_q_values).item())
                min_target_qvals.append(target_q_values.min().item())
                max_target_qvals.append(target_q_values.max().item())

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Keep current Q-values for logs
            min_current_q_values = th.min(th.cat(current_q_values, dim=1), dim=1, keepdim=True)[0]
            mean_cur_qvals.append(th.mean(min_current_q_values).item())
            min_cur_qvals.append(min_current_q_values.min().item())
            max_cur_qvals.append(min_current_q_values.max().item())

            ## Compute critic loss
            critic_loss_weights = 1.
            if self.w_discriminator is True:
                # Get predictions of Discriminator for the rollout state-action pairs
                rollout_discr_preds = self.discriminator.predict(replay_data.observations, replay_data.actions)
                critic_loss_weights = rollout_discr_preds
                # Also for the demonstrated state-action pairs, just for logs
                expert_discr_preds = self.discriminator.predict(
                    expert_observations, self.scale_and_clamp_demo_actions(expert_actions)
                )
                # Keep logs
                rollout_discr_preds_values.append(rollout_discr_preds.mean().item())
                expert_discr_preds_values.append(expert_discr_preds.mean().item())
            # SAC critic loss
            critic_loss = 0.5 * sum(
                th.mean(
                    th.pow(current_q - target_q_values, 2) * critic_loss_weights
                ) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            # Constraint critic loss
            if self.w_q_values is True:
                (
                    constraint_critic_loss,
                    lower_bound_constraint_critic_loss,
                    mean_cur_dem_logprobs_value,
                    min_cur_dem_logprobs_value,
                    max_cur_dem_logprobs_value,
                    mean_cur_dem_probs_value,
                    min_cur_dem_probs_value,
                    max_cur_dem_probs_value,
                    mean_cur_dem_qvals_value,
                    min_cur_dem_qvals_value,
                    max_cur_dem_qvals_value,
                    mean_dem_logprobs_value,
                    min_dem_logprobs_value,
                    max_dem_logprobs_value,
                    mean_dem_probs_value,
                    min_dem_probs_value,
                    max_dem_probs_value,
                    mean_dem_qvals_value,
                    min_dem_qvals_value,
                    max_dem_qvals_value
                 ) = self.calc_constraint_q_loss_term(
                    [expert_actions, expert_observations],
                    [replay_data.actions, replay_data.observations],
                    current_q_values,
                    critic_loss_weights
                )
                # Keep it for logs
                critic_loss_values_wo_constraint_term.append(critic_loss.item())
                constraint_critic_loss_term_values.append(constraint_critic_loss.item())
                lower_bound_constraint_critic_loss_term_values.append(lower_bound_constraint_critic_loss.item())
                # Add the constraint term to critic loss
                critic_loss_weight = 1.0
                if self.w_discriminator is False:
                    critic_loss_weight = self.constraint_lambda.item()
                critic_loss += critic_loss_weight * constraint_critic_loss
                if self.w_lower_bound is True:
                    critic_loss += critic_loss_weight * lower_bound_constraint_critic_loss
            # Keep for logs the total critic loss
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = th.mean(ent_coef * log_prob - min_qf_pi)

            # Keep Q-values for logs
            mean_qvals.append(th.mean(q_values_pi).item())
            min_qvals.append(q_values_pi.min().item())
            max_qvals.append(q_values_pi.max().item())

            ### Actor loss wrt constraints
            if self.w_kl_div is True:
                # Keep actor loss without the constraint term for logs
                policy_loss_value_wo_constraint_term = actor_loss.item()
                policy_loss_value_wo_constraint_terms.append(policy_loss_value_wo_constraint_term)
                ## Calculate policy loss wrt constraints
                (
                    constraint_policy_loss_term,
                    constraint_policy_loss_nll_term_value,
                    constraint_policy_loss_mse_term_value,
                    mean_cur_dem_logprobs_value,
                    min_cur_dem_logprobs_value,
                    max_cur_dem_logprobs_value,
                    mean_cur_dem_probs_value,
                    min_cur_dem_probs_value,
                    max_cur_dem_probs_value,
                    mean_cur_dem_qvals_value,
                    min_cur_dem_qvals_value,
                    max_cur_dem_qvals_value,
                    mean_dem_logprobs_value,
                    min_dem_logprobs_value,
                    max_dem_logprobs_value,
                    mean_dem_probs_value,
                    min_dem_probs_value,
                    max_dem_probs_value,
                    mean_dem_qvals_value,
                    min_dem_qvals_value,
                    max_dem_qvals_value
                ) = self.calc_constraint_policy_loss_term([expert_actions, expert_observations])
                # Add constraint term loss to policy loss
                actor_loss = actor_loss + self.constraint_lambda.item() * constraint_policy_loss_term
                # Keep it for logs
                constraint_policy_loss_term_value = constraint_policy_loss_term.item()
                constraint_policy_loss_term_values.append(constraint_policy_loss_term_value)
                constraint_policy_loss_nll_term_values.append(constraint_policy_loss_nll_term_value)
                constraint_policy_loss_mse_term_values.append(constraint_policy_loss_mse_term_value)

            # Keep the logs for the constraint optimization
            if self.w_kl_div is True or self.w_q_values is True:
                mean_cur_dem_logprobs.append(mean_cur_dem_logprobs_value)
                min_cur_dem_logprobs.append(min_cur_dem_logprobs_value)
                max_cur_dem_logprobs.append(max_cur_dem_logprobs_value)
                mean_cur_dem_probs.append(mean_cur_dem_probs_value)
                min_cur_dem_probs.append(min_cur_dem_probs_value)
                max_cur_dem_probs.append(max_cur_dem_probs_value)
                mean_cur_dem_qvals.append(mean_cur_dem_qvals_value)
                min_cur_dem_qvals.append(min_cur_dem_qvals_value)
                max_cur_dem_qvals.append(max_cur_dem_qvals_value)
                mean_dem_logprobs.append(mean_dem_logprobs_value)
                min_dem_logprobs.append(min_dem_logprobs_value)
                max_dem_logprobs.append(max_dem_logprobs_value)
                mean_dem_probs.append(mean_dem_probs_value)
                min_dem_probs.append(min_dem_probs_value)
                max_dem_probs.append(max_dem_probs_value)
                mean_dem_qvals.append(mean_dem_qvals_value)
                min_dem_qvals.append(min_dem_qvals_value)
                max_dem_qvals.append(max_dem_qvals_value)

            # Store total actor loss
            actor_losses.append(actor_loss.item())

            ## Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping
            if self.clip_grad_norm is True:
                grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                grad_norm_clipped = grad_norm.item() - self.max_grad_norm
                grad_norms_clipped.append(grad_norm_clipped)
            self.actor.optimizer.step()

            ## Computation of 'constraint_lambda' loss and optimizer step
            if (
                    self.w_constraint_optimization is True and
                    (
                            self.w_kl_div is True or
                            (self.w_q_values is True and self.w_discriminator is False)
                    )
            ):
                # Calculate lambda loss
                if self.w_kl_div is True:
                    constraint_lambda_loss = self.calc_constraint_lambda_loss_w_kl_div(
                        [expert_actions, expert_observations]
                    )
                elif self.w_q_values is True:
                    constraint_lambda_loss = self.calc_constraint_lambda_loss_w_q_values(
                        [expert_actions, expert_observations],
                        [replay_data.actions, replay_data.observations],
                        current_q_values
                    )
                else:
                    raise NotImplementedError('The constraint optimization is not implemented for this case.')
                # Keep it for logs
                constraint_lambda_loss_value = constraint_lambda_loss.item()
                constraint_lambda_loss_values.append(constraint_lambda_loss_value)
                if self.w_dual_grad_desc is True:
                    # 'lambda' optimizer step
                    self.constraint_lambda_optimizer.zero_grad()
                    constraint_lambda_loss.backward()
                    self.constraint_lambda_optimizer.step()
                # Keep the lambda value for logs
                constraint_lambda_value = self.constraint_lambda.item()
                constraint_lambdas.append(constraint_lambda_value)

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        constraint_policy_loss_term_values_mean = np.nan
        constraint_policy_loss_nll_term_values_mean = np.nan
        constraint_policy_loss_mse_term_values_mean = np.nan
        constraint_lambda_loss_values_mean = np.nan
        policy_loss_value_wo_constraint_terms_mean = np.nan
        constraint_lambdas_mean = np.nan
        grad_norms_clipped_mean = np.nan
        mean_dem_qvals_mean = np.nan
        min_dem_qvals_mean = np.nan
        max_dem_qvals_mean = np.nan
        mean_dem_logprobs_mean = np.nan
        min_dem_logprobs_mean = np.nan
        max_dem_logprobs_mean = np.nan
        mean_dem_probs_mean = np.nan
        min_dem_probs_mean = np.nan
        max_dem_probs_mean = np.nan
        mean_cur_dem_qvals_mean = np.nan
        min_cur_dem_qvals_mean = np.nan
        max_cur_dem_qvals_mean = np.nan
        mean_cur_dem_logprobs_mean = np.nan
        min_cur_dem_logprobs_mean = np.nan
        max_cur_dem_logprobs_mean = np.nan
        mean_cur_dem_probs_mean = np.nan
        min_cur_dem_probs_mean = np.nan
        max_cur_dem_probs_mean = np.nan
        critic_loss_values_wo_constraint_term_mean = np.nan
        constraint_critic_loss_term_values_mean = np.nan
        lower_bound_constraint_critic_loss_term_values_mean = np.nan
        rollout_discr_preds_values_mean = np.nan
        expert_discr_preds_values_mean = np.nan
        if self.w_constraint_optimization is True:
            mean_dem_qvals_mean = mean(mean_dem_qvals)
            min_dem_qvals_mean = mean(min_dem_qvals)
            max_dem_qvals_mean = mean(max_dem_qvals)
            mean_dem_logprobs_mean = mean(mean_dem_logprobs)
            min_dem_logprobs_mean = mean(min_dem_logprobs)
            max_dem_logprobs_mean = mean(max_dem_logprobs)
            mean_dem_probs_mean = mean(mean_dem_probs)
            min_dem_probs_mean = mean(min_dem_probs)
            max_dem_probs_mean = mean(max_dem_probs)
            mean_cur_dem_qvals_mean = mean(mean_cur_dem_qvals)
            min_cur_dem_qvals_mean = mean(min_cur_dem_qvals)
            max_cur_dem_qvals_mean = mean(max_cur_dem_qvals)
            mean_cur_dem_logprobs_mean = mean(mean_cur_dem_logprobs)
            min_cur_dem_logprobs_mean = mean(min_cur_dem_logprobs)
            max_cur_dem_logprobs_mean = mean(max_cur_dem_logprobs)
            mean_cur_dem_probs_mean = mean(mean_cur_dem_probs)
            min_cur_dem_probs_mean = mean(min_cur_dem_probs)
            max_cur_dem_probs_mean = mean(max_cur_dem_probs)
            if self.w_discriminator is False:
                constraint_lambda_loss_values_mean = mean(constraint_lambda_loss_values)
                constraint_lambdas_mean = mean(constraint_lambdas)
            if self.w_kl_div is True:
                constraint_policy_loss_term_values_mean = mean(constraint_policy_loss_term_values)
                constraint_policy_loss_nll_term_values_mean = mean(constraint_policy_loss_nll_term_values)
                constraint_policy_loss_mse_term_values_mean = mean(constraint_policy_loss_mse_term_values)
                policy_loss_value_wo_constraint_terms_mean = mean(policy_loss_value_wo_constraint_terms)
            if self.w_q_values is True:
                critic_loss_values_wo_constraint_term_mean = mean(critic_loss_values_wo_constraint_term)
                constraint_critic_loss_term_values_mean = mean(constraint_critic_loss_term_values)
                lower_bound_constraint_critic_loss_term_values_mean = mean(
                    lower_bound_constraint_critic_loss_term_values
                )
                if self.w_discriminator is True:
                    rollout_discr_preds_values_mean = mean(rollout_discr_preds_values)
                    expert_discr_preds_values_mean = mean(expert_discr_preds_values)
        if self.clip_grad_norm is True:
            grad_norms_clipped_mean = mean(grad_norms_clipped)

        logs_dict = {
            'actor_loss': mean(actor_losses),
            'critic_loss': mean(critic_losses),
            'entr_coef_loss': np.nan if self.ent_coef_optimizer is None else mean(ent_coef_losses),
            'entr_coef': mean(ent_coefs),
            'constraint_policy_loss_term': constraint_policy_loss_term_values_mean,
            'constraint_policy_loss_nll_term': constraint_policy_loss_nll_term_values_mean,
            'constraint_policy_loss_mse_term': constraint_policy_loss_mse_term_values_mean,
            'constraint_lambda_loss': constraint_lambda_loss_values_mean,
            'policy_loss_value_wo_constraint_term': policy_loss_value_wo_constraint_terms_mean,
            'constraint_lambda': constraint_lambdas_mean,
            'grad_norms_clipped': grad_norms_clipped_mean,
            'mean_qvals': mean(mean_qvals),
            'min_qvals': mean(min_qvals),
            'max_qvals': mean(max_qvals),
            'mean_cur_qvals': mean(mean_cur_qvals),
            'min_cur_qvals': mean(min_cur_qvals),
            'max_cur_qvals': mean(max_cur_qvals),
            'mean_next_qvals': mean(mean_next_qvals),
            'min_next_qvals': mean(min_next_qvals),
            'max_next_qvals': mean(max_next_qvals),
            'mean_target_qvals': mean(mean_target_qvals),
            'min_target_qvals': mean(min_target_qvals),
            'max_target_qvals': mean(max_target_qvals),
            'mean_logprobs': mean(mean_logprobs),
            'min_logprobs': mean(min_logprobs),
            'max_logprobs': mean(max_logprobs),
            'mean_probs': mean(mean_probs),
            'min_probs': mean(min_probs),
            'max_probs': mean(max_probs),
            'mean_dem_qvals': mean_dem_qvals_mean,
            'min_dem_qvals': min_dem_qvals_mean,
            'max_dem_qvals': max_dem_qvals_mean,
            'mean_dem_logprobs': mean_dem_logprobs_mean,
            'min_dem_logprobs': min_dem_logprobs_mean,
            'max_dem_logprobs': max_dem_logprobs_mean,
            'mean_dem_probs': mean_dem_probs_mean,
            'min_dem_probs': min_dem_probs_mean,
            'max_dem_probs': max_dem_probs_mean,
            'mean_cur_dem_qvals': mean_cur_dem_qvals_mean,
            'min_cur_dem_qvals': min_cur_dem_qvals_mean,
            'max_cur_dem_qvals': max_cur_dem_qvals_mean,
            'mean_cur_dem_logprobs': mean_cur_dem_logprobs_mean,
            'min_cur_dem_logprobs': min_cur_dem_logprobs_mean,
            'max_cur_dem_logprobs': max_cur_dem_logprobs_mean,
            'mean_cur_dem_probs': mean_cur_dem_probs_mean,
            'min_cur_dem_probs': min_cur_dem_probs_mean,
            'max_cur_dem_probs': max_cur_dem_probs_mean,
            'critic_loss_values_wo_constraint_term': critic_loss_values_wo_constraint_term_mean,
            'constraint_critic_loss_term_values': constraint_critic_loss_term_values_mean,
            'lower_bound_constraint_critic_loss_term_values': lower_bound_constraint_critic_loss_term_values_mean,
            'rollout_discr_preds': rollout_discr_preds_values_mean,
            'expert_discr_preds': expert_discr_preds_values_mean,
        }
        logs_dict.update(discriminator_train_logs)

        return logs_dict

    def calc_constraint_lambda_loss_w_kl_div(self, demonstrations):

        constraint_policy_loss_term, *_ = self.calc_constraint_policy_loss_term(demonstrations)
        constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * constraint_policy_loss_term.item()

        return constraint_lambda_loss

    def calc_constraint_lambda_loss_w_q_values(self, demonstrations, rollout_data, rollout_q_values):

        constraint_critic_loss, lower_bound_constraint_critic_loss, *_ = self.calc_constraint_q_loss_term(
            demonstrations,
            rollout_data,
            rollout_q_values
        )

        constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * constraint_critic_loss.item()
        if self.w_lower_bound is True:
            constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * lower_bound_constraint_critic_loss.item()

        return constraint_lambda_loss

    def calc_constraint_policy_loss_term(self, demonstrations):
        """
        Calculates policy loss wrt the specified constraints

        :param demonstrations: List with: 1) actions of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, *action_space.shape],
                                      and 2) observations of type torch.Tensor with dtype=torch.float32 and
                                             shape=[batch_size, *observation_space.shape].
        :return: Policy loss wrt constraints.
        """

        actions = demonstrations[0]
        observations = demonstrations[1]
        self.check_demonstrations_format(actions, observations)

        # Log probabilities and entropy of the given actions
        log_probs, actions_entropy = self.actor.evaluate_actions(
            observations,
            actions,
            scale_actions=True,
            adjust_entropy=self.adjust_entropy,
            w_std_grads=self.w_std_grads
        )
        if self.w_entropy_in_constraint_policy_loss_term is True:
            # Entropy regularized NLL
            if self.log_ent_coef is not None:
                ent_coef = self.log_ent_coef.exp().item()
            else:
                ent_coef = self.ent_coef_tensor.item()
            constraint_policy_loss_term = -th.mean(log_probs) - (ent_coef * th.mean(actions_entropy))
        else:
            # NLL
            constraint_policy_loss_term = -th.mean(log_probs)
        # Keep it for logs
        constraint_policy_loss_nll_term_value = constraint_policy_loss_term.item()

        # MSE between the policy's actions and the expert actions
        actions_pi, log_probs_pi = self.actor.action_log_prob(observations)
        scaled_expert_actions = self.scale_and_clamp_demo_actions(actions)
        mse_loss = self.mse_loss_func(actions_pi, scaled_expert_actions)
        if self.w_mse is True:
            constraint_policy_loss_term = self.nll_factor*constraint_policy_loss_term + self.mse_factor*mse_loss
        # Keep it for logs
        constraint_policy_loss_mse_term_value = mse_loss.item()

        # Keep logs for probs of the demonstrated actions
        mean_cur_dem_logprobs_value = th.mean(log_probs).item()
        min_cur_dem_logprobs_value = log_probs.min().item()
        max_cur_dem_logprobs_value = log_probs.max().item()
        mean_cur_dem_probs_value = th.mean(log_probs).exp().item()
        min_cur_dem_probs_value = th.min(log_probs).exp().item()
        max_cur_dem_probs_value = th.max(log_probs).exp().item()

        # Keep logs for Q-values of the demonstrated actions
        q_values = th.cat(self.critic(observations, scaled_expert_actions), dim=1)
        min_qf, _ = th.min(q_values, dim=1, keepdim=True)
        mean_cur_dem_qvals_value = th.mean(min_qf).item()
        min_cur_dem_qvals_value = min_qf.min().item()
        max_cur_dem_qvals_value = min_qf.max().item()

        # Keep logs for probs of the policy actions
        mean_dem_logprobs_value = th.mean(log_probs_pi).item()
        min_dem_logprobs_value = log_probs_pi.min().item()
        max_dem_logprobs_value = log_probs_pi.max().item()
        mean_dem_probs_value = th.mean(log_probs_pi).exp().item()
        min_dem_probs_value = log_probs_pi.min().exp().item()
        max_dem_probs_value = log_probs_pi.max().exp().item()

        # Keep logs for Q-values of the demonstrated actions
        q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        mean_dem_qvals_value = th.mean(min_qf_pi).item()
        min_dem_qvals_value = min_qf_pi.min().item()
        max_dem_qvals_value = min_qf_pi.max().item()

        return (
            constraint_policy_loss_term,
            constraint_policy_loss_nll_term_value,
            constraint_policy_loss_mse_term_value,
            mean_cur_dem_logprobs_value,
            min_cur_dem_logprobs_value,
            max_cur_dem_logprobs_value,
            mean_cur_dem_probs_value,
            min_cur_dem_probs_value,
            max_cur_dem_probs_value,
            mean_cur_dem_qvals_value,
            min_cur_dem_qvals_value,
            max_cur_dem_qvals_value,
            mean_dem_logprobs_value,
            min_dem_logprobs_value,
            max_dem_logprobs_value,
            mean_dem_probs_value,
            min_dem_probs_value,
            max_dem_probs_value,
            mean_dem_qvals_value,
            min_dem_qvals_value,
            max_dem_qvals_value
        )

    def calc_constraint_q_loss_term(
            self,
            demonstrations: List[th.Tensor],
            rollout_data: List[th.Tensor],
            rollout_q_values: Tuple[th.Tensor],
            discriminator_rollout_preds: Optional[th.Tensor] = None
    ) -> (
            th.Tensor,
            th.Tensor,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
    ):
        """
        Calculates policy loss wrt the specified constraints

        :param demonstrations: List with: 1) actions of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, *action_space.shape],
                                      and 2) observations of type torch.Tensor with dtype=torch.float32 and
                                             shape=[batch_size, *observation_space.shape].
        :param rollout_data: List similar to 'demonstrations'.
        :param rollout_q_values: Tuple with the Q-values of the rollout state-action pairs.
            Note that the tuple consists of two tensors, one for the estimated Q-values of each critic.
        :param discriminator_rollout_preds: Predictions of the Discriminator for the rollout state-action pairs.
            Applicable only when self.w_discriminator is True.

        :return: Critic Q loss wrt constraints.
        """

        # Demonstrations' state-action pairs
        actions = demonstrations[0]
        observations = demonstrations[1]
        self.check_demonstrations_format(actions, observations)

        # Rollouts' state-action pairs
        rollout_actions = rollout_data[0]
        rollout_observations = rollout_data[1]

        ## Compute the constraint critic loss
        # Get the minimum Q-value of the demonstrated state-action pairs to use it as the target (without grads)
        scaled_expert_actions = self.scale_and_clamp_demo_actions(actions)
        dem_q_values = self.critic(observations, scaled_expert_actions)
        dem_q_values_catted = th.cat(dem_q_values, dim=1)
        min_dem_q_value = th.min(dem_q_values_catted).detach()
        if self.w_use_target_critic is True:
            min_dem_q_value = th.min(th.cat(self.critic_target(observations, scaled_expert_actions), dim=1)).detach()
        # Constraint critic loss
        constraint_critic_loss = 0.5 * sum(
            th.mean(
                th.pow(
                    (th.maximum(rollout_q, min_dem_q_value) if self.w_max_min is True else rollout_q) - min_dem_q_value,
                    2
                ) * (1. if self.w_discriminator is False else (1. - discriminator_rollout_preds))
            ) for rollout_q in rollout_q_values
        )

        ## Compute the lower bound constraint loss
        # Get the maximum Q-value of the rollout state-action pairs to use it as the target (without grads)
        max_rollout_q_value = th.max(th.cat(rollout_q_values, dim=1)).detach()
        if self.w_use_target_critic is True:
            max_rollout_q_value = th.max(
                th.cat(self.critic_target(rollout_observations, rollout_actions), dim=1)
            ).detach()
        # Lower bound constraint loss
        lower_bound_constraint_critic_loss = 0.5 * sum(
            th.mean(
                th.pow(
                    (th.minimum(dem_q, max_rollout_q_value) if self.w_max_min is True else dem_q) - max_rollout_q_value,
                    2
                )
            ) for dem_q in dem_q_values
        )

        ## Keep for logs
        # Log probabilities of the given actions
        log_probs, _ = self.actor.evaluate_actions(
            observations,
            actions,
            scale_actions=True,
            adjust_entropy=False,
            w_std_grads=False
        )
        mean_cur_dem_logprobs_value = th.mean(log_probs).item()
        min_cur_dem_logprobs_value = log_probs.min().item()
        max_cur_dem_logprobs_value = log_probs.max().item()
        mean_cur_dem_probs_value = th.mean(log_probs).exp().item()
        min_cur_dem_probs_value = th.min(log_probs).exp().item()
        max_cur_dem_probs_value = th.max(log_probs).exp().item()
        # Q-values of the demonstrated actions
        min_dem_q_values, _ = th.min(dem_q_values_catted, dim=1, keepdim=True)
        mean_cur_dem_qvals_value = th.mean(min_dem_q_values).item()
        min_cur_dem_qvals_value = min_dem_q_values.min().item()
        max_cur_dem_qvals_value = min_dem_q_values.max().item()
        # Probs of the policy actions
        actions_pi, log_probs_pi = self.actor.action_log_prob(observations)
        mean_dem_logprobs_value = th.mean(log_probs_pi).item()
        min_dem_logprobs_value = log_probs_pi.min().item()
        max_dem_logprobs_value = log_probs_pi.max().item()
        mean_dem_probs_value = th.mean(log_probs_pi).exp().item()
        min_dem_probs_value = log_probs_pi.min().exp().item()
        max_dem_probs_value = log_probs_pi.max().exp().item()
        # Q-values of the demonstrated actions
        q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
        min_q_values_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        mean_dem_qvals_value = th.mean(min_q_values_pi).item()
        min_dem_qvals_value = min_q_values_pi.min().item()
        max_dem_qvals_value = min_q_values_pi.max().item()

        return (
            constraint_critic_loss,
            lower_bound_constraint_critic_loss,
            mean_cur_dem_logprobs_value,
            min_cur_dem_logprobs_value,
            max_cur_dem_logprobs_value,
            mean_cur_dem_probs_value,
            min_cur_dem_probs_value,
            max_cur_dem_probs_value,
            mean_cur_dem_qvals_value,
            min_cur_dem_qvals_value,
            max_cur_dem_qvals_value,
            mean_dem_logprobs_value,
            min_dem_logprobs_value,
            max_dem_logprobs_value,
            mean_dem_probs_value,
            min_dem_probs_value,
            max_dem_probs_value,
            mean_dem_qvals_value,
            min_dem_qvals_value,
            max_dem_qvals_value
        )

    def check_demonstrations_format(self, actions, observations):
        """
        Check if the provided demonstration samples have the right format.

        :param actions: actions of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, *action_space.shape].
        :param observations: observations of type torch.Tensor with dtype=torch.float32 and
            shape=[batch_size, *observation_space.shape].

        :return:
        """

        assert actions.shape[1:] == self.action_space.shape, \
            (
                    "actions.shape: " + str(actions.shape) +
                    " self.action_space.shape: " + str(self.action_space.shape)
            )
        assert observations.shape[1:] == self.observation_space.shape, \
            (
                    "observations.shape: " + str(observations.shape) +
                    " self.observation_space.shape: " + str(self.observation_space.shape)
            )

    def get_samples_from_demonstrations(self):
        expert_actions, expert_observations = next(iter(self.expert_train_loader))

        return expert_actions, expert_observations

    def pretrain_func(self) -> dict:

        # Initialize vars for logging
        mse_losses = []
        nll_losses = []
        losses = []
        grad_norms_clipped = []
        log_probs = []
        probs = []

        for epoch in range(self.pretrain_epochs):
            print(f"\nPretraining epoch: {epoch}")

            # Initialize epoch vars for logging
            epoch_mse_losses = []
            epoch_nll_losses = []
            epoch_losses = []
            epoch_grad_norms_clipped = []
            epoch_log_probs = []
            epoch_probs = []

            for expert_actions, expert_observations in self.expert_train_loader:

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    self.actor.reset_noise()

                # Action by the current actor for the provided demonstrations
                actions_pi, log_prob = self.actor.action_log_prob(expert_observations)

                # Log probabilities and entropy of the given actions
                expert_log_prob, expert_actions_entropy = self.actor.evaluate_actions(
                    expert_observations,
                    expert_actions,
                    scale_actions=True,
                    adjust_entropy=self.adjust_entropy,
                    w_std_grads=self.w_std_grads
                )

                # Compute loss
                scaled_expert_actions = self.scale_and_clamp_demo_actions(expert_actions)
                mse_loss = self.mse_loss_func(actions_pi, scaled_expert_actions)
                nll_loss = -th.mean(expert_log_prob)
                loss = self.pretrain_mse_factor*mse_loss + self.pretrain_nll_factor*nll_loss

                # Keep logs
                epoch_mse_losses.append(mse_loss.item())
                epoch_nll_losses.append(nll_loss.item())
                epoch_losses.append(loss.item())
                epoch_log_probs.append(th.mean(expert_log_prob).item())
                epoch_probs.append(th.exp(th.mean(expert_log_prob)).item())

                ## Actor optimizer step
                self.actor.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.clip_grad_norm is True:
                    grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    grad_norm_clipped = grad_norm.item() - self.max_grad_norm
                    epoch_grad_norms_clipped.append(grad_norm_clipped)
                self.actor.optimizer.step()

            # Log the mean values of the epoch
            mse_losses.append(mean(epoch_mse_losses))
            nll_losses.append(mean(epoch_nll_losses))
            losses.append(mean(epoch_losses))
            log_probs.append(mean(epoch_log_probs))
            probs.append(mean(epoch_probs))
            if self.clip_grad_norm is True:
                grad_norms_clipped.append(mean(epoch_grad_norms_clipped))

            # Print the mean values of the epoch
            print(f"MSE loss: {round(mse_losses[-1], 2)}")
            print(f"NLL loss: {round(nll_losses[-1], 2)}")
            print(f"Loss: {round(losses[-1], 2)}")
            print(f"Log probs: {round(log_probs[-1], 2)}")
            print(f"Probs: {round(probs[-1], 6)}")
            if self.clip_grad_norm is True:
                print(f"Grad norms clipped: {round(grad_norms_clipped[-1], 2)}")

        print("\nPretraining completed!")

        return {
            'mse_loss': mse_losses,
            'nll_loss': nll_losses,
            'loss': losses,
            'log_probs': log_probs,
            'probs': probs,
            'grad_norms_clipped': grad_norms_clipped
        }

    def scale_and_clamp_demo_actions(self, expert_actions):
        scaled_expert_actions = self.actor.scale_action(expert_actions)

        # Since the policy projects the actions to (-1, 1) using tanh, we need to clamp the actions;
        # otherwise, the probability will be zero.
        eps = 0.0001
        scaled_expert_actions = scaled_expert_actions.clamp(min=-1.0 + eps, max=1.0 - eps)

        return scaled_expert_actions

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def sample_action(
            self,
            obs: np.ndarray,
            num_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param obs: The agent's observation
        :param num_timesteps: The current number of timesteps
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differ when the action space is not normalized (bounds are not [-1, 1]).
        """

        # Select action randomly or according to policy
        if num_timesteps < self.learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC
            unscaled_action = self.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        scaled_action = self.policy.scale_action(unscaled_action)
        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)

        # Remove batch dimension if needed
        if len(action.shape) == 2:
            assert action.shape[0] == 1, f"action shape: {action.shape}"
            action = action[0]
        if len(buffer_action.shape) == 2:
            assert buffer_action.shape[0] == 1, f"buffer_action shape: {buffer_action.shape}"
            buffer_action = buffer_action[0]

        return action, buffer_action

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(self, prefix: str, path: Union[str, pathlib.Path], override: bool = True) -> None:

        # If all models should be saved, the corresponding directory should be created
        if override is False:
            checkpoint_dir_suffix = 0
            while os.path.exists(os.path.join(path, f"chkpts_{checkpoint_dir_suffix}")):
                checkpoint_dir_suffix += 1
            path = os.path.join(path, f"chkpts_{checkpoint_dir_suffix}")
            os.mkdir(path)
        else:
            path = os.path.join(path, "chkpts")
            if os.path.exists(path) is False:
                os.mkdir(path)

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        # Save Actors and Critics
        actor_path = os.path.join(path, f"{prefix}_sac_actor.pt")
        self.actor.save(actor_path)
        print('Saving {} to {} ...'.format('actor', actor_path))
        critic_path = os.path.join(path, f"{prefix}_sac_critic.pt")
        self.critic.save(critic_path)
        print('Saving {} to {} ...'.format('critic', critic_path))
        critic_target_path = os.path.join(path, f"{prefix}_sac_critic_target.pt")
        self.critic_target.save(critic_target_path)
        print('Saving {} to {} ...'.format('target critic', critic_target_path))

        # Save entropy coefficient
        if self.log_ent_coef is not None:
            log_ent_coef_path = os.path.join(path, f'{prefix}_sac_log_ent_coef.pt')
            print('Saving {} to {} ...'.format('log_ent_coef', log_ent_coef_path))
            th.save(self.log_ent_coef, log_ent_coef_path)
        else:
            ent_coef_path = os.path.join(path, f'{prefix}_sac_ent_coef.pt')
            print('Saving {} to {} ...'.format('ent_coef', ent_coef_path))
            th.save(self.ent_coef_tensor, ent_coef_path)

        # Save 'constraint_lambda' variable
        if self.w_constraint_optimization is True:
            constraint_lambda_path = os.path.join(path, f'{prefix}_sac_constraint_lambda.pt')
            print('Saving {} to {} ...'.format('constraint lambda', constraint_lambda_path))
            th.save(self.constraint_lambda, constraint_lambda_path)

    def load(
            self,
            prefix: str,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        """
        Load the models from .pt files.

        :param prefix: Prefix of the model files
        :param path: Path to the directory where the models are stored
        """

        # Load Actors and Critics
        actor_path = os.path.join(path, f'{prefix}_sac_actor.pt')
        self.actor.load_model(actor_path, self.device)
        print('Loading {} from {} ...'.format('actor', actor_path))
        critic_path = os.path.join(path, f'{prefix}_sac_critic.pt')
        self.critic.load_model(critic_path, self.device)
        print('Loading {} from {} ...'.format('critic', critic_path))
        critic_target_path = os.path.join(path, f'{prefix}_sac_critic_target.pt')
        self.critic_target.load_model(critic_target_path, self.device)
        print('Loading {} from {} ...'.format('target critic', critic_target_path))

        if self.only_test is False:
            # Load entropy coefficient
            if self.log_ent_coef is not None:
                log_ent_coef_path = os.path.join(path, f'{prefix}_sac_log_ent_coef.pt')
                print('Loading {} from {} ...'.format('log_ent_coef', log_ent_coef_path))
                self.log_ent_coef = th.load(log_ent_coef_path, device=self.device)
            else:
                ent_coef_path = os.path.join(path, f'{prefix}_sac_ent_coef.pt')
                print('Loading {} from {} ...'.format('ent_coef', ent_coef_path))
                self.ent_coef_tensor = th.load(ent_coef_path, device=self.device)

            # Load 'constraint_lambda' variable
            if self.w_constraint_optimization is True:
                constraint_lambda_path = os.path.join(path, f'{prefix}_sac_constraint_lambda.pt')
                print('Loading {} from {} ...'.format('constraint lambda', constraint_lambda_path))
                self.constraint_lambda = th.load(constraint_lambda_path, device=self.device)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g., normalizing images).

        :param observation: the input observation
        :param deterministic: Whether to return deterministic actions or not.
        :return: the model's action
        """
        return self.policy.predict(observation, deterministic)
