# Code based on: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
import copy
import os
import pathlib
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from abc import ABC
import io
from statistics import mean

import numpy as np
import torch
import torch as th
from gymnasium import spaces

from .buffer import ReplayBuffer
from .discriminator import Discriminator
from .networks_continuous import Actor, CnnPolicy, MlpPolicy, SACPolicy, ContinuousCritic
from .utils import (
    get_device,
    recursive_getattr,
    get_parameters_by_name,
    polyak_update
)
from SCOPIL.utils.demonstration_utils import ExpertDataset, stats_discounted_rew_values
from SCOPIL.utils.torch_utils import ExpectileLoss
from SCOPIL.utils.exp_utils import set_random_seed

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
            seed: Optional[int] = None
    ):
        self.config = config
        self.device = get_device(device)
        self.only_test = only_test
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
            # Used for gSDE only
            self.use_sde_at_warmup: bool = self.config['SAC']['use_sde_at_warmup']
            # Entropy coefficient / Entropy temperature
            # Inverse of the reward scale
            self.target_entropy: Union[float, str] = self.config['SAC']['target_entropy']
            self.log_ent_coef: Optional[th.Tensor] = None
            self.ent_coef_tensor: Optional[th.Tensor] = None
            self.ent_coef: Union[float, str] = self.config['SAC']['init_entropy_coef']
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
            self.max_min_coef: float = self.config['SAC']['max_min_coef']
            self.w_lower_bound: bool = self.config['SAC']['w_lower_bound']
            self.w_expectile_loss: bool = self.config['SAC']['w_expectile_loss']
            self.expectile_t: float = self.config['SAC']['expectile_t']
            self.w_use_target_critic: bool = self.config['SAC']['w_use_target_critic']
            self.w_compute_analytically_min_dem_q_value: bool = self.config['SAC']['w_compute_analytically_min_dem_q_value']
            self.dem_q_value_stat: str = self.config['SAC']['dem_q_value_stat']
            self.w_closest_state_min: bool = self.config['SAC']['w_closest_state_min']
            self.closest_state_min_func: bool = self.config['SAC']['closest_state_min_func']
            self.w_discriminator: bool = self.config['SAC']['w_discriminator']
            self.w_threshold_in_discriminator_weights: bool = self.config['SAC']['w_threshold_in_discriminator_weights']
            self.threshold_in_discriminator_weights: float = self.config['SAC']['threshold_in_discriminator_weights']
            self.w_demonstrations_rl_term: bool = self.config['SAC']['w_demonstrations_rl_term']
            self.demonstrations_rl_term_coef: float = self.config['SAC']['demonstrations_rl_term_coef']
            self.w_demonstrations_next_actions_in_demonstrations_rl_term: bool = self.config['SAC']['w_demonstrations_next_actions_in_demonstrations_rl_term']
            self.w_entropy_in_demonstrations_rl_term: bool = self.config['SAC']['w_entropy_in_demonstrations_rl_term']
            self.w_compute_analytically_target_in_demonstrations_rl_term: bool = self.config['SAC']['w_compute_analytically_target_in_demonstrations_rl_term']
            self.w_ood_rl_term: bool = self.config['SAC']['w_ood_rl_term']
            self.ood_rl_term_coef: float = self.config['SAC']['ood_rl_term_coef']
            self.w_discriminator_rewards_in_ood_rl_term: bool = self.config['SAC']['w_discriminator_rewards_in_ood_rl_term']
            self.discriminator_reward_function_in_ood_rl_term: str = self.config['SAC']['discriminator_reward_function_in_ood_rl_term']
            self.w_discriminator_discounted_rewards_in_ood_rl_term: bool = self.config['SAC']['w_discriminator_discounted_rewards_in_ood_rl_term']
            self.w_entropy_in_ood_rl_term: bool = self.config['SAC']['w_entropy_in_ood_rl_term']
            self.w_gail_sac: bool = self.config['SAC']['w_gail_sac']
            self.discriminator_reward_function_in_gail_sac: str = self.config['SAC']['discriminator_reward_function_in_gail_sac']
            self.w_target_discriminator: bool = self.config['SAC']['w_target_discriminator']
            self.discriminator_tau: float = self.config['SAC']['discriminator_tau']
            self.discriminator_target_update_interval: int = self.config['SAC']['discriminator_target_update_interval']
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
                self.expert_dataset = ExpertDataset(
                    self.config['SAC']['expert_dataset_path'],
                    device=self.device,
                    use_images=self.use_image_obs,
                    load_to_memory=self.config['SAC']['load_demos_in_memory'],
                    env_id=self.config['game']['env_id'],
                    normalize_features=self.config['Experiment']['normalize_features'],
                    normalize_rewards=self.config['Experiment']['normalize_rewards'],
                    smooth_actions=self.config['SAC']['smooth_actions'],
                    smooth_factor=self.config['SAC']['smooth_factor'],
                    compute_discounted_rewards=self.w_compute_analytically_target_in_demonstrations_rl_term,
                    build_search_memory=self.w_closest_state_min,
                    search_func=self.closest_state_min_func,
                )
                # Define torch loader based on torch dataset for training the policy wrt the constraints
                drop_last = len(self.expert_dataset) > self.batch_size
                self.expert_train_loader = th.utils.data.DataLoader(
                    dataset=self.expert_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                    num_workers=self.config['Experiment']['dataloader_num_workers'],
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
                    self.discriminator_to_use = self.discriminator
                    if self.w_target_discriminator is True:
                        # Create the target Discriminator as a copy of the Discriminator
                        self.discriminator_target = copy.deepcopy(self.discriminator)
                        # Target networks should always be in eval mode
                        self.discriminator_target.set_training_mode(False)
                        # Running mean and running var
                        self.discriminator_batch_norm_stats = get_parameters_by_name(self.discriminator, ["running_"])
                        self.discriminator_batch_norm_stats_target = get_parameters_by_name(self.discriminator_target, ["running_"])
                        self.discriminator_to_use = self.discriminator_target
                    # In the case of the min closest state search, pass the 'Discriminator' to the 'Dataset'
                    self.expert_dataset.discriminator = self.discriminator_to_use
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
                if self.w_q_values is True:
                    if self.w_compute_analytically_min_dem_q_value is True:
                        (
                            self.max_disc_reward,
                            self.min_disc_reward,
                            self.median_disc_reward,
                            self.mean_disc_reward,
                            self.twenty_five_quant_disc_reward,
                            self.seventy_five_quant_disc_reward
                        ) = stats_discounted_rew_values(
                            self.config['game']['env_id'],
                            self.config['Experiment']['normalize_rewards'],
                            self.config['SAC']['expert_dataset_path'],
                            self.gamma
                        )
                        self.max_disc_reward = th.tensor(
                            [self.max_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                        self.min_disc_reward = th.tensor(
                            [self.min_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                        self.median_disc_reward = th.tensor(
                            [self.median_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                        self.mean_disc_reward = th.tensor(
                            [self.mean_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                        self.twenty_five_quant_disc_reward = th.tensor(
                            [self.twenty_five_quant_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                        self.seventy_five_quant_disc_reward = th.tensor(
                            [self.seventy_five_quant_disc_reward],
                            dtype=th.float32,
                            device=self.device
                        )
                    if self.w_expectile_loss is True:
                        # We use 'none' reduction to allow discounting with Discriminator estimates
                        self.expectile_loss = ExpectileLoss(expectile=self.expectile_t, reduction="none")

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
        dem_rl_term_constraint_critic_loss_values = []
        td_term_constraint_critic_loss_values = []
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
            expert_done = None
            expert_reward = None
            expert_disc_reward = None
            expert_next_actions = None
            expert_next_observations = None
            if self.w_constraint_optimization is True:
                (
                    expert_actions,
                    expert_observations,
                    expert_done,
                    expert_reward,
                    expert_disc_reward,
                    expert_next_actions,
                    expert_next_observations
                ) = self.get_samples_from_demonstrations()

            # Get discriminator estimates
            critic_loss_weights = 1.
            if self.w_discriminator is True:
                # Get predictions of Discriminator for the rollout state-action pairs
                rollout_discr_preds = self.discriminator_to_use.predict(
                    replay_data.observations, replay_data.actions
                )
                critic_loss_weights = rollout_discr_preds
                # Also for the demonstrated state-action pairs, just for logs
                expert_discr_preds = self.discriminator_to_use.predict(
                    expert_observations, self.scale_and_clamp_demo_actions(expert_actions)
                )
                # Keep logs
                rollout_discr_preds_values.append(rollout_discr_preds.mean().item())
                expert_discr_preds_values.append(expert_discr_preds.mean().item())

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
                next_q_values = th.min(next_q_values, dim=1, keepdim=True)[0].detach()
                # add entropy term
                next_q_values_wo_entropy = next_q_values.clone()
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values_wo_entropy_wo_reward = (1 - replay_data.dones) * self.gamma * next_q_values_wo_entropy
                target_q_values_wo_reward = (1 - replay_data.dones) * self.gamma * next_q_values
                ## add reward or Discriminator's estimates
                if self.w_gail_sac is True:
                    # Add the Discriminator's estimates as reward signal
                    if self.discriminator_reward_function_in_gail_sac == 'GAIL':
                        discr_rew = -torch.log(1 - critic_loss_weights)
                    elif self.discriminator_reward_function_in_gail_sac == 'saturing_GANs_loss':
                        discr_rew = torch.log(critic_loss_weights)
                    elif self.discriminator_reward_function_in_gail_sac == 'AIRL':
                        discr_rew = torch.log(critic_loss_weights) - torch.log(1 - critic_loss_weights)
                    else:
                        raise ValueError(
                            "The selected 'discriminator_reward_function_in_gail_sac': "
                            f"{self.discriminator_reward_function_in_gail_sac} is not supported!"
                        )
                    target_q_values = discr_rew + target_q_values_wo_reward
                    # Make 'critic_loss_weights' ones again, in order to not weigh the critic loss
                    critic_loss_weights = 1.
                else:
                    # add reward
                    target_q_values = replay_data.rewards + target_q_values_wo_reward
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
            # Compute the critic loss coefficient
            critic_loss_coef = critic_loss_weights
            if self.w_threshold_in_discriminator_weights is True:
                critic_loss_coef = (critic_loss_weights >= self.threshold_in_discriminator_weights).float()
            # SAC critic loss
            critic_loss = 0.5 * sum(
                th.mean(
                    th.pow(current_q - target_q_values, 2) * critic_loss_coef
                ) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            # Constraint critic loss
            if self.w_q_values is True and self.w_gail_sac is False:
                (
                    constraint_critic_loss,
                    lower_bound_constraint_critic_loss,
                    dem_rl_term_constraint_critic_loss,
                    td_term_constraint_critic_loss,
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
                    [expert_actions, expert_observations, expert_done, expert_reward, expert_disc_reward, expert_next_actions, expert_next_observations],
                    [replay_data.actions, replay_data.observations, replay_data.dones, replay_data.rewards],
                    current_q_values,
                    target_q_values_wo_reward,
                    target_q_values_wo_entropy_wo_reward,
                    ent_coef,
                    critic_loss_weights,
                )
                # Keep it for logs
                critic_loss_values_wo_constraint_term.append(critic_loss.item())
                constraint_critic_loss_term_values.append(constraint_critic_loss.item())
                lower_bound_constraint_critic_loss_term_values.append(lower_bound_constraint_critic_loss.item())
                td_term_constraint_critic_loss_values.append(td_term_constraint_critic_loss.item())
                if dem_rl_term_constraint_critic_loss is not None:
                    dem_rl_term_constraint_critic_loss_values.append(dem_rl_term_constraint_critic_loss.item())
                # Add the constraint term to critic loss
                critic_loss_weight = 1.0
                if self.w_discriminator is False:
                    critic_loss_weight = self.constraint_lambda.item()
                critic_loss += critic_loss_weight * constraint_critic_loss
                if self.w_lower_bound is True:
                    critic_loss += critic_loss_weight * lower_bound_constraint_critic_loss
                if self.w_demonstrations_rl_term is True:
                    critic_loss += dem_rl_term_constraint_critic_loss
                if self.w_ood_rl_term is True:
                    critic_loss += td_term_constraint_critic_loss
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
                actor_loss += self.constraint_lambda.item() * constraint_policy_loss_term
                # Keep it for logs
                constraint_policy_loss_term_value = constraint_policy_loss_term.item()
                constraint_policy_loss_term_values.append(constraint_policy_loss_term_value)
                constraint_policy_loss_nll_term_values.append(constraint_policy_loss_nll_term_value)
                constraint_policy_loss_mse_term_values.append(constraint_policy_loss_mse_term_value)

            # Keep the logs for the constraint optimization
            if self.w_kl_div is True or (self.w_q_values is True and self.w_gail_sac is False):
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
                        [expert_actions, expert_observations, expert_done, expert_reward, expert_disc_reward, expert_next_actions, expert_next_observations],
                        [replay_data.actions, replay_data.observations, replay_data.dones, replay_data.rewards],
                        current_q_values,
                        target_q_values_wo_reward,
                        target_q_values_wo_entropy_wo_reward,
                        ent_coef
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

            # Update target Discriminator
            if self.w_target_discriminator is True and gradient_step % self.discriminator_target_update_interval == 0:
                polyak_update(self.discriminator.parameters(), self.discriminator_target.parameters(), self.discriminator_tau)
                polyak_update(self.discriminator_batch_norm_stats, self.discriminator_batch_norm_stats_target, 1.0)

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
        dem_rl_term_constraint_critic_loss_values_mean = np.nan
        td_term_constraint_critic_loss_values_mean = np.nan
        if self.w_constraint_optimization is True:
            if self.w_kl_div is True or (self.w_q_values is True and self.w_gail_sac is False):
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
                if self.w_gail_sac is False:
                    critic_loss_values_wo_constraint_term_mean = mean(critic_loss_values_wo_constraint_term)
                    constraint_critic_loss_term_values_mean = mean(constraint_critic_loss_term_values)
                    lower_bound_constraint_critic_loss_term_values_mean = mean(
                        lower_bound_constraint_critic_loss_term_values
                    )
                    td_term_constraint_critic_loss_values_mean = mean(td_term_constraint_critic_loss_values)
                if self.w_discriminator is True:
                    rollout_discr_preds_values_mean = mean(rollout_discr_preds_values)
                    expert_discr_preds_values_mean = mean(expert_discr_preds_values)
                if self.w_demonstrations_rl_term is True:
                    dem_rl_term_constraint_critic_loss_values_mean = mean(dem_rl_term_constraint_critic_loss_values)
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
            'dem_rl_term_constraint_critic_loss': dem_rl_term_constraint_critic_loss_values_mean,
            'td_term_constraint_critic_loss': td_term_constraint_critic_loss_values_mean
        }
        logs_dict.update(discriminator_train_logs)

        return logs_dict

    def calc_constraint_lambda_loss_w_kl_div(self, demonstrations):

        constraint_policy_loss_term, *_ = self.calc_constraint_policy_loss_term(demonstrations)
        constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * constraint_policy_loss_term.item()

        return constraint_lambda_loss

    def calc_constraint_lambda_loss_w_q_values(
            self,
            demonstrations,
            rollout_data,
            rollout_q_values,
            target_q_values_wo_reward,
            target_q_values_wo_entropy_wo_reward,
            ent_coef
    ):

        constraint_critic_loss, lower_bound_constraint_critic_loss, *_ = self.calc_constraint_q_loss_term(
            demonstrations,
            rollout_data,
            rollout_q_values,
            target_q_values_wo_reward,
            target_q_values_wo_entropy_wo_reward,
            ent_coef
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
            target_rollout_q_values_wo_reward: th.Tensor,
            target_rollout_q_values_wo_entropy_wo_reward: th.Tensor,
            ent_coef: th.Tensor,
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
                                          2) observations of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, *observation_space.shape],
                                          3) done of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, ],
                                          4) reward of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, ],
                                          5) discounted reward of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, ],
                                          6) next_actions of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, *action_space.shape],
                                          7) next_observations of type torch.Tensor with dtype=torch.float32
                                             and shape=[batch_size, *observation_space.shape],
        :param rollout_data: List similar to 'demonstrations' but for rollouts.
        :param rollout_q_values: Tuple with the Q-values of the rollout state-action pairs.
            Note that the tuple consists of two tensors, one for the estimated Q-values of each critic.
        :param target_rollout_q_values_wo_reward: Torch.Tensor with the targets without rewards for the TD loss, that is,
            the next Q-values of the rollout next_state-next_action pairs and the entropy term.
            Note that the tensor has a single batch dimension, that is, the min estimated Q-values of the critics is used.
        :param target_rollout_q_values_wo_entropy_wo_reward: Torch.Tensor similar to 'target_rollout_q_values_wo_reward',
            but also without the entropy term.
        :param ent_coef: Entropy coefficient.
        :param discriminator_rollout_preds: Predictions of the Discriminator for the rollout state-action pairs.
            Applicable only when self.w_discriminator is True.

        :return: Critic Q loss wrt constraints.
        """

        # Demonstrations' state-action pairs
        actions = demonstrations[0]
        observations = demonstrations[1]
        dones = demonstrations[2]
        rewards = demonstrations[3]
        disc_rewards = demonstrations[4]
        next_actions = demonstrations[5]
        next_observations = demonstrations[6]
        self.check_demonstrations_format(actions, observations, dones, rewards, disc_rewards, next_actions, next_observations)

        # Rollouts' state-action pairs
        rollout_actions = rollout_data[0]
        rollout_observations = rollout_data[1]
        rollout_dones = rollout_data[2]
        rollout_rewards = rollout_data[3]

        # Compute the coefficient of the critic loss
        critic_loss_coef = 1.
        if self.w_discriminator is True:
            critic_loss_coef = 1. - discriminator_rollout_preds
            if self.w_threshold_in_discriminator_weights is True:
                critic_loss_coef = (discriminator_rollout_preds < self.threshold_in_discriminator_weights).float()

        ## Compute the constraint critic loss and the corresponding TD term
        # Get the specified Q-value statistic of the demonstrated state-action pairs
        # to use it as the target (without grads)
        scaled_expert_actions = self.scale_and_clamp_demo_actions(actions)
        dem_q_values = self.critic(observations, scaled_expert_actions)
        dem_q_values_catted = th.cat(dem_q_values, dim=1)
        if self.w_closest_state_min is True:
            # Get the statistic Q-value, that is, different value as target for each rollout Q-value
            (
                _,
                _,
                closest_rewards,
                closest_dones,
                closest_next_states,
                closest_next_actions
            ) = self.expert_dataset.find_closest_states_batch(rollout_observations.detach().cpu().numpy())
            closest_rewards = th.tensor(closest_rewards[:, None], device=self.device, dtype=th.float32)
            closest_dones = th.tensor(closest_dones[:, None], device=self.device, dtype=th.float32)
            closest_next_states = th.tensor(closest_next_states, device=self.device, dtype=th.float32)
            closest_next_actions = th.tensor(closest_next_actions, device=self.device, dtype=th.float32)
            closest_next_actions = self.scale_and_clamp_demo_actions(closest_next_actions)
            with th.no_grad():  # To speed-up computations
                closest_next_q_values = th.min(
                    th.cat(self.critic_target(closest_next_states, closest_next_actions), dim=1),
                    dim=1,
                    keepdim=True
                )[0].detach()
            dem_q_value_stat = closest_rewards + (1 - closest_dones) * self.gamma * closest_next_q_values
        else:
            # Get the statistic Q-value, that is, a single value to use as target for all rollout Q-values
            dem_q_values_catted_target = th.cat(self.critic_target(observations, scaled_expert_actions), dim=1)
            if self.dem_q_value_stat == 'min':
                dem_q_value_stat = th.min(dem_q_values_catted).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.min(dem_q_values_catted_target).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.min_disc_reward
            elif self.dem_q_value_stat == 'max':
                dem_q_value_stat = th.max(dem_q_values_catted).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.max(dem_q_values_catted_target).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.max_disc_reward
            elif self.dem_q_value_stat == 'mean':
                dem_q_value_stat = th.mean(dem_q_values_catted).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.mean(dem_q_values_catted_target).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.mean_disc_reward
            elif self.dem_q_value_stat == 'median':
                dem_q_value_stat = th.median(dem_q_values_catted).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.median(dem_q_values_catted_target).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.median_disc_reward
            elif self.dem_q_value_stat == '25_quant':
                dem_q_value_stat = th.quantile(dem_q_values_catted, 0.25).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.quantile(dem_q_values_catted_target, 0.25).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.twenty_five_quant_disc_reward
            elif self.dem_q_value_stat == '75_quant':
                dem_q_value_stat = th.quantile(dem_q_values_catted, 0.75).detach()
                if self.w_use_target_critic is True:
                    dem_q_value_stat = th.quantile(dem_q_values_catted_target, 0.75).detach()
                elif self.w_compute_analytically_min_dem_q_value is True:
                    dem_q_value_stat = self.seventy_five_quant_disc_reward
            else:
                raise ValueError(f"The specified statistic is not supported: {self.dem_q_value_stat}")
        # Constraint critic loss
        if self.w_expectile_loss is True:
            constraint_critic_loss = 0.5 * (self.max_min_coef if self.w_max_min is True else 1.0) * sum(
                th.mean(
                    self.expectile_loss(
                        rollout_q, dem_q_value_stat
                    ) * critic_loss_coef
                ) for rollout_q in rollout_q_values
            )
        else:
            constraint_critic_loss = 0.5 * (self.max_min_coef if self.w_max_min is True else 1.0) * sum(
                th.mean(
                    th.pow(
                        (th.maximum(rollout_q, dem_q_value_stat) if self.w_max_min is True else rollout_q) - dem_q_value_stat,
                        2
                    ) * critic_loss_coef
                ) for rollout_q in rollout_q_values
            )
        ## TD term
        if self.w_entropy_in_ood_rl_term is True:
            target_rollout_q_values = target_rollout_q_values_wo_reward
        else:
            target_rollout_q_values = target_rollout_q_values_wo_entropy_wo_reward
        if self.w_discriminator_rewards_in_ood_rl_term is True:
            # Add the Discriminator's estimates as reward signal
            # TODO: When action is not provided to the Discriminator we should use the estimates for the next state
            # TODO: as reward. The way we do it now is like taking reward delayed by one step.
            if self.discriminator_reward_function_in_ood_rl_term == 'GAIL':
                target_rollout_q_values += -torch.log(1-discriminator_rollout_preds)
            elif self.discriminator_reward_function_in_ood_rl_term == 'saturing_GANs_loss':
                target_rollout_q_values += torch.log(discriminator_rollout_preds)
            elif self.discriminator_reward_function_in_ood_rl_term == 'AIRL':
                target_rollout_q_values += torch.log(discriminator_rollout_preds) - torch.log(1-discriminator_rollout_preds)
            else:
                raise ValueError(
                    "The selected 'discriminator_reward_function_in_ood_rl_term': "
                    f"{self.discriminator_reward_function_in_ood_rl_term} is not supported!"
                )
        elif self.w_discriminator_discounted_rewards_in_ood_rl_term is True:
            # Add the environment reward discounted by the Discriminator's estimates
            target_rollout_q_values += discriminator_rollout_preds*rollout_rewards
        td_term_constraint_critic_loss = 0.5 * self.ood_rl_term_coef * sum(
            th.mean(
                th.pow(
                    rollout_q - target_rollout_q_values,
                    2
                ) * critic_loss_coef
            ) for rollout_q in rollout_q_values
        )

        ## Compute the lower bound constraint critic loss
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

        ### Compute RL term for demonstrations of constraint critic loss
        dem_rl_term_constraint_critic_loss = None
        if self.w_demonstrations_rl_term is True:
            if self.w_compute_analytically_target_in_demonstrations_rl_term is True:
                # Add extra dimension to 'rewards' to match 'current_q_values' dimensionality
                target_q_values = disc_rewards[:, None]
            else:
                ## Compute the targets
                with th.no_grad():
                    # Select actions according to demonstrations
                    if self.w_demonstrations_next_actions_in_demonstrations_rl_term is True:
                        dem_rl_term_next_actions = self.scale_and_clamp_demo_actions(next_actions)
                        # Log probs
                        dem_rl_term_next_log_prob, _ = self.actor.evaluate_actions(
                            next_observations,
                            next_actions,
                            scale_actions=True,
                            adjust_entropy=False,
                            w_std_grads=False
                        )
                    # Select actions according to policy
                    else:
                        dem_rl_term_next_actions, dem_rl_term_next_log_prob = self.actor.action_log_prob(next_observations)
                    # Compute the next Q values: min over all critics' targets
                    next_q_values = th.cat(self.critic_target(next_observations, dem_rl_term_next_actions), dim=1)
                    next_q_values = th.min(next_q_values, dim=1, keepdim=True)[0].detach()
                    # add entropy term
                    if self.w_entropy_in_demonstrations_rl_term is True:
                        next_q_values -= ent_coef * dem_rl_term_next_log_prob.reshape(-1, 1)
                # td error + entropy term
                # Add extra dimension to 'rewards' and 'dones' to match 'next_q_values' dimensionality
                target_q_values = rewards[:, None] + (1 - dones[:, None]) * self.gamma * next_q_values
            ## Get current Q-values estimates for each critic network using actions from demonstrations
            current_q_values = self.critic(observations, scaled_expert_actions)
            ## Compute the loss
            dem_rl_term_constraint_critic_loss = 0.5 * self.demonstrations_rl_term_coef * sum(
                th.mean(
                    th.pow(current_q - target_q_values, 2)
                ) for current_q in current_q_values
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
        # Q-values of the policy actions for the demonstrations' observations
        q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
        min_q_values_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        mean_dem_qvals_value = th.mean(min_q_values_pi).item()
        min_dem_qvals_value = min_q_values_pi.min().item()
        max_dem_qvals_value = min_q_values_pi.max().item()

        return (
            constraint_critic_loss,
            lower_bound_constraint_critic_loss,
            dem_rl_term_constraint_critic_loss,
            td_term_constraint_critic_loss,
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

    def check_demonstrations_format(
            self,
            actions,
            observations,
            done=None,
            reward=None,
            disc_reward=None,
            next_actions=None,
            next_observations=None
    ):
        """
        Check if the provided demonstration samples have the right format.

        :param actions: actions of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, *action_space.shape].
        :param observations: observations of type torch.Tensor with dtype=torch.float32 and
            shape=[batch_size, *observation_space.shape].
        :param done: None or done of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, ].
        :param reward: None or done of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, ].
        :param disc_reward: None or done of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, ].
        :param next_actions: None or next actions of type torch.Tensor with dtype=torch.float32
            and shape=[batch_size, *action_space.shape].
        :param next_observations: None or next observations of type torch.Tensor with dtype=torch.float32 and
            shape=[batch_size, *observation_space.shape].

        :return:
        """

        assert actions.shape[1:] == self.action_space.shape and actions.dtype == torch.float32, \
            (
                    "actions.shape: " + str(actions.shape) +
                    "\nself.action_space.shape: " + str(self.action_space.shape) +
                    "\nactions.dtype: " + str(actions.dtype)
            )
        assert observations.shape[1:] == self.observation_space.shape and observations.dtype == torch.float32, \
            (
                    "observations.shape: " + str(observations.shape) +
                    "\nself.observation_space.shape: " + str(self.observation_space.shape) +
                    "\nobservations.dtype: " + str(observations.dtype)
            )
        if done is not None:
            assert len(done.shape) == 1 and done.dtype == torch.float32, \
                (
                        "done.shape: " + str(done.shape) +
                        "\ndone.dtype: " + str(done.dtype)
                )
        if reward is not None:
            assert len(reward.shape) == 1 and reward.dtype == torch.float32, \
                (
                        "reward.shape: " + str(reward.shape) +
                        "\nreward.dtype: " + str(reward.dtype)
                )
        if disc_reward is not None:
            assert len(disc_reward.shape) == 1 and disc_reward.dtype == torch.float32, \
                (
                        "disc_reward.shape: " + str(disc_reward.shape) +
                        "\ndisc_reward.dtype: " + str(disc_reward.dtype)
                )
        if next_actions is not None:
            assert next_actions.shape[1:] == self.action_space.shape and next_actions.dtype == torch.float32, \
                (
                        "next_actions.shape: " + str(next_actions.shape) +
                        "\nself.action_space.shape: " + str(self.action_space.shape) +
                        "\nnext_actions.dtype: " + str(next_actions.dtype)
                )
        if next_observations is not None:
            assert next_observations.shape[1:] == self.observation_space.shape and next_observations.dtype == torch.float32, \
                (
                        "next_observations.shape: " + str(next_observations.shape) +
                        "\nself.observation_space.shape: " + str(self.observation_space.shape) +
                        "\nnext_observations.dtype: " + str(next_observations.dtype)
                )

    def get_samples_from_demonstrations(self):
        (
            expert_actions,
            expert_observations,
            expert_done,
            expert_reward,
            expert_disc_reward,
            expert_next_actions,
            expert_next_observations
        ) = next(iter(self.expert_train_loader))

        return (
            expert_actions,
            expert_observations,
            expert_done,
            expert_reward,
            expert_disc_reward,
            expert_next_actions,
            expert_next_observations
        )

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

            for expert_actions, expert_observations, *_, in self.expert_train_loader:

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

        # Save 'constraint_lambda' variable / 'discriminator'
        if self.w_constraint_optimization is True:
            if self.w_discriminator is True:
                discriminator_path = os.path.join(path, f'{prefix}_sac_constraint_discriminator.pt')
                print('Saving {} to {} ...'.format('constraint discriminator', discriminator_path))
                self.discriminator.save(discriminator_path)
            else:
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
                self.log_ent_coef = th.load(log_ent_coef_path, device=self.device, weights_only=False)
            else:
                ent_coef_path = os.path.join(path, f'{prefix}_sac_ent_coef.pt')
                print('Loading {} from {} ...'.format('ent_coef', ent_coef_path))
                self.ent_coef_tensor = th.load(ent_coef_path, device=self.device, weights_only=False)

            # Load 'constraint_lambda' variable
            if self.w_constraint_optimization is True:
                if self.w_discriminator is True:
                    discriminator_path = os.path.join(path, f'{prefix}_sac_constraint_discriminator.pt')
                    print('Loading {} from {} ...'.format('constraint discriminator', discriminator_path))
                    self.discriminator.load_model(discriminator_path)
                else:
                    constraint_lambda_path = os.path.join(path, f'{prefix}_sac_constraint_lambda.pt')
                    print('Loading {} from {} ...'.format('constraint lambda', constraint_lambda_path))
                    self.constraint_lambda = th.load(constraint_lambda_path, device=self.device, weights_only=False)

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
