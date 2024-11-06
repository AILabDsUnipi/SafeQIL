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
            self.adjust_entropy = self.config['SAC']['adjust_entropy']
            self.pretrain_epochs = self.config['SAC']['pretrain_epochs']
            self.pretrain_mse_factor = self.config['SAC']['pretrain_mse_factor']
            self.pretrain_nll_factor = self.config['SAC']['pretrain_nll_factor']
            # Buffer placeholder
            self.replay_buffer: Optional[ReplayBuffer] = None
        # Define policy keyword arguments
        self.use_sde: bool = self.config['SAC']['use_sde']
        self.policy_kwargs = {
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
                self.w_entropy_in_constraint_policy_loss_term = \
                    self.config['SAC']['w_entropy_in_constraint_policy_loss_term']
                self.w_dual_grad_desc = self.config['SAC']['w_dual_grad_desc']
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
                self.drop_last = len(expert_dataset) > self.batch_size
                self.train_loader = th.utils.data.DataLoader(
                    dataset=expert_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=self.drop_last
                )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Basic SAC logs
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        grad_norms_clipped = []

        # Logs for constraint optimization
        constraint_policy_loss_term_values = []
        constraint_lambda_loss_values = []
        policy_loss_value_wo_constraint_terms = []
        constraint_lambdas = []

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

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
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

            ### Actor loss wrt constraints
            if self.w_constraint_optimization is True:
                # Keep actor loss without the constraint term for logs
                policy_loss_value_wo_constraint_term = actor_loss.item()
                policy_loss_value_wo_constraint_terms.append(policy_loss_value_wo_constraint_term)
                ## Calculate policy loss wrt constraints
                constraint_policy_loss_term = self.calc_constraint_policy_loss_term([
                    expert_actions, expert_observations
                ])
                # Add constraint term loss to policy loss
                actor_loss = actor_loss + self.constraint_lambda.item() * constraint_policy_loss_term
                # Keep it for logs
                constraint_policy_loss_term_value = constraint_policy_loss_term.item()
                constraint_policy_loss_term_values.append(constraint_policy_loss_term_value)

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
            if self.w_constraint_optimization is True:
                # Calculate lambda loss
                constraint_lambda_loss = self.calc_constraint_lambda_loss([expert_actions, expert_observations])
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
        constraint_lambda_loss_values_mean = np.nan
        policy_loss_value_wo_constraint_terms_mean = np.nan
        constraint_lambdas_mean = np.nan
        grad_norms_clipped_mean = np.nan
        if self.w_constraint_optimization is True:
            constraint_policy_loss_term_values_mean = mean(constraint_policy_loss_term_values)
            constraint_lambda_loss_values_mean = mean(constraint_lambda_loss_values)
            policy_loss_value_wo_constraint_terms_mean = mean(policy_loss_value_wo_constraint_terms)
            constraint_lambdas_mean = mean(constraint_lambdas)
        if self.clip_grad_norm is True:
            grad_norms_clipped_mean = mean(grad_norms_clipped)

        return (
            mean(actor_losses),
            mean(critic_losses),
            np.nan if self.ent_coef_optimizer is None else mean(ent_coef_losses),
            mean(ent_coefs),
            constraint_policy_loss_term_values_mean,
            constraint_lambda_loss_values_mean,
            policy_loss_value_wo_constraint_terms_mean,
            constraint_lambdas_mean,
            grad_norms_clipped_mean
        )

    def calc_constraint_lambda_loss(self, demonstrations):
        constraint_policy_loss_term = self.calc_constraint_policy_loss_term(demonstrations)
        constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * constraint_policy_loss_term.item()

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

        # Log probabilities and entropy of the given actions
        log_probs, actions_entropy = self.actor.evaluate_actions(
            observations,
            actions,
            scale_actions=True,
            adjust_entropy=self.adjust_entropy
        )
        if self.w_entropy_in_constraint_policy_loss_term is True:
            if self.log_ent_coef is not None:
                ent_coef = self.log_ent_coef.exp().item()
            else:
                ent_coef = self.ent_coef_tensor.item()
            # Calculate final loss
            constraint_policy_loss_term = -th.mean(log_probs) - (ent_coef * th.mean(actions_entropy))
        else:
            constraint_policy_loss_term = -th.mean(log_probs)

        return constraint_policy_loss_term

    def get_samples_from_demonstrations(self):
        expert_actions, expert_observations = next(iter(self.train_loader))

        return expert_actions, expert_observations

    def pretrain(self) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:

        # Initialize vars for logging
        mse_losses = []
        nll_losses = []
        losses = []
        grad_norms_clipped = []
        log_probs = []
        probs = []

        # Define the MSE loss function
        mse_loss_func = th.nn.MSELoss()

        for epoch in range(self.pretrain_epochs):
            print(f"\nPretraining epoch: {epoch}")

            # Initialize epoch vars for logging
            epoch_mse_losses = []
            epoch_nll_losses = []
            epoch_losses = []
            epoch_grad_norms_clipped = []
            epoch_log_probs = []
            epoch_probs = []

            for expert_actions, expert_observations in self.train_loader:

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
                    adjust_entropy=self.adjust_entropy
                )

                # Compute loss
                scaled_expert_actions = self.scale_and_clamp_demo_actions(expert_actions)
                mse_loss = mse_loss_func(actions_pi, scaled_expert_actions)
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

        return mse_losses, nll_losses, losses, log_probs, probs, grad_norms_clipped

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
