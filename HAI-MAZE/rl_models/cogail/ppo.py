"""
References:
    co-GAIL paper: https://arxiv.org/abs/2108.06038
    co-GAIL code: https://github.com/j96w/cogail
    PPO paper: https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

from rl_models.cogail.utils import update_linear_schedule

class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 device=torch.device("cpu"),
                 constr_ball_only_at_the_right_side_wrt_hole=False,
                 constr_ball_only_at_the_up_side_wrt_hole=False,
                 initial_lambda_constraint=1.05,
                 eps_constraint=0.2,
                 delta_constraint=0.05,
                 w_actor_critic_gradient_clipping=True,
                 rollout_storage=None):

        self.device = device

        self.actor_critic = actor_critic
        self.rollout_storage = rollout_storage

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.action_space = actor_critic.action_space
        self.half_action_space = actor_critic.half_action_space
        self.human_controls_axis = actor_critic.human_controls_axis
        self.opt_robot_w_env_rewards = actor_critic.opt_robot_w_env_rewards
        self.constr_ball_only_at_the_right_side_wrt_hole = constr_ball_only_at_the_right_side_wrt_hole
        self.constr_ball_only_at_the_up_side_wrt_hole = constr_ball_only_at_the_up_side_wrt_hole
        self.eps_constraint = eps_constraint
        self.delta_constraint = delta_constraint
        self.w_actor_critic_gradient_clipping = w_actor_critic_gradient_clipping

        self.update_linear_schedule = update_linear_schedule
        self.lr = lr
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.lr, eps=eps)
        self.CEloss = nn.CrossEntropyLoss().to(self.device)

        if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
            self.constraint_lambda = torch.tensor(initial_lambda_constraint, requires_grad=True, device=self.device)
            self.constraint_lambda_optimizer = optim.Adam([self.constraint_lambda], lr=self.lr, eps=eps)

    def save_model(self):
        # Save actor critic
        self.actor_critic.save_model(None if not self.opt_robot_w_env_rewards else self.rollout_storage.ret_rms)
        # Save constraint lambda
        if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
            torch.save(self.constraint_lambda, os.path.join(self.actor_critic.chkpt_dir, 'constraint_lambda_cogail.pt'))

    def pretrain(self, gail_train_loader):

        """
        Train actor with BC to get an initial policy distribution closer to that of demonstrators.
        This is also applied by the authors of GAIL to reduce exploration and achieve more stable
        training.
        """

        all_loss = []
        for expert_batch in gail_train_loader:
            expert_state, expert_action, _ = expert_batch
            expert_state = torch.from_numpy(expert_state).float().to(self.device)
            expert_action = expert_action.to(self.device)

            bs = len(expert_state)
            expert_seed = torch.from_numpy(np.array([[random.uniform(-1, 1),
                                                       random.uniform(-1, 1)] for _ in range(bs)])).float().view(bs, 2).to(self.device)
            logits = self.actor_critic.act(expert_state, expert_seed, deterministic=True, return_logits=True)[3]

            # In the original code of co-GAIL where continuous action space is considered, the MSE loss is used.
            # Here, since we use discrete action space, we use a CE loss for each agent.
            human_action_logits = logits[:, :self.half_action_space] if self.human_controls_axis == 'X' else \
                                  (logits[:, self.half_action_space:] if self.human_controls_axis == 'Y' else None)
            expert_human_actions = expert_action[:, 0] if self.human_controls_axis == 'X' else \
                                  (expert_action[:, 1] if self.human_controls_axis == 'Y' else None)
            human_action_CEloss = self.CEloss(human_action_logits, expert_human_actions)

            robot_action_logits = logits[:, self.half_action_space:] if self.human_controls_axis == 'X' else \
                                  (logits[:, :self.half_action_space] if self.human_controls_axis == 'Y' else None)
            expert_robot_actions = expert_action[:, 1] if self.human_controls_axis == 'X' else \
                                  (expert_action[:, 0] if self.human_controls_axis == 'Y' else None)
            robot_action_CEloss = self.CEloss(robot_action_logits, expert_robot_actions)

            loss = (human_action_CEloss+robot_action_CEloss) / 2.

            all_loss.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(all_loss)

    def update(self, rollouts, expert_loader):
        """
        Update Actor's, Critic's, and ψ network's parameters
        """

        self.actor_critic.train()

        advantages = rollouts.returns[:rollouts.step] - rollouts.value_preds[:rollouts.step]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        env_advantages = None
        if self.opt_robot_w_env_rewards:
            env_advantages = rollouts.env_returns[:rollouts.step] - rollouts.env_value_preds[:rollouts.step]
            env_advantages = (env_advantages - env_advantages.mean()) / (env_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        code_loss_epoch = 0
        inv_loss_epoch = 0
        actor_critic_total_grad_norm_clipped_value = 0
        discr_value_loss_epoch = 0
        env_value_loss_epoch = 0
        human_action_loss_epoch = 0
        robot_action_loss_epoch = 0
        robot_final_constraint_term_loss_epoch = 0
        constraint_lambda_loss_epoch = 0

        for e in tqdm(range(self.ppo_epoch), desc='Policy training epochs'):

            # Get samples from buffer. Note that the buffer contains samples only from the last run since PPO
            # is an on-policy algorithm.
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch, env_advantages=env_advantages)

            for expert_batch, sample in zip(expert_loader, data_generator):
                obs_batch, random_seed_batch, actions_batch, value_preds_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                    env_value_preds_batch, env_return_batch, env_adv_targ, next_obs_constr_batch = sample

                values, action_log_probs, dist_entropy, pred_code, pred_human_action_probs = \
                    self.actor_critic.evaluate_actions(obs_batch, random_seed_batch, actions_batch)

                # L_z loss. Equation (2) of co-GAIL paper. It is used to enforce the human policy to
                # take into account the latent strategy 'z'.
                # Intuitively, it maximizes the mutual information between the latent strategy and the human action.
                code_loss = torch.norm(pred_code - random_seed_batch, dim=1).mean() * 0.1

                ####################################################################################
                ## Computation of L_a loss
                expert_state, expert_action, expert_id = expert_batch

                expert_state = torch.from_numpy(expert_state).float().to(self.device)
                expert_action = expert_action.to(self.device)
                pred_codes = self.actor_critic.evaluate_code(expert_state, expert_action)
                pred_action_logits = self.actor_critic.act(expert_state, pred_codes,
                                                           deterministic=True, return_logits=True)[3]
                pred_human_action_logits = pred_action_logits[:, :self.half_action_space] \
                                           if self.human_controls_axis == 'X' else \
                                           (pred_action_logits[:, self.half_action_space:]
                                            if self.human_controls_axis == 'Y' else None)
                pred_human_action_probs = F.softmax(pred_human_action_logits, dim=1)

                # L_a loss. Equation (3) of co-GAIL paper. It is used to minimize the reconstruction error of
                # human-policy (π_co^H) actions (a^H) wrt to human-demonstrator actions.
                # Even though the training dataset consists of actions provided by couples of humans,
                # as 'human-demonstrator' we denote the human whose actions
                # correspond to and optimize the human-policy (recall that π_co consists of two policies which
                # co-evolve: robot-policy (π_co^R) and human-policy (π_co^H)).
                # Intuitively, this loss aims to make π_co^H to account for all demonstrated actions, even the
                # rare ones.
                # Note that in the original co-GAIL code L_a loss is calculated over the raw expert actions and
                # the predicted actions (mean, not sampled) because of the continuous action space.
                # However, since we use discrete action space, we should calculate L_a loss over the
                # one-hot encoded expert actions and the predicted probabilities of actions (i.e., the predicted
                # probabilities obtained after applying the softmax operation to the predicted logits)
                assert expert_action.size(1) == int(self.action_space / self.half_action_space)
                human_expert_action = expert_action[:, :int((self.action_space / self.half_action_space) / 2)] \
                                      if self.human_controls_axis == 'X' else \
                                      (expert_action[:, int((self.action_space / self.half_action_space) / 2):]
                                       if self.human_controls_axis == 'Y' else None)
                # Flatten human actions before encoding them
                one_hot_human_expert_action = F.one_hot(human_expert_action.view(-1), num_classes=self.half_action_space)

                inv_loss = torch.norm(one_hot_human_expert_action - pred_human_action_probs, dim=1).mean() * 0.1
                ####################################################################################

                ####################################################################################
                ## Computation of PPO loss.
                # Equation (7) of PPO paper, ie, Actor loss.
                action_loss = None
                if not self.opt_robot_w_env_rewards:
                    # log probabilities are summed together due to the properties of log.
                    # If we used probabilities, we should multiply them which would
                    # make the final probability to be zero due the loss of precision caused by the
                    # 64 bits used.
                    # This is one of the reasons why we use log probabilities instead of probabilities.
                    action_log_probs = action_log_probs.sum(dim=1, keepdim=True)
                    old_action_log_probs_batch = old_action_log_probs_batch.sum(dim=1, keepdim=True)

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()
                else:
                    # In case that we optimize robot policy wrt environment rewards, we need to calculate two action losses
                    human_action_log_probs = action_log_probs[:, :int((self.action_space / self.half_action_space) / 2)] \
                                             if self.human_controls_axis == 'X' else \
                                             (action_log_probs[:, int((self.action_space / self.half_action_space) / 2):]
                                              if self.human_controls_axis == 'Y' else None)
                    human_old_action_log_probs_batch = old_action_log_probs_batch[:, :int((self.action_space / self.half_action_space) / 2)] \
                                                       if self.human_controls_axis == 'X' else \
                                                       (old_action_log_probs_batch[:, int((self.action_space / self.half_action_space) / 2):]
                                                        if self.human_controls_axis == 'Y' else None)

                    human_ratio = torch.exp(human_action_log_probs - human_old_action_log_probs_batch)
                    human_surr1 = human_ratio * adv_targ
                    human_surr2 = torch.clamp(human_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                    human_action_loss = -torch.min(human_surr1, human_surr2).mean()

                    robot_action_log_probs = action_log_probs[:, int((self.action_space / self.half_action_space) / 2):] \
                                             if self.human_controls_axis == 'X' else \
                                             (action_log_probs[:, :int((self.action_space / self.half_action_space) / 2)]
                                              if self.human_controls_axis == 'Y' else None)
                    robot_old_action_log_probs_batch = old_action_log_probs_batch[:, int((self.action_space / self.half_action_space) / 2):] \
                                                       if self.human_controls_axis == 'X' else \
                                                       (old_action_log_probs_batch[:, :int((self.action_space / self.half_action_space) / 2)]
                                                        if self.human_controls_axis == 'Y' else None)

                    robot_ratio = torch.exp(robot_action_log_probs - robot_old_action_log_probs_batch)
                    robot_surr1 = robot_ratio * env_adv_targ
                    robot_surr2 = torch.clamp(robot_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * env_adv_targ
                    robot_action_loss = -torch.min(robot_surr1, robot_surr2).mean()

                    action_loss = human_action_loss + robot_action_loss

                if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
                    # In these cases, we optimize the robot policy wrt a constraint

                    if not self.opt_robot_w_env_rewards:
                        robot_action_log_probs = \
                            action_log_probs[:, int((self.action_space / self.half_action_space) / 2):] \
                            if self.human_controls_axis == 'X' else \
                            (action_log_probs[:, :int((self.action_space / self.half_action_space) / 2)]
                             if self.human_controls_axis == 'Y' else None)
                        robot_old_action_log_probs_batch = \
                            old_action_log_probs_batch[:, int((self.action_space / self.half_action_space) / 2):] \
                            if self.human_controls_axis == 'X' else \
                            (old_action_log_probs_batch[:, :int((self.action_space / self.half_action_space) / 2)]
                             if self.human_controls_axis == 'Y' else None)

                        robot_ratio = torch.exp(robot_action_log_probs - robot_old_action_log_probs_batch)

                    robot_constraint_term = robot_ratio * next_obs_constr_batch
                    robot_clipped_constraint_term = \
                        torch.clamp(robot_constraint_term,
                                    1 - self.eps_constraint, 1 + self.eps_constraint) * next_obs_constr_batch
                    robot_final_constraint_term = torch.min(robot_constraint_term, robot_clipped_constraint_term).mean()
                    robot_final_constraint_term_loss = -self.constraint_lambda * robot_final_constraint_term

                    if action_loss is not None:
                        action_loss += robot_final_constraint_term_loss
                    else:
                        action_loss = robot_final_constraint_term_loss

                # Critic loss. Note that in PPO paper the MSE loss is used whereas here, the clipped values are also
                # considered and then MSE is calculated over the maximum between the clipped and original calculated
                # values.
                # This change can be found in openai baselines as 'PPO2':
                # (https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L1)
                # A great explanation can be found here (https://github.com/openai/baselines/issues/91)
                # Intuitively, 'value' is clipped (as happens with the 'ratio') in order to guarantee that the 'value'
                # will fall into the 'trust region' after the update .
                if not self.opt_robot_w_env_rewards:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    # In case that we optimize robot policy wrt environment rewards, we need to calculate two values losses
                    discr_values = values[:, 0].unsqueeze(dim=1)

                    discr_value_pred_clipped = value_preds_batch + \
                                               (discr_values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    discr_value_losses = (discr_values - return_batch).pow(2)
                    discr_value_losses_clipped = (discr_value_pred_clipped - return_batch).pow(2)
                    discr_value_loss = 0.25 * torch.max(discr_value_losses, discr_value_losses_clipped).mean()

                    env_values = values[:, 1].unsqueeze(dim=1)
                    env_value_pred_clipped = env_value_preds_batch + \
                                             (env_values - env_value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    env_value_losses = (env_values - env_return_batch).pow(2)
                    env_value_losses_clipped = (env_value_pred_clipped - env_return_batch).pow(2)
                    env_value_loss = 0.25 * torch.max(env_value_losses, env_value_losses_clipped).mean()

                    value_loss = discr_value_loss + env_value_loss
                ####################################################################################

                ####################################################################################
                # Optimize wrt all loss (except lambda in case of constraints  optimizations)
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + # Critic loss
                 action_loss - dist_entropy * self.entropy_coef + # losses wrt Actor's parameters (π_co, ie, both π_co_H and π_co_R)
                 code_loss + inv_loss # ψ and π_co_H loss
                 ).backward()

                # Gradient clipping is applied in PPO2 as an improvement of PPO.
                actor_critic_grad_norm = torch.tensor([0.0]).to(self.device)
                if self.w_actor_critic_gradient_clipping:
                    actor_critic_grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                code_loss_epoch += code_loss.item()
                inv_loss_epoch += inv_loss.item()
                actor_critic_grad_norm_clipped_value = actor_critic_grad_norm.detach().cpu().numpy() - self.max_grad_norm \
                                                       if actor_critic_grad_norm.detach().cpu().numpy() > self.max_grad_norm \
                                                       else 0.0
                actor_critic_total_grad_norm_clipped_value += actor_critic_grad_norm_clipped_value
                if self.opt_robot_w_env_rewards:
                    discr_value_loss_epoch += discr_value_loss.item()
                    env_value_loss_epoch += env_value_loss.item()
                    human_action_loss_epoch += human_action_loss.item()
                    robot_action_loss_epoch += robot_action_loss.item()
                if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
                    robot_final_constraint_term_loss_epoch += robot_final_constraint_term.item()
                ####################################################################################

                if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
                    ####################################################################################
                    # Computation of 'constraint_lambda' loss.
                    action_log_probs_new = self.actor_critic.evaluate_actions(obs_batch, random_seed_batch, actions_batch)[1]

                    robot_action_log_probs_new = \
                        action_log_probs_new[:, int((self.action_space / self.half_action_space) / 2):] \
                            if self.human_controls_axis == 'X' else \
                            (action_log_probs_new[:, :int((self.action_space / self.half_action_space) / 2)]
                             if self.human_controls_axis == 'Y' else None)
                    robot_old_action_log_probs_batch = \
                        old_action_log_probs_batch[:, int((self.action_space / self.half_action_space) / 2):] \
                            if self.human_controls_axis == 'X' else \
                            (old_action_log_probs_batch[:, :int((self.action_space / self.half_action_space) / 2)]
                             if self.human_controls_axis == 'Y' else None)

                    robot_ratio_new = torch.exp(robot_action_log_probs_new - robot_old_action_log_probs_batch)

                    robot_constraint_term_new = robot_ratio_new * next_obs_constr_batch
                    robot_clipped_constraint_term_new = \
                        torch.clamp(robot_constraint_term_new,
                                    1 - self.eps_constraint, 1 + self.eps_constraint) * next_obs_constr_batch
                    robot_final_constraint_term_new = torch.min(robot_constraint_term_new, robot_clipped_constraint_term_new).mean()

                    constraint_lambda_loss = torch.log(self.constraint_lambda) * \
                                             torch.min(robot_final_constraint_term_new + (1. + self.delta_constraint),
                                                       torch.max(robot_final_constraint_term_new - (1. + self.delta_constraint),
                                                                 torch.tensor([0.0]).to(self.device))
                                                       ).mean()

                    # Optimization of loss wrt 'constraint_lambda'
                    self.constraint_lambda_optimizer.zero_grad()
                    constraint_lambda_loss.backward()
                    self.constraint_lambda_optimizer.step()

                    constraint_lambda_loss_epoch += constraint_lambda_loss.item()
                    ####################################################################################

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        code_loss_epoch /= num_updates
        inv_loss_epoch /= num_updates
        actor_critic_avg_grad_norm_clipped_value = actor_critic_total_grad_norm_clipped_value / num_updates
        if self.opt_robot_w_env_rewards:
            discr_value_loss_epoch /= num_updates
            env_value_loss_epoch /= num_updates
            human_action_loss_epoch /= num_updates
            robot_action_loss_epoch /= num_updates
        if self.constr_ball_only_at_the_right_side_wrt_hole or self.constr_ball_only_at_the_up_side_wrt_hole:
            robot_final_constraint_term_loss_epoch /= num_updates
            constraint_lambda_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
               code_loss_epoch, inv_loss_epoch, actor_critic_avg_grad_norm_clipped_value, \
               discr_value_loss_epoch, env_value_loss_epoch, human_action_loss_epoch, robot_action_loss_epoch, \
               robot_final_constraint_term_loss_epoch, constraint_lambda_loss_epoch, \
               np.nan if not (self.constr_ball_only_at_the_up_side_wrt_hole or self.constr_ball_only_at_the_right_side_wrt_hole)\
                      else self.constraint_lambda.detach().cpu().numpy().copy()
