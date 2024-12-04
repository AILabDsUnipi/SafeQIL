import torch
import numpy as np
import torch.nn.functional as F
import math
import os

from rl_models.sac_discrete.networks_discrete import update_params, Actor, Critic, ReplayBuffer
from rl_models.utils.expert_dataset import exprt_dataset


class DiscreteSACAgent:
    def __init__(
            self,
            config,
            input_dims=(8,),
            n_actions=2,
            chkpt_dir=None,
            axis_agent='X',
            only_test=False,
            device=torch.device("cpu"),
            transform_actions_to_action_func=None
    ):

        # Specifications
        self.device = device
        self.input_dims = input_dims[0]
        self.chkpt_dir = chkpt_dir
        self.axis_agent = axis_agent
        self.n_actions = n_actions
        self.constr_ball_only_at_the_right_side_wrt_hole = \
            config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']
        self.constr_ball_only_at_the_up_side_wrt_hole = \
            config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']
        self.constr_ball_not_in_circle = \
            config['Experiment']['constraint_ball_not_in_circle']
        self.only_test = only_test

        # SAC params
        self.batch_size = config['SAC']['batch_size']
        self.layer1_size = config['SAC']['layer1_size']
        self.layer2_size = config['SAC']['layer2_size']
        if self.only_test is False:
            self.gamma = config['SAC']['gamma']
            self.tau = config['SAC']['tau']
            self.alpha = config['SAC']['alpha']
            self.beta = config['SAC']['beta']
            self.buffer_max_size = config['SAC']['buffer_memory_size']
            self.lr = config['SAC']['entropy_coefficient_lr']
            self.clip_grad_norm = config['SAC']['clip_grad_norm']
            self.max_grad_norm = config['SAC']['max_grad_norm']
            # target -> maximum entropy (same prob for each action)
            # - log ( 1 / A) = log A
            # target_entropy = -log(1.0 / n_actions) * target_entropy_ratio
            #                = log(n_actions) * target_entropy_ratio
            self.target_entropy_ratio = config['SAC']['target_entropy_ratio']
            self.target_entropy = np.log(n_actions) * self.target_entropy_ratio
            self.SL_finetune = config['SAC']['SL_finetuning']
            self.SL_finetuning_batch_size = config['SAC']['SL_finetuning_batch_size']
            self.SL_finetuning_lr = config['SAC']['SL_finetuning_lr']
            self.w_constraint_optimization = config['SAC']['w_constraint_optimization']

        self.actor = Actor(
            self.input_dims,
            self.n_actions,
            self.layer1_size,
            self.layer2_size,
            name='actor_' + self.axis_agent,
            chkpt_dir=self.chkpt_dir,
            device=self.device
        ).to(self.device)
        self.critic = Critic(
            self.input_dims,
            self.n_actions,
            self.layer1_size,
            self.layer2_size,
            name='critic_' + self.axis_agent,
            chkpt_dir=self.chkpt_dir
        ).to(self.device)
        self.target_critic = Critic(
            self.input_dims,
            self.n_actions,
            self.layer1_size,
            self.layer2_size,
            name='target_critic_' + self.axis_agent,
            chkpt_dir=self.chkpt_dir
        ).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        if self.axis_agent == 'X_Y':
            self.transform_actions_to_action_func = transform_actions_to_action_func

        if self.only_test is False:
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(),
                lr=self.alpha if not self.SL_finetune else self.SL_finetuning_lr,
                eps=1e-4
            )
            self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=self.beta, eps=1e-4)
            self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=self.beta, eps=1e-4)

            # Entropy coefficient
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)

            self.memory = ReplayBuffer(self.buffer_max_size)

            if self.w_constraint_optimization is True:
                self.initial_lambda_constraint = config['SAC']['initial_lambda_constraint']
                self.constraint_lambda_lr = config['SAC']['lambda_constraint_lr']
                self.w_entropy_in_constraint_policy_loss_term = config['SAC']['w_entropy_in_constraint_policy_loss_term']
                self.w_dual_grad_desc = config['SAC']['w_dual_grad_desc']
                # Define 'constraint_lambda'
                self.constraint_lambda = torch.tensor(
                    [self.initial_lambda_constraint],
                    dtype=torch.float32,
                    requires_grad=self.w_dual_grad_desc,
                    device=self.device
                )
                # Define optimizer for 'constraint_lambda'
                self.constraint_lambda_optimizer = torch.optim.Adam(
                    [self.constraint_lambda],
                    lr=self.constraint_lambda_lr,
                    eps=1e-4
                ) if self.w_dual_grad_desc is True else None
                # Define torch dataset for demonstrations
                expert_dataset = exprt_dataset(config['SAC']['expert_dataset_paths'])
                # Define torch loader based on torch dataset for training the policy wrt the constraints
                self.drop_last = len(expert_dataset) > (self.batch_size if not self.SL_finetune else self.SL_finetuning_batch_size)
                # When finetuning with SL, batch_size should not be lower than the number of samples provided
                assert not (self.SL_finetune is True and self.drop_last is True)
                self.train_loader = torch.utils.data.DataLoader(
                    dataset=expert_dataset,
                    batch_size=self.batch_size if not self.SL_finetune else self.SL_finetuning_batch_size,
                    shuffle=True,
                    drop_last=self.drop_last
                )

        else:
            self.actor.eval()
            self.critic.eval()
            self.target_critic.eval()

    def learn(self, interaction=None, soft_update_target=False):

        ##### Samples from agent-environment interactions ######
        if interaction is None:
            states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, states_, dones = interaction
            states, actions, rewards, states_, dones = [
                np.asarray([states]),
                np.asarray([actions]),
                np.asarray([rewards]),
                np.asarray([states_]),
                np.asarray([dones])
            ]

        states = torch.from_numpy(states).float().to(self.device)
        states_ = torch.from_numpy(states_).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        batch_transitions = states, actions, rewards, states_, dones
        ##############################################################

        ###### Samples from demonstrations ######
        expert_state = None
        expert_action = None
        if self.w_constraint_optimization is True:
            expert_state, expert_action = self.get_samples_from_demonstrations()
        ##############################################################

        ### Computation of SAC losses and optimizers' step
        weights = 1.  # default

        # Critics losses
        q1_loss, q2_loss = self.calc_critic_loss(batch_transitions)

        # Policy loss
        policy_loss, entropies = self.calc_policy_loss(batch_transitions, weights)

        ### Policy loss wrt constraints
        constraint_policy_loss_term_value = None
        policy_loss_value_wo_constraint_term = None
        if self.w_constraint_optimization is True:
            ## Calculate policy loss wrt constraints
            constraint_policy_loss_term = self.calc_constraint_policy_loss_term([expert_action, expert_state])
            # Keep it for logs
            policy_loss_value_wo_constraint_term = policy_loss.clone().detach().cpu().numpy().copy()

            # Add constraint term loss to policy loss
            policy_loss = policy_loss + self.constraint_lambda.item() * constraint_policy_loss_term

            # Keep it for logs
            constraint_policy_loss_term_value = constraint_policy_loss_term.clone().detach().cpu().numpy()

        # Entropy loss
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        # Optimizers step
        q1_grad_norm_clipped_value = update_params(
            self.critic_q1_optim,
            q1_loss,
            clip_grad_norm=self.clip_grad_norm,
            max_grad_norm=self.max_grad_norm,
            params=self.critic.qnet1.parameters()
        )
        q2_grad_norm_clipped_value = update_params(
            self.critic_q2_optim,
            q2_loss,
            clip_grad_norm=self.clip_grad_norm,
            max_grad_norm=self.max_grad_norm,
            params=self.critic.qnet2.parameters()
        )

        actor_grad_norm_clipped_value = update_params(
            self.actor_optim,
            policy_loss,
            clip_grad_norm=self.clip_grad_norm,
            max_grad_norm=self.max_grad_norm,
            params=self.actor.parameters()
        )

        update_params(self.alpha_optim, entropy_loss)

        ## Computation of 'constraint_lambda' loss and optimizer step
        constraint_lambda_loss_value = None
        if self.w_constraint_optimization is True:
            # Calculate lambda loss
            constraint_lambda_loss = self.calc_constraint_lambda_loss([expert_action, expert_state])
            # Keep it for logs
            constraint_lambda_loss_value = constraint_lambda_loss.clone().detach().cpu().numpy()
            if self.w_dual_grad_desc is True:
                # 'lambda' optimizer step
                update_params(self.constraint_lambda_optimizer, constraint_lambda_loss, retain_graph=False)

        # Update target networks
        if soft_update_target:
            self.soft_update_target()

        return (
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            entropies.mean().detach().cpu().numpy(),
            entropy_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
            self.log_alpha.exp().detach().cpu().numpy().copy(),
            q1_grad_norm_clipped_value,
            q2_grad_norm_clipped_value,
            actor_grad_norm_clipped_value,
            constraint_policy_loss_term_value,
            constraint_lambda_loss_value,
            policy_loss_value_wo_constraint_term,
            np.nan if self.w_constraint_optimization is False
                   else
            self.constraint_lambda.detach().cpu().numpy().copy()
        )

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to the chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():

            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8 # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)

            alpha = self.log_alpha.exp()
            next_q = action_probs * (torch.min(next_q1, next_q2) - alpha * log_action_probs)
            next_q = next_q.sum(dim=1)

            target_q = rewards + (1 - dones) * self.gamma * next_q
            return target_q.unsqueeze(1)

    def calc_critic_loss(self, batch):
        target_q = self.calc_target_q(*batch)

        curr_q1, curr_q2 = self.calc_current_q(*batch)

        # Get total critics loss
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # Q for every action to calculate expectations of Q.
        q1, q2 = self.critic(states)

        alpha = self.log_alpha.exp().clone().detach()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2).detach() * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies

    def calc_constraint_policy_loss_term(self, demonstrations):
        """
        Calculates policy loss wrt the specified constraints

        :param demonstrations: List with: 1) actions torch.Tensor with dtype=torch.int64 and shape=[batch_size, 1]
                                      and 2) observations torch.FloatTensor.
        :return: Policy loss wrt constraints.
        """

        actions = demonstrations[0]
        states = demonstrations[1]

        assert len(actions.size()) == 2 and actions.size()[1] == 1, \
            "len(actions.size()): " + str(len(actions.size())) + " actions.size()[1]: " + str(actions.size()[1])

        # (Log of) probabilities
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_all_action_probs = torch.log(action_probs + z)

        # select the 'log_action_probs' corresponding to the chosen 'actions'
        log_action_probs = log_all_action_probs.gather(1, actions)

        # Negative log-likelihood
        negative_log_action_probs = -log_action_probs

        if self.w_entropy_in_constraint_policy_loss_term is True:
            # Calculate entropy
            entropies = - torch.sum(action_probs * log_all_action_probs, dim=1)
            alpha = self.log_alpha.exp().item()
            # Calculate final loss
            constraint_policy_loss_term = negative_log_action_probs.mean() - (alpha*entropies.mean())
        else:
            constraint_policy_loss_term = negative_log_action_probs.mean()

        return constraint_policy_loss_term

    def calc_constraint_lambda_loss(self, demonstrations):
        constraint_policy_loss_term = self.calc_constraint_policy_loss_term(demonstrations)

        constraint_lambda_loss = -self.constraint_lambda.squeeze(0) * constraint_policy_loss_term.item()
        return constraint_lambda_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increase alpha when entropy is less than target
        # entropy, and vice versa.

        entropy_loss = \
            -torch.mean(self.log_alpha * (self.target_entropy - entropies).detach() * weights)
        return entropy_loss

    def get_samples_from_demonstrations(self):
        assert self.w_constraint_optimization is True

        expert_state, expert_action, _ = next(iter(self.train_loader))
        expert_state = expert_state.to(self.device)

        # Get expert action in the right format
        if self.axis_agent == 'X_Y':
            # Transform actions (from two actions-indices to one action-index)
            expert_action = self.transform_actions_to_action_func(expert_action, math.sqrt(self.n_actions)).to(self.device)
        elif self.axis_agent == 'X':
            # Keep only the action of X-axis
            expert_action = expert_action[:, 0].unsqueeze(dim=1).to(self.device)
        elif self.axis_agent == 'Y':
            # Keep only the action of Y-axis
            expert_action = expert_action[:, 1].unsqueeze(dim=1).to(self.device)
        else:
            raise NotImplementedError

        return expert_state, expert_action

    def SL_finetuning(self):
        assert not self.drop_last, "Batch size is smaller that the number of samples."

        # Get samples
        expert_states, expert_actions = self.get_samples_from_demonstrations()
        # Get model outputs and calculate loss
        constraint_policy_loss_term = self.calc_constraint_policy_loss_term([expert_actions, expert_states])
        # Optimizer step
        actor_grad_norm_clipped_value = update_params(
            self.actor_optim,
            constraint_policy_loss_term,
            retain_graph=False,
            clip_grad_norm=self.clip_grad_norm,
            max_grad_norm=self.max_grad_norm,
            params=self.actor.parameters()
        )

        return (
            constraint_policy_loss_term.item(),
            actor_grad_norm_clipped_value
                if self.clip_grad_norm is False
                else
            actor_grad_norm_clipped_value.item())

    def save_models(self, override=True):

        new_chkpt_dir = self.chkpt_dir
        checkpoint_dir_suffix = ''

        # If all models should be saved, the corresponding directory should be created
        if override is False:
            checkpoint_dir_suffix = 0
            while os.path.exists(os.path.join(self.chkpt_dir, str(checkpoint_dir_suffix))):
                checkpoint_dir_suffix += 1
            new_chkpt_dir = os.path.join(self.chkpt_dir, str(checkpoint_dir_suffix))
            os.mkdir(new_chkpt_dir)

        # Save Actors and Critics
        self.actor.save_checkpoint(override=override, checkpoint_dir_suffix=checkpoint_dir_suffix)
        self.critic.save_checkpoint(override=override, checkpoint_dir_suffix=checkpoint_dir_suffix)
        self.target_critic.save_checkpoint(override=override, checkpoint_dir_suffix=checkpoint_dir_suffix)

        # Save 'log_alpha' variable
        log_alpha_var_name = self.axis_agent+'_sac_log_alpha.pt'
        log_alpha_checkpoint_file = os.path.join(new_chkpt_dir, log_alpha_var_name)
        print('Saving {} to {} ...'.format(log_alpha_var_name, log_alpha_checkpoint_file))
        torch.save(self.log_alpha, log_alpha_checkpoint_file)

        # Save 'constraint_lambda' variable
        if self.w_constraint_optimization is True:
            constraint_lambda_var_name = self.axis_agent + '_sac_constraint_lambda.pt'
            constraint_lambda_checkpoint_file = os.path.join(new_chkpt_dir, constraint_lambda_var_name)
            print('Saving {} to {} ...'.format(constraint_lambda_var_name, constraint_lambda_checkpoint_file))
            torch.save(self.constraint_lambda, constraint_lambda_checkpoint_file)

    def load_models(self, load_checkpoint_path_name):
        self.actor.load_checkpoint(
            os.path.join(load_checkpoint_path_name, 'actor_' + self.axis_agent + '_sac')
        )
        self.critic.load_checkpoint(
            os.path.join(load_checkpoint_path_name, 'critic_' + self.axis_agent + '_sac')
        )
        self.target_critic.load_checkpoint(
            os.path.join(load_checkpoint_path_name, 'target_critic_' + self.axis_agent + '_sac')
        )
        if self.only_test is False:
            print(
                'Loading {} from {} ...'.format(
                    self.axis_agent + '_sac_log_alpha.pt',
                    os.path.join(load_checkpoint_path_name, self.axis_agent + '_sac_log_alpha.pt')
                )
            )
            self.log_alpha = torch.load(
                os.path.join(load_checkpoint_path_name, self.axis_agent + '_sac_log_alpha.pt'),
                map_location=self.device
            )
            if self.w_constraint_optimization is True:
                print(
                    'Loading {} from {} ...'.format(
                        self.axis_agent + '_sac_constraint_lambda.pt',
                        os.path.join(load_checkpoint_path_name, self.axis_agent + '_sac_constraint_lambda.pt')
                    )
                )
                self.constraint_lambda = torch.load(
                    os.path.join(load_checkpoint_path_name, self.axis_agent + '_sac_constraint_lambda.pt'),
                    map_location=self.device
                )
                # Define again optimizer for 'constraint_lambda', otherwise it will not be updated during finetuning
                self.constraint_lambda_optimizer = torch.optim.Adam(
                    [self.constraint_lambda],
                    lr=self.constraint_lambda_lr,
                    eps=1e-4
                )

