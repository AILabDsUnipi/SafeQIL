import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd
import torch.nn.functional as F
import pickle
import os

from rl_models.cogail.utils import RunningMeanStd


class DiscNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiscNet, self).__init__()

        # Use Tanh activations for the first two layers and linear for the last layer.
        # This is suggested when the BCE loss is used (suitable for low dimension controllers as in our case),
        # as shown here: https://github.com/j96w/cogail/blob/main/a2c_ppo_acktr/algo/gail.py
        # and here: https://github.com/j96w/cogail/blob/main/configs/exp1_config.py
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, state_only_input_dim, hidden_dim, device, chkpt_dir,  human_controls_axis='Y',
                 opt_robot_w_env_rewards=False, only_test=False, expert_torch_dataset=None, batch_size=128):
        super(Discriminator, self).__init__()

        self.device = device
        self.opt_robot_w_env_rewards = opt_robot_w_env_rewards
        self.human_controls_axis = human_controls_axis
        self.state_only_input_dim = state_only_input_dim
        self.only_test = only_test

        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'discriminator_cogail.pt')

        # The input to Discriminator is state+human_action+robot_action. For the actions, we use the selected action-class
        # instead of the action probs (or the one-hot representation of human experts action) because will be very easy
        # for Discriminator to distinguish probabilities from one-hot representations.
        self.trunk = DiscNet(input_dim, hidden_dim).to(self.device)
        if not self.only_test:
            self.optimizer = torch.optim.Adam(self.trunk.parameters())
            self.crit = nn.MSELoss()
            # Define torch loader for training the Discriminator
            drop_last = len(expert_torch_dataset) > batch_size
            self.train_loader = torch.utils.data.DataLoader(dataset=expert_torch_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True, drop_last=drop_last)
        else:
            self.trunk.eval()

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self, expert_state, expert_action, policy_state, policy_action, lambda_=10):

        """
        This gradient penalty is used as an improvement of Wasserstein GANs.
        Intuitively, this technique yields more stable training by replacing
        weight clipping with penalization of the norm of gradient of the critic
        with respect to its input. Specifically, it enforces the gradients of
        the critic’s output w.r.t the inputs to have unit norm.
        paper reference: https://arxiv.org/abs/1704.00028
        find a great explanation here:
        https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
        """

        # Sample 'alpha' from uniform distribution
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        # Interpolation between real data and fake data.
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, rollouts):
        self.train()

        policy_data_generator = \
            rollouts.feed_forward_generator(None, mini_batch_size=int(self.train_loader.batch_size))

        BCE_loss = 0
        grad_pen_loss = 0
        n = 0

        for expert_batch, policy_batch in zip(self.train_loader, policy_data_generator):

            policy_state, policy_action = policy_batch[0], policy_batch[2]
            expert_state, expert_action, _ = expert_batch

            assert policy_state.size()[1] == expert_state.size()[1] == self.state_only_input_dim[0]

            if self.opt_robot_w_env_rewards:
                # In case that we optimize robot policy wrt environments rewards,
                # we feed Discriminator only with human part of the action

                assert len(policy_action.size()) == len(expert_action.size()) == 2 and \
                       policy_action.size()[1] == expert_action.size()[1]
                half_action_size = int(policy_action.size()[1] / 2)

                policy_action = policy_action[:, :half_action_size] if self.human_controls_axis == 'X' else \
                                (policy_action[:, half_action_size:] if self.human_controls_axis == 'Y' else None)

                expert_action = expert_action[:, :half_action_size] if self.human_controls_axis == 'X' else \
                                (expert_action[:, half_action_size:] if self.human_controls_axis == 'Y' else None)

            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

            expert_state = torch.from_numpy(expert_state).float().to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            # Use BCE loss which works better for low dimension controllers,
            # as mentioned here: https://github.com/j96w/cogail/blob/main/configs/exp1_config.py .
            # Combining the knowledge from co-GAIL paper and the corresponding code,
            # it seems that the authors used LSGAN loss only for high-dimensional manipulation tasks.
            # In our case, the dimensionality is low.

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = (expert_loss + policy_loss) / 2.0

            # This gradient penalty is not referred in co-GAIL paper.
            grad_pen = self.compute_grad_pen(expert_state, expert_action, policy_state, policy_action)

            BCE_loss += gail_loss.item()
            grad_pen_loss += grad_pen.item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return BCE_loss / n, grad_pen_loss / n

    def predict_reward(self, state, action, gamma, masks):
        with torch.no_grad():
            self.eval()

            assert len(state.size()) == 2 and state.size()[1] == self.state_only_input_dim[0] and \
                   len(action.size()) == 2 and len(masks.size()) == 2

            if self.opt_robot_w_env_rewards:
                # In case that we optimize robot policy wrt environments rewards,
                # we feed Discriminator only with human part of the action

                half_action_size = int(action.size()[1] / 2)

                action = action[:, :half_action_size] if self.human_controls_axis == 'X' else \
                         (action[:, half_action_size:] if self.human_controls_axis == 'Y' else None)

            first_time = False
            if self.returns is None:
                first_time = True
                print('\nGetting Discriminator rewards for the first time!! '
                      '\nInstead of the running variance, the first reward will be divided by 1.0!!\n')

            # Reward when using BCE loss for Discriminator.
            # Note that in the original paper of GAIL (https://arxiv.org/pdf/1606.03476.pdf)
            # the reward used is R = -log(D).
            # Here, the reward used is an improvement suggested by
            # the authors of paper "LEARNING ROBUST REWARDS WITH ADVERSARIAL INVERSE REINFORCEMENT LEARNING"
            # (https://openreview.net/forum?id=rkHywl-A-) where the goal is to learn a reward function which
            # is disentangled from the environment dynamics. In this way, the reward describes better the
            # objective of the demonstrations (i.e., gives better answer on the question:
            # what do the demonstrators want to achieve?) instead of focusing on how the environment state changes
            # due to the applied action. Additionally, in the context of multi-agent setting, this could help the
            # agent policies to converge faster. That is, it can mitigate the problem of non-stationarity.

            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()

            np_reward = reward.detach().cpu().numpy()
            np_masks = masks.detach().cpu().numpy()

            self.returns = np.array([[0.0]], dtype=np.float64)
            ret_rms_var = []

            # Iterate over rewards and compute the running average and variation of discounted rewards (over the opposite directions),
            # in order to calculate them properly as in the original code of co-GAIL
            # (i.e., compute them based only on the previous rewards
            # of each step and not based on the entire batch rewards).

            for rew in range(np_reward.shape[0]):
                mask_ = np.expand_dims(np_masks[rew], axis=1)
                reward_ = np.expand_dims(np_reward[rew], axis=1)

                self.returns = self.returns * mask_ * gamma + reward_
                self.ret_rms.update(self.returns)
                # At the first time that the running variance is computed it is a very low float (about 1e-4)
                # and therefore its square root is a lower float. For this reason, we use 1.0 to divide the first
                # discriminator's reward, otherwise a very high reward is produced (about 90).
                ret_rms_var.append(self.ret_rms.var[0] if not (first_time and rew == 0) else 1.0)

            # This is an alternative way to compute the rewards. It probably aims to properly normalize the rewards
            # in order to reduce its variance and therefore the variance of policy.
            # It can be found to be used only here:
            # https://github.com/DLR-RM/stable-baselines3/blob/12e9917c24dc23d7de7694a924f017c6a8e9a6ce/stable_baselines3/common/buffers.py#L294
            # but not for PPO nor for GAIL, only for offline algorithms (like, DQN, DDPG etc).
            # Its usage is not referenced in the paper of co-GAIL.
            # print()
            # print('train')
            # print(reward)
            # print(torch.FloatTensor(np.sqrt(np.expand_dims(np.array(ret_rms_var), axis=1) + 1e-8)).to(self.device))
            # print(reward / torch.FloatTensor(np.sqrt(np.expand_dims(np.array(ret_rms_var), axis=1) + 1e-8)).to(self.device))
            return reward / torch.from_numpy(np.sqrt(np.expand_dims(np.array(ret_rms_var), axis=1) + 1e-8)).float().to(self.device)

    def predict_reward_test(self, state, action):

        with torch.no_grad():
            self.eval()

            assert len(state.size()) == 2 and state.size()[1] == self.state_only_input_dim[0] and len(action.size()) == 2

            if self.opt_robot_w_env_rewards:
                # In case that we optimize robot policy wrt environments rewards,
                # we feed Discriminator only with human part of the action

                half_action_size = int(action.size()[1] / 2)

                action = action[:, :half_action_size] if self.human_controls_axis == 'X' else \
                         (action[:, half_action_size:] if self.human_controls_axis == 'Y' else None)

            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()

        # print('test')
        # print(reward)
        # print(torch.FloatTensor(np.sqrt(np.expand_dims(
        #                                   np.array([self.ret_rms.var if len(self.ret_rms.var.shape) == 0 else
        #                                             self.ret_rms.var[0]]), axis=1) + 1e-8)).to(self.device))
        # print(reward / torch.FloatTensor(np.sqrt(np.expand_dims(
        #                                   np.array([self.ret_rms.var if len(self.ret_rms.var.shape) == 0 else
        #                                             self.ret_rms.var[0]]), axis=1) + 1e-8)).to(self.device))
        return reward / torch.from_numpy(np.sqrt(np.expand_dims(
            np.array([self.ret_rms.var if len(self.ret_rms.var.shape) == 0 else
                      self.ret_rms.var[0]]), axis=1
                                                                ) + 1e-8)).float().to(self.device)

    def save_model(self):
        print('\nSaving discriminator model ...\n')
        torch.save(self.state_dict(), self.checkpoint_file)

        with open(os.path.join(self.chkpt_dir, 'ret_rms_discriminator_cogail.pkl'), 'wb') as ret_rms_file:
            pickle.dump(self.ret_rms, ret_rms_file, pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        print('\nLoading discriminator model ...\n')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))

        with open(os.path.join(self.chkpt_dir, 'ret_rms_discriminator_cogail.pkl'), 'rb') as ret_rms_file:
            self.ret_rms = pickle.load(ret_rms_file)
