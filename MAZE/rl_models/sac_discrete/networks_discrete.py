import os
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


# why retain the graph? Do not auto-free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss, retain_graph=True, clip_grad_norm=False, max_grad_norm=5, params=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if clip_grad_norm:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optim.step()

    grad_norm_clipped_value = 0.0
    if clip_grad_norm:
        grad_norm_clipped_value = \
            grad_norm.detach().cpu().numpy() - max_grad_norm \
                if grad_norm.detach().cpu().numpy() > max_grad_norm \
                else \
            0.0

    return grad_norm_clipped_value


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get_size(self):
        return len(self.storage)

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class Actor(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            n_hidden_units_layer1,
            n_hidden_units_layer2,
            name='actor',
            chkpt_dir='tmp/sac',
            device=torch.device("cpu")
    ):

        super(Actor, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.device = device
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units_layer1),
            nn.ReLU(),
            nn.Linear(n_hidden_units_layer1, n_hidden_units_layer2),
            nn.ReLU(),
            nn.Linear(n_hidden_units_layer2, action_dim)
        ).apply(init_weights)

    def forward(self, s):

        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(self.device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(self.device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()
        return action.item()

    def save_checkpoint(self, override=True, checkpoint_dir_suffix=0):
        model_name = self.name+'_sac'
        checkpoint_file = os.path.join(self.checkpoint_dir, model_name)

        if override is False:
            new_checkpoint_dir = os.path.join(self.checkpoint_dir, str(checkpoint_dir_suffix))
            if os.path.exists(new_checkpoint_dir) is False:
                os.mkdir(new_checkpoint_dir)
            checkpoint_file = os.path.join(new_checkpoint_dir, model_name)

        print('Saving {} model to {} ...'.format(model_name, checkpoint_file))
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, load_checkpoint_path):
        print('Loading {} model from {} ...'.format(self.name+'_sac', load_checkpoint_path))
        self.load_state_dict(torch.load(load_checkpoint_path, map_location=self.device))


class Critic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            n_hidden_units_layer1,
            n_hidden_units_layer2,
            name='critic',
            chkpt_dir='tmp/sac',
            device=torch.device("cpu")
    ):

        super(Critic, self).__init__()

        self.name = name
        self.device = device
        self.checkpoint_dir = chkpt_dir

        self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units_layer1, n_hidden_units_layer2)
        self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units_layer1, n_hidden_units_layer2)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)

        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2

    def save_checkpoint(self, override=True, checkpoint_dir_suffix=0):
        model_name = self.name + '_sac'
        checkpoint_file = os.path.join(self.checkpoint_dir, model_name)

        if override is False:
            new_checkpoint_dir = os.path.join(self.checkpoint_dir, str(checkpoint_dir_suffix))
            if os.path.exists(new_checkpoint_dir) is False:
                os.mkdir(new_checkpoint_dir)
            checkpoint_file = os.path.join(new_checkpoint_dir, model_name)

        print('Saving {} model to {} ...'.format(model_name, checkpoint_file))
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, load_checkpoint_path):
        print('Loading {} model from {} ...'.format(self.name + '_sac', load_checkpoint_path))
        self.load_state_dict(torch.load(load_checkpoint_path, map_location=self.device))


class DuelQNet(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            n_hidden_units_layer1,
            n_hidden_units_layer2
    ):

        super(DuelQNet, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units_layer1),
            nn.ReLU(),
            nn.Linear(n_hidden_units_layer1, n_hidden_units_layer2),
            nn.ReLU()
        ).apply(init_weights)

        self.action_head = nn.Linear(n_hidden_units_layer2, action_dim).apply(init_weights)
        self.value_head = nn.Linear(n_hidden_units_layer2, 1).apply(init_weights)

    def forward(self, s):

        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
