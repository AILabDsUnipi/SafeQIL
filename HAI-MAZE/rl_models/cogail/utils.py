import torch.nn as nn
import torch
from typing import Tuple
import numpy as np
import random

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):

        assert len(actions.size()) == 2 and actions.size(1) == self.logits.size(1)

        return super().log_prob(actions)

    def mode(self):
        return self.probs.argmax(dim=2, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Categorical, self).__init__()

        self.action_space = action_space
        self.half_action_space = int(action_space / 2)

        # Initialize the weights with orthogonal initialization to avoid exploding/vanishing gradients
        # Also, initialize the biases with zero values.
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, action_space))

    def forward(self, x, return_logits=False):
        x = self.linear(x)

        # We need to reshape 'x' so as the input in categorical distribution to shaped as
        # [batch_size, num_of_different_distributions, action_space_of_each_player] where
        # 'num_of_different_distributions' is equal to the number of players (2 in our case).
        x_reshaped = x.view(x.size(0), int(self.action_space / self.half_action_space), self.half_action_space)

        if return_logits:
            return FixedCategorical(logits=x_reshaped), x
        else:
            return FixedCategorical(logits=x_reshaped)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class RunningMeanStd:

    """
    code reference: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class latent_code_variable:

    def __init__(self, device):

        self.device = device

        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.from_numpy(np.array([[i, j] for i in self.linspace for j in self.linspace])).float()
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0
        # When 'pivot_id' reaches 'pivot_num', the following formula "resets" 'pivot_id' to 1.
        # In this way, there are 25 indices (1-25), and for this reason 0 index is omitted
        # (the first pivot index used below to initialize 'random_variable' is 1).
        self.pivot_id = (self.pivot_id + 1) % self.pivot_num

        self.random_variable_noise = \
            torch.from_numpy(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).float()
        self.random_variable = self.pivot[self.pivot_id] + self.random_variable_noise

    def get_next_code(self):

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = \
            torch.from_numpy(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).float()
        self.random_variable = self.pivot[self.pivot_id] + self.random_variable_noise

        return self.random_variable.to(self.device)

    def get_current_code(self):
        return self.random_variable.to(self.device)

    def reset_pivot(self):
        self.pivot_id = 0
