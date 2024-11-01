import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

from rl_models.cogail.utils import init, Categorical

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size=64, code_size=2,
                 chkpt_dir=None, device=None, human_controls_axis='Y', opt_robot_w_env_rewards=False,
                 only_test=False):

        super(Policy, self).__init__()

        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'actor_critic_cogail.pt')

        self.device = device
        self.opt_robot_w_env_rewards = opt_robot_w_env_rewards
        self.only_test = only_test

        self.base = MLPBase(obs_shape[0], hidden_size=hidden_size,
                            code_size=code_size, opt_robot_w_env_rewards=opt_robot_w_env_rewards)

        # Discrete action space
        self.dist = Categorical(self.base.output_size, action_space)
        self.action_space = action_space
        self.half_action_space = int(action_space / 2)
        self.human_controls_axis = human_controls_axis

        # Initialize the weights with orthogonal initialization to avoid exploding/vanishing gradients
        # Also, initialize the biases with zero values.
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.recode = nn.Sequential(init_(nn.Linear(obs_shape[0] + self.half_action_space, hidden_size)), nn.ReLU(),
                                    init_(nn.Linear(hidden_size, code_size)), nn.Tanh())

        if self.only_test:
            self.base.eval()
            self.dist.eval()
            self.recode.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, random_seeds, deterministic=False, return_logits=False):

        value, actor_features = self.base(inputs, random_seeds)

        if return_logits:
            dist, logits = self.dist(actor_features, return_logits=return_logits)
        else:
            dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        assert len(action.size()) == 3 and action.size(2) == 1

        action_log_probs = dist.log_probs(action.squeeze(dim=2))

        if return_logits:
            return value, action.squeeze(dim=2), action_log_probs, logits
        else:
            return value, action.squeeze(dim=2), action_log_probs

    def get_value(self, inputs, random_seeds):

        value, _ = self.base(inputs, random_seeds)
        return value

    def evaluate_actions(self, inputs, random_seeds, action):

        value, actor_features = self.base(inputs, random_seeds)
        dist, pred_action_logits = self.dist(actor_features, return_logits=True)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        assert pred_action_logits.size(1) == self.action_space
        # We use discrete action space, thus we should use the probability of each action as the input of ψ network
        # in order to allow the gradients to flow from ψ network to actor network. If we used the selected action
        # (which is obtained through the non-differentiable argmax), this would not be possible.
        pred_human_action_logits = pred_action_logits[:, :self.half_action_space] \
                                   if self.human_controls_axis == 'X' else \
                                   (pred_action_logits[:, self.half_action_space:]
                                    if self.human_controls_axis == 'Y' else None)
        pred_human_action_probs = F.softmax(pred_human_action_logits, dim=1)

        input_code = torch.cat((inputs, pred_human_action_probs), dim=1)
        pred_code = self.recode(input_code)

        return value, action_log_probs, dist_entropy, pred_code, pred_human_action_probs

    def evaluate_code(self, inputs, action):
        """
        Predicts the latent code based on the demonstrated observations and actions
        :param inputs: torch Tensor observations obtained by demonstrations
        :param action: torch Tensor action obtained by demonstrations in the form of integers (i.e, action-class index).
        """

        assert action.size(1) == int(self.action_space / self.half_action_space) and len(action.size()) == 2
        # As stated above, we use the probability of each action as the input of ψ network.
        # Correspondingly, we should encode the human expert action (which is the form of an integer)
        # as one-hot.
        human_action = action[:, :int((self.action_space / self.half_action_space) / 2)] \
                       if self.human_controls_axis == 'X' else \
                       (action[:, int((self.action_space / self.half_action_space) / 2):]
                        if self.human_controls_axis == 'Y' else None)
        # Flatten human action before encoding them
        one_hot_human_action = F.one_hot(human_action.view(-1), num_classes=self.half_action_space)

        input_code = torch.cat((inputs, one_hot_human_action), dim=1)
        pred_code = self.recode(input_code)

        return pred_code

    def save_model(self, ret_rms_env_value=None):
        print('\nSaving actor critic model ...\n')
        torch.save(self.state_dict(), self.checkpoint_file)

        if self.opt_robot_w_env_rewards:
            with open(os.path.join(self.chkpt_dir, 'ret_rms_env_value_cogail.pkl'), 'wb') as ret_rms_file:
                pickle.dump(ret_rms_env_value, ret_rms_file, pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        print('\nLoading actor critic model ...\n')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))

class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64, code_size=2, opt_robot_w_env_rewards=False):
        super(MLPBase, self).__init__()

        self.code_size = code_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.opt_robot_w_env_rewards = opt_robot_w_env_rewards

        # Initialize the weights with orthogonal initialization to avoid exploding/vanishing gradients
        # (not used in PPO or PPO2).
        # Also, initialize the biases with zero values.
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # Use small base for policy which works better for low dimension controllers,
        # as mentioned here: https://github.com/j96w/cogail/blob/main/configs/exp1_config.py
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, self.output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs + self.code_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # In case that we optimize robot policy for environment rewards, we need two predicted values:
        # 1) Value wrt Discriminator rewards (for human policy)
        # 2) Value wrt environment rewards (for robot policy)
        self.critic_linear = init_(nn.Linear(hidden_size, 1 if not self.opt_robot_w_env_rewards else 2))

    def forward(self, inputs, random_seed):

        assert len(inputs.size()) == 2 and len(random_seed.size()) == 2

        x = torch.cat((inputs, random_seed), dim=1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor
