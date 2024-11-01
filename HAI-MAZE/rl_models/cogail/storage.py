import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

from rl_models.cogail.utils import RunningMeanStd

class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, random_seed_shape, device, opt_robot_w_env_rewards=False, num_constraints_values=0):

        self.obs_shape = obs_shape
        self.random_seed_shape = random_seed_shape
        self.num_steps = num_steps
        self.device = device
        self.opt_robot_w_env_rewards = opt_robot_w_env_rewards
        self.num_constraints_values = num_constraints_values

        self.obs = None
        self.random_seed = None
        self.rewards = None
        self.value_preds = None
        self.returns = None
        self.action_log_probs = None
        self.actions = None
        self.masks = None
        self.bad_masks = None
        self.step = None
        if self.opt_robot_w_env_rewards:
            self.env_rewards = None
            self.env_value_preds = None
            self.env_returns = None
            self.returns_ = None
            self.ret_rms = RunningMeanStd(shape=())
        if num_constraints_values > 0:
            self.next_obs_constr = None

        self.reset()

    def reset(self):
        # Initialize with nans to make debugging easier.
        # That is, since we have not a standard rollouts number of steps for each episode,
        # we need to monitor the number of steps. Therefore, if there is any bug in monitoring and
        # using the stored data, we can find it out by the resulting nan if any calculation will be applied
        # on these nan-samples.
        self.obs = torch.zeros(self.num_steps + 1, *self.obs_shape) * np.nan
        self.random_seed = torch.zeros(self.num_steps + 1, self.random_seed_shape) * np.nan
        self.rewards = torch.zeros(self.num_steps, 1) * np.nan
        self.value_preds = torch.zeros(self.num_steps + 1, 1) * np.nan
        self.returns = torch.zeros(self.num_steps + 1, 1) * np.nan
        # Discrete action space with two players
        self.action_log_probs = torch.zeros(self.num_steps, 2) * np.nan
        self.actions = torch.zeros(self.num_steps, 2).long() * np.nan
        self.masks = torch.zeros(self.num_steps + 1, 1) * np.nan
        # Masks that indicate whether it's a true terminal state or due to hitting the time limit
        self.bad_masks = torch.zeros(self.num_steps + 1, 1) * np.nan
        if self.opt_robot_w_env_rewards:
            self.env_rewards = torch.zeros(self.num_steps, 1) * np.nan
            self.env_value_preds = torch.zeros(self.num_steps + 1, 1) * np.nan
            self.env_returns = torch.zeros(self.num_steps + 1, 1) * np.nan
        if self.num_constraints_values > 0:
            self.next_obs_constr = torch.zeros(self.num_steps, self.num_constraints_values) * np.nan

        self.step = 0

        self.to()

    def to(self):
        self.obs = self.obs.to(self.device)
        self.random_seed = self.random_seed.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.env_rewards = self.env_rewards.to(self.device)
        self.value_preds = self.value_preds.to(self.device)
        self.returns = self.returns.to(self.device)
        self.action_log_probs = self.action_log_probs.to(self.device)
        self.actions = self.actions.to(self.device)
        self.masks = self.masks.to(self.device)
        self.bad_masks = self.bad_masks.to(self.device)
        if self.opt_robot_w_env_rewards:
            self.env_rewards = self.env_rewards.to(self.device)
            self.env_value_preds = self.env_value_preds.to(self.device)
            self.env_returns = self.env_returns.to(self.device)
        if self.num_constraints_values > 0:
            self.next_obs_constr = self.next_obs_constr.to(self.device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks, random_seed,
               env_rewards=None, env_value_preds=None):

        if self.num_constraints_values > 0:
            self.next_obs_constr[self.step].copy_(obs[self.obs_shape[0]:])
            self.obs[self.step + 1].copy_(obs[:self.obs_shape[0]])
        else:
            self.obs[self.step + 1].copy_(obs)
        self.random_seed[self.step + 1].copy_(random_seed)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if self.opt_robot_w_env_rewards:
            self.env_rewards[self.step].copy_(env_rewards)
            self.env_value_preds[self.step].copy_(env_value_preds)

        self.step += 1

    def after_update(self):
        self.reset()

    def compute_returns(self, gamma, gae_lambda):

        assert str(self.rewards[self.step, 0].detach().cpu().numpy()) == 'nan' and \
               str(self.value_preds[self.step + 1, 0].detach().cpu().numpy()) == 'nan' and \
               str(self.masks[self.step + 1, 0].detach().cpu().numpy()) == 'nan' and \
               str(self.bad_masks[self.step + 1, 0].detach().cpu().numpy()) == 'nan' and \
               str(self.returns[self.step, 0].detach().cpu().numpy()) == 'nan'

        # Use 'GAE' (Generalized Advantage Estimation) and 'proper time limits'.
        # For the latter, see how the 'bad_masks' are used. Intuitively, when 'bad_bask = True'
        # (i.e., when time horizon is hit) GAE becomes zero. In this way, the corresponding state is not
        # taken into account during value loss calculation (since the return is used as targets for the
        # predicted values).
        #
        # Using 'GAE', we trade off between bias and variance which occur when computing
        # A_t^(1) and A_t^(n) to estimate the advantage of state 's_t' with 1 or 'n' steps, respectively.
        # 'GAE' for timestep 't' is computed as follows:
        # A_t^GAE(γ,λ) := (1-λ)(A_t^(1) + λ * A_t^(2) + λ^2 * A_t^(3) + ...)
        #               = Σ_{l=0}^{+inf} (γ*λ)^l * δ_{t+l}^V
        #               = δ_t^V + γ*λ * δ_{t+1}^V + (γ*λ)^2 * δ_{t+2}^V + ...        (1)
        # Accordingly,
        # A_{t+1}^GAE(γ,λ) = δ_{t+1}^V + γ*λ * δ_{t+2}^V + (γ*λ)^2 * δ_{t+3}^V + ... (2)
        # Pay attention to the recursive nature of (1) and (2) based on which A_t^GAE(γ,λ)
        # can be calculated using A_{t+1}^GAE(γ,λ): A_t^GAE(γ,λ) = γ*λ * A_{t+1}^GAE(γ,λ).
        # The same is true for A_{t+1}^GAE(γ,λ) and A_{t+2}^GAE(γ,λ), etc.
        # This is why we use the 'reversed' for loop.
        # Concerning delta, it is calculated as follows:
        # δ_t^V = A^(1)_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # which is exactly the same as in the code below with the additional 'mask' to account for
        # the terminal states.
        # Note that δ_t^V is defined as the TD estimate of advantage for 1 step, i.e., δ_t^V = A_t^(1) .
        #
        # References:
        # 1) https://arxiv.org/pdf/1506.02438.pdf
        # 2) https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
        # 3) https://nn.labml.ai/rl/ppo/gae.html

        gae = 0
        for step in reversed(range(self.step)):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            gae = gae * self.bad_masks[step + 1]
            # Here we add 'value_preds' to 'gae' since we calculate 'returns' as targets for the predicted 'values'.
            # Intuitively, 'gae' should be close to zero if 'value_preds' are accurately predicted by the corresponding network.
            # This is because 'self.value_preds[step + 1] - self.value_preds[step]' should be equal to 'self.rewards[step]'.
            self.returns[step] = gae + self.value_preds[step]

        # If we optimize robot policy wrt environment rewards, we need compute the corresponding advantages and returns
        # based on the environment rewards
        if self.opt_robot_w_env_rewards:

            assert str(self.env_rewards[self.step, 0].detach().cpu().numpy()) == 'nan' and \
                   str(self.env_value_preds[self.step + 1, 0].detach().cpu().numpy()) == 'nan' and \
                   str(self.env_returns[self.step, 0].detach().cpu().numpy()) == 'nan'

            env_gae = 0
            for step in reversed(range(self.step)):
                env_delta = self.env_rewards[step] + gamma * self.env_value_preds[step + 1] * self.masks[step + 1] - self.env_value_preds[step]
                env_gae = env_delta + gamma * gae_lambda * self.masks[step + 1] * env_gae
                env_gae = env_gae * self.bad_masks[step + 1]
                self.env_returns[step] = env_gae + self.env_value_preds[step]

    def normalize_env_rewards(self, gamma):

        first_time = False
        if self.returns_ is None:
            first_time = True
            print('\nNormalizing environment rewards for the first time!! '
                  '\nInstead of the running variance, the first reward will be divided by 1.0!!\n')

        reward = self.env_rewards[:self.step]
        masks = self.masks[:self.step]

        np_reward = reward.detach().cpu().numpy()
        np_masks = masks.detach().cpu().numpy()

        self.returns_ = np.array([[0.0]], dtype=np.float64)
        ret_rms_var = []

        for rew in range(np_reward.shape[0]):
            mask_ = np.expand_dims(np_masks[rew], axis=1)
            reward_ = np.expand_dims(np_reward[rew], axis=1)

            self.returns_ = self.returns_ * mask_ * gamma + reward_
            self.ret_rms.update(self.returns_)

            ret_rms_var.append(self.ret_rms.var[0] if not (first_time and rew == 0) else 1.0)

        self.env_rewards[:self.step] = \
            reward / torch.from_numpy(np.sqrt(np.expand_dims(np.array(ret_rms_var), axis=1) + 1e-8)).float().to(self.device)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, env_advantages=None):

        batch_size = self.step # Number of steps performed

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of steps = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_steps, num_mini_batch))

            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[indices]
            random_seed_batch = self.random_seed[indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            masks_batch = self.masks[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]
            if env_advantages is None:
                env_adv_targ = None
            else:
                env_adv_targ = env_advantages.view(-1, 1)[indices]
            if not self.opt_robot_w_env_rewards:
                env_value_preds_batch = None
                env_return_batch = None
            else:
                env_value_preds_batch = self.env_value_preds[indices]
                env_return_batch = self.env_returns[indices]
            if self.num_constraints_values == 0:
                next_obs_constr_batch = None
            else:
                next_obs_constr_batch = self.next_obs_constr[indices]

            yield obs_batch, random_seed_batch, actions_batch, value_preds_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                  env_value_preds_batch, env_return_batch, env_adv_targ, next_obs_constr_batch

