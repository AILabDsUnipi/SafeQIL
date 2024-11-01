import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from torch import nn
from tqdm import tqdm


class ConstraintNet(nn.Module):
    """
    Code based on:
    https://github.com/shehryar-malik/icrl/blob/master/icrl/constraint_net.py
    """
    def __init__(self,
                 config: Dict,
                 obs_space: int,
                 n_actions: int,
                 expert_obs: np.ndarray,
                 expert_acs: np.ndarray,
                 device: torch.device = torch.device("cpu"),
                 only_test: bool = False,
                 chkpt_dir: str = "constr_net",
                 axis_agent: str = "X"):

        super(ConstraintNet, self).__init__()

        self.obs_space = obs_space
        self.n_actions = n_actions
        self.input_dims = self.obs_space + self.n_actions
        self.n_hidden_units_layer1 = config['PPO']['ICRL_constraint_net_layer1_size']
        self.n_hidden_units_layer2 = config['PPO']['ICRL_constraint_net_layer2_size']
        self.device = device
        self.only_test = only_test
        self.axis_agent = axis_agent
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.axis_agent + '_ppo')

        if self.only_test is False:
            self.expert_obs = expert_obs
            self.expert_acs = expert_acs

            n_iters = config['PPO']['n_iters']
            anneal_clr_by_factor = config['PPO']['ICRL_constraint_net_anneal_clr_by_factor']
            cn_lr = config['PPO']['ICRL_constraint_net_cn_lr']
            self.lr_schedule = lambda x: (anneal_clr_by_factor**(n_iters*(1 - x))) * cn_lr

            self.batch_size = config['PPO']['ICRL_constraint_net_batch_size']
            self.regularizer_coeff = config['PPO']['ICRL_constraint_net_regularizer_coeff']
            self.target_kl_old_new = config['PPO']['ICRL_constraint_net_target_kl_old_new']
            self.target_kl_new_old = config['PPO']['ICRL_constraint_net_target_kl_new_old']
            self.eps = 1e-05

            self.current_progress_remaining = 1.

        # Create the network and the optimizer
        self._build()

    def _build(self) -> None:

        # Create network and add sigmoid at the end
        if isinstance(self.n_hidden_units_layer2, int):
            self.network = nn.Sequential(
                    nn.Linear(self.input_dims, self.n_hidden_units_layer1),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden_units_layer1, self.n_hidden_units_layer2),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden_units_layer2, 1),
                    nn.Sigmoid())
        else:
            self.network = nn.Sequential(
                nn.Linear(self.input_dims, self.n_hidden_units_layer1),
                nn.ReLU(),
                nn.Linear(self.n_hidden_units_layer1, 1),
                nn.Sigmoid())
        self.network.to(self.device)

        if self.only_test is False:
            # Build optimizer
            self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr_schedule(1), eps=1e-05)

    def forward(self, x: th.tensor) -> th.tensor:
        return self.network(x)

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:

        assert obs.shape[1] == self.obs_space, ""
        assert len(acs.shape) == 1 and acs.shape[0] == 1, ""

        x = self.prepare_data(obs, acs)

        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=1)

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train_(self,
               iterations: np.ndarray,
               nominal_obs: np.ndarray,
               nominal_acs: np.ndarray,
               episode_lengths: np.ndarray,
               current_progress_remaining: float = 1):

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Prepare data
        assert len(nominal_acs.shape) == len(self.expert_acs.shape) == 1, ""
        assert nominal_obs.shape[1] == self.expert_obs.shape[1] == self.obs_space, ""
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions since we use importance sampling
        with th.no_grad():
            start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations), desc='Constraint-net training iteration: '):

            # Compute IS weights
            with th.no_grad():
                current_preds = self.forward(nominal_data).detach()
            is_weights, kl_old_new, kl_new_old = \
                self.compute_is_weights(start_preds.clone(), current_preds.clone(), episode_lengths)

            # Break if kl is very large
            if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                 (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                early_stop_itr = itr
                break

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):

                # Get batch data
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)

                ## Calculate loss
                expert_loss = th.mean(th.log(expert_preds + self.eps))
                nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                # Regularizer loss is R(θ) = -δ Σ|1-ζ_θ(τ)|
                regularizer_loss = self.regularizer_coeff * (th.mean(1-expert_preds) + th.mean(1-nominal_preds))
                loss = (-expert_loss + nominal_loss) + regularizer_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item(), \
                expert_loss.item(), \
                th.mean(th.log(nominal_preds + self.eps)).item(), \
                nominal_loss.item(),\
                regularizer_loss.item(), \
                th.mean(is_weights).detach().item(), \
                th.max(is_weights).detach().item(), \
                th.min(is_weights).detach().item(), \
                th.max(nominal_preds).item(), \
                th.min(nominal_preds).item(), \
                th.mean(nominal_preds).item(), \
                th.max(expert_preds).item(), \
                th.min(expert_preds).item(), \
                th.mean(expert_preds).item(), \
                kl_old_new.item(), \
                kl_new_old.item(), \
                early_stop_itr

    def compute_is_weights(self, preds_old: th.Tensor, preds_new: th.Tensor, episode_lengths: np.ndarray) -> th.tensor:
        with th.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [th.prod(ratio[cumulative[j]:cumulative[j+1]]) for j in range(n_episodes)]
            prod = th.tensor(prod)

            # Per step importance sampling
            is_weights = th.tensor(ratio/th.mean(ratio))

            # KL divergence formula: D_KL(P||Q) = Σ P(X)*log(P(X)/Q(X)) = Σ -P(X)*log(Q(X)/P(X))
            # However in the paper it is mentioned that an upper bound of the KL is computed as follows:
            # (kl_old_new) D_KL(π_old || π) <= 2log(E[ratio]) (here, the 2 is omitted)
            # (kl_new_old)  D_KL(π || π_old) <= E[(Π[ratio] - E[ratio]) * log(Π[ration])] / E[ratio]
            # Compute KL(old, current)
            kl_old_new = th.mean(-th.log(prod+self.eps))
            # Compute KL(current, old)
            prod_mean = th.mean(prod)
            kl_new_old = th.mean((prod-prod_mean)*th.log(prod+self.eps)/(prod_mean+self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(self,
                     obs: np.ndarray,
                     acs: np.ndarray) -> th.tensor:

        acs = self.reshape_actions(acs)
        concat = np.concatenate([obs, acs], axis=1)
        return th.tensor(concat, dtype=th.float32).to(self.device)

    def reshape_actions(self, acs):

        # Since we use discrete actions, we should transform the corresponding index to one-hot encoding.
        acs_ = acs.astype(int)
        acs = np.zeros([acs.shape[0], self.n_actions])
        acs[np.arange(acs_.shape[0]), acs_] = 1.

        return acs

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if not isinstance(self.batch_size, int):
            # In ICRL, this condition is used with 22500 samples.
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            size = min(nom_size, exp_size)
            indices = np.random.permutation(size)

            start_idx = 0
            while start_idx < size:
                batch_indices = indices[start_idx:start_idx+self.batch_size]
                yield batch_indices, batch_indices
                start_idx += self.batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule(current_progress_remaining)

    def save_models(self):
        print('Saving {} model to {} ...'.format(self.axis_agent + '_ppo_constraint_net', self.checkpoint_file + '_constraint_net'))
        torch.save(self.state_dict(), self.checkpoint_file + '_constraint_net')

    def load_models(self, load_checkpoint_path_name):
        print('Loading {} model from {} ...'.format(self.axis_agent + '_ppo_constraint_net', os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_constraint_net')))
        self.load_state_dict(torch.load(os.path.join(load_checkpoint_path_name, self.axis_agent + '_ppo_constraint_net'), map_location=self.device))
