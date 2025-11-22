import os
from itertools import accumulate
from typing import Dict, Type, Optional, Any, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm
from gymnasium import spaces

from SCOPIL.algos.icrl.networks_continuous import BaseFeaturesExtractor, FlattenExtractor
from SCOPIL.utils.demonstration_utils import ExpertDataset


class ConstraintNet(nn.Module):
    """
    Code based on:
    https://github.com/shehryar-malik/icrl/blob/master/icrl/constraint_net.py
    """
    def __init__(
            self,
            config: Dict,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            device: th.device = th.device("cpu"),
            only_test: bool = False,
    ):

        super(ConstraintNet, self).__init__()

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space

        self.features_extractor = features_extractor_class(self.observation_space, **features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        # Define 'n_actions'
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, "Error: the action space must be a vector"
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            self.n_actions = action_space.n
        else:
            raise NotImplementedError(
                "Error: constraint net, not implemented for action space"
                f"of type {type(action_space)}."
                " Must be of type Gym Spaces: Box, Discrete."
            )

        self.input_dims = self.features_dim + self.n_actions
        self.n_hidden_units_layer1 = config['ICRL']['constraint_net_layer1_size']
        self.n_hidden_units_layer2 = config['ICRL']['constraint_net_layer2_size']
        self.device = device
        self.only_test = only_test

        if self.only_test is False:
            # Get the expert samples
            expert_dataset = ExpertDataset(
                config['ICRL']['expert_dataset_path'],
                device=self.device,
                use_images=config['game']['use_image_obs'],
                load_to_memory=config['ICRL']['load_demos_in_memory'],
                env_id=config['game']['env_id'],
                normalize_features=config['Experiment']['normalize_features'],
                normalize_rewards=config['Experiment']['normalize_rewards'],
                smooth_actions=config['ICRL']['smooth_actions'],
                smooth_factor=config['ICRL']['smooth_factor']
            )
            self.expert_acts, self.expert_obs = expert_dataset.get_all_actions_and_observations()
            # Delete the dataset to save memory
            del expert_dataset

            # Set the learning rate schedule
            n_iters = config['ICRL']['n_iters']
            anneal_clr_by_factor = config['ICRL']['constraint_net_anneal_clr_by_factor']
            cn_lr = config['ICRL']['constraint_net_cn_lr']
            self.lr_schedule = lambda x: (anneal_clr_by_factor**(n_iters*(1 - x))) * cn_lr

            # Get the hyperparameters from config
            self.batch_size = config['ICRL']['constraint_net_batch_size']
            self.regularizer_coeff = config['ICRL']['constraint_net_regularizer_coeff']
            self.target_kl_old_new = config['ICRL']['constraint_net_target_kl_old_new']
            self.target_kl_new_old = config['ICRL']['constraint_net_target_kl_new_old']
            self.constraint_net_importance_sampling = config['ICRL']['constraint_net_importance_sampling']

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
                    nn.Sigmoid()
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(self.input_dims, self.n_hidden_units_layer1),
                nn.ReLU(),
                nn.Linear(self.n_hidden_units_layer1, 1),
                nn.Sigmoid()
            )
        self.network.to(self.device)

        if self.only_test is False:
            # Build optimizer
            self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr_schedule(1), eps=1e-05)

    def check_data_format(
            self,
            obs: Union[th.Tensor, np.ndarray],
            acts: Union[th.Tensor, np.ndarray]
    ):

        # Check observation
        assert isinstance(obs, (th.Tensor, np.ndarray))
        assert obs.shape[1:] == self.observation_space.shape

        # Check actions
        assert isinstance(acts, (th.Tensor, np.ndarray))
        if isinstance(self.action_space, spaces.Box):
            assert len(acts.shape) == 2 and acts.shape[1] == self.n_actions
        elif isinstance(self.action_space, spaces.Discrete):
            assert len(acts.shape) == 2 and acts.shape[1] == 1
        else:
            raise NotImplementedError(
                "Error: constraint net, not implemented for action space"
                f"of type {type(self.action_space)}."
                " Must be of type Gym Spaces: Box, Discrete."
            )

        # Check the batch size dim
        assert obs.shape[0] == acts.shape[0]

    def forward(self, obs: Union[th.Tensor, np.ndarray], acts: Union[th.Tensor, np.ndarray]) -> th.tensor:

        self.check_data_format(obs, acts)
        obs, acts = self.prepare_data(obs, acts)

        obs = self.features_extractor(obs)
        x = th.cat([obs, acts], dim=1)

        return self.network(x)

    def cost_function(self, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:

        ## Add batch dim if needed
        # in 'obs'
        if obs.shape == self.observation_space.shape:
            obs = obs[None, ...]
        # in 'acts'
        if isinstance(self.action_space, spaces.Box):
            if len(acts.shape) == 1 and acts.shape[0] == self.n_actions:
                acts = acts[None, ...]
        elif isinstance(self.action_space, spaces.Discrete):
            if len(acts.shape) == 1 and acts.shape[0] == 1:
                acts = acts[None, ...]

        with th.no_grad():
            out = self.__call__(obs, acts)

        # Convert to numpy
        out = out.detach().cpu().numpy()

        # Remove batch dim
        out = out.squeeze(axis=1)

        cost = 1 - out
        return cost

    def train_network(
            self,
            iterations: int,
            nominal_obs: np.ndarray,
            nominal_acts: np.ndarray,
            episode_lengths: np.ndarray,
            current_progress_remaining: float = 1
    ):

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Save current network predictions since we use importance sampling
        with th.no_grad():
            start_preds = self.forward(nominal_obs, nominal_acts).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations), desc='Constraint-net training iteration: '):

            # Compute IS weights
            with th.no_grad():
                current_preds = self.forward(nominal_obs, nominal_acts).detach()
            is_weights, kl_old_new, kl_new_old = self.compute_is_weights(
                start_preds.clone(),
                current_preds.clone(),
                episode_lengths
            )

            # Break if kl is very large
            if (
                    (self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                    (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)
            ):
                early_stop_itr = itr
                break

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_obs.shape[0], self.expert_obs.shape[0]):

                # Get batch data
                nominal_obs_batch = nominal_obs[nom_batch_indices]
                nominal_acts_batch = nominal_acts[nom_batch_indices]
                expert_obs_batch = self.expert_obs[exp_batch_indices]
                expert_acts_batch = self.expert_acts[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None]
                if self.constraint_net_importance_sampling is False:
                    is_batch = th.ones(nominal_obs_batch.shape, dtype=th.float32, device=self.device)

                # Make predictions
                nominal_preds = self.__call__(nominal_obs_batch, nominal_acts_batch)
                expert_preds = self.__call__(expert_obs_batch, expert_acts_batch)

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

        training_returns = {
            "constraint_net_total_loss": loss.item(),
            "constraint_net_expert_loss": expert_loss.item(),
            "constraint_net_nominal_loss": nominal_loss.item(),
            "constraint_net_nominal_preds": th.mean(th.log(nominal_preds + self.eps)).item(),
            "constraint_net_regularizer_loss": regularizer_loss.item(),
            "mean_constraint_net_is_weight": th.mean(is_weights).detach().item(),
            "min_constraint_net_is_weight": th.min(is_weights).detach().item(),
            "max_constraint_net_is_weight": th.max(is_weights).detach().item(),
            "constraint_net_log_nominal_preds": th.mean(th.log(nominal_preds + self.eps)).item(),
            "mean_constraint_net_nominal_preds": th.mean(nominal_preds).item(),
            "min_constraint_net_nominal_preds": th.min(nominal_preds).item(),
            "max_constraint_net_nominal_preds": th.max(nominal_preds).item(),
            "constraint_net_log_expert_preds": th.mean(th.log(expert_preds + self.eps)).item(),
            "mean_constraint_net_expert_preds": th.mean(expert_preds).item(),
            "min_constraint_net_expert_preds": th.min(expert_preds).item(),
            "max_constraint_net_expert_preds": th.max(expert_preds).item(),
            "constraint_net_kl_old_new": kl_old_new.item(),
            "constraint_net_kl_new_old": kl_new_old.item(),
            "constraint_net_early_stop_iterations": early_stop_itr
        }
        return training_returns

    def compute_is_weights(
            self,
            preds_old: th.Tensor,
            preds_new: th.Tensor,
            episode_lengths: np.ndarray
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        with th.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [th.prod(ratio[cumulative[j]:cumulative[j+1]]) for j in range(n_episodes)]
            prod = th.tensor(prod)

            # Per step importance sampling
            is_weights = ratio/th.mean(ratio)

            # KL divergence formula: D_KL(P||Q) = Σ P(X)*log(P(X)/Q(X)) = Σ -P(X)*log(Q(X)/P(X))
            # However, in the paper it is mentioned that an upper bound of the KL is computed as follows:
            # (kl_old_new) D_KL(π_old || π) <= 2log(E[ratio]) (here, the '2' is omitted)
            # (kl_new_old)  D_KL(π || π_old) <= E[(Π[ratio] - E[ratio]) * log(Π[ration])] / E[ratio]
            # Compute KL(old, current)
            kl_old_new = th.mean(-th.log(prod+self.eps))
            # Compute KL(current, old)
            prod_mean = th.mean(prod)
            kl_new_old = th.mean((prod-prod_mean)*th.log(prod+self.eps)/(prod_mean+self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(
            self,
            obs: Union[np.ndarray, th.Tensor],
            acts: Union[np.ndarray, th.Tensor]
    ) -> Tuple[th.Tensor, th.Tensor]:

        acts = self.clip_actions(acts)
        acts = self.reshape_actions(acts)

        # Convert from numpy to torch
        if isinstance(acts, np.ndarray):
            acts = th.tensor(acts, dtype=th.float32).to(self.device)
        else:
            assert isinstance(acts, th.Tensor)
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs, dtype=th.float32).to(self.device)
        else:
            assert isinstance(obs, th.Tensor)

        return obs, acts

    def reshape_actions(self, acts: Union[th.Tensor, np.ndarray]) -> Union[th.Tensor, np.ndarray]:

        if isinstance(self.action_space, spaces.Discrete):
            # Transform the corresponding index to one-hot encoding.
            if isinstance(acts, np.ndarray):
                acts_int = acts.astype(int)
                acts = np.zeros([acts.shape[0], self.n_actions])
                acts[np.arange(acts_int.shape[0]), acts_int] = 1.
            elif isinstance(acts, th.Tensor):
                acts_int = acts.to(th.int)
                acts = th.zeros(acts.shape[0], self.n_actions, dtype=th.float32)
                acts[th.arange(acts_int.shape[0]), acts_int] = 1
            else:
                raise NotImplementedError(
                    f"Invalid type of actions in constraint net 'reshape_actions' function: {type(acts)}"
                )

        return acts

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if not isinstance(self.batch_size, int):  # In ICRL, this condition is used with 22500 samples.
            min_size = min(nom_size, exp_size)
            # Randomly select indices without replacements from each set.
            # This approach, which differs from the original ICRL, ensures that
            # no samples are repeatedly excluded across iterations when the nominal samples
            # are always less than the expert samples (and vice versa).
            nom_indices = np.random.choice(nom_size, size=min_size, replace=False)
            exp_indices = np.random.choice(exp_size, size=min_size, replace=False)
            yield nom_indices, exp_indices
        else:
            size = min(nom_size, exp_size)
            indices = np.random.permutation(size)

            start_idx = 0
            while start_idx < size:
                batch_indices = indices[start_idx:start_idx+self.batch_size]
                yield batch_indices, batch_indices
                start_idx += self.batch_size

    def clip_actions(self, acts: Union[th.Tensor, np.ndarray]) -> Union[th.Tensor, np.ndarray]:
        clipped_actions = acts
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, spaces.Box):
            if isinstance(acts, np.ndarray):
                clipped_actions = np.clip(
                    acts,
                    self.action_space.low,
                    self.action_space.high
                )
            elif isinstance(acts, th.Tensor):
                clipped_actions = th.clamp(
                    acts,
                    min=th.tensor(self.action_space.low, dtype=acts.dtype, device=acts.device),
                    max=th.tensor(self.action_space.high, dtype=acts.dtype, device=acts.device)
                )
            else:
                raise NotImplementedError(
                    f"Invalid type of actions in constraint net 'clip_action' function: {type(acts)}"
                )
        return clipped_actions

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule(current_progress_remaining)

    def save_models(self, prefix_model_name, path):
        path_to_save_constraint_net = os.path.join(path, f'{prefix_model_name}_constraint_net.pt')
        print(
            'Saving {} model to {} ...'.format(
                'constraint_net',
                path_to_save_constraint_net
            )
        )
        th.save(self.state_dict(), path_to_save_constraint_net)

    def load_models(self, prefix_model_name, path):
        path_to_load_constraint_net = os.path.join(path, f'{prefix_model_name}_constraint_net.pt')
        print(
            'Loading {} model from {} ...'.format(
                'constraint_net',
                path_to_load_constraint_net
            )
        )
        self.load_state_dict(
            th.load(
                path_to_load_constraint_net,
                map_location=self.device,
                weights_only=False
            )
        )
