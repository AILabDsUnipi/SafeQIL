import torch
from statistics import mean
from tqdm import tqdm

from rl_models.cogail.ppo import PPO
from rl_models.cogail.model import Policy
from rl_models.cogail.gail import Discriminator
from rl_models.cogail.storage import RolloutStorage
from rl_models.utils.expert_dataset import exprt_dataset
from rl_models.cogail.utils import latent_code_variable

def get_cogail_agents(config, env, chkpt_dir=None, only_test=False):
    return coGAIL_agents(config, env, chkpt_dir, only_test)


class coGAIL_agents:
    def __init__(self, config, env, chkpt_dir=None, only_test=False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.chkpt_dir = chkpt_dir

        self.actor_critic = Policy(obs_shape=env.observation_shape,
                                   # x2 because we want the action space for both axes (but not as a combination of them)
                                   action_space=env.action_space.actions_number * 2,
                                   hidden_size=config['coGAIL']['hidden_size'],
                                   code_size=config['coGAIL']['code_size'],
                                   chkpt_dir=self.chkpt_dir,
                                   device=self.device,
                                   human_controls_axis=config['coGAIL']['human_controls_axis'],
                                   opt_robot_w_env_rewards=config['coGAIL']['opt_robot_w_env_rewards'],
                                   only_test=only_test).to(self.device)

        self.code_variable_test = latent_code_variable(self.device)

        # Plus 2 because each agent has only one discrete action, and we have two agents
        self.discr = Discriminator(
            input_dim=env.observation_shape[0] + (2 if not config['coGAIL']['opt_robot_w_env_rewards'] else 1),
            state_only_input_dim=env.observation_shape,
            hidden_dim=config['coGAIL']['hidden_size'],
            device=self.device,
            chkpt_dir=self.chkpt_dir,
            human_controls_axis=config['coGAIL']['human_controls_axis'],
            opt_robot_w_env_rewards=config['coGAIL']['opt_robot_w_env_rewards'],
            only_test=only_test,
            expert_torch_dataset=
            None if only_test else exprt_dataset(config['coGAIL']['expert_dataset_paths']),
            batch_size=config['coGAIL']['gail_batch_size']).to(self.device)

        self.pi_co = None
        self.rollout_storage = None
        self.code_variable_train = None

        if not only_test:

            self.constr_ball_only_at_the_right_side_wrt_hole = \
                config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']
            self.constr_ball_only_at_the_up_side_wrt_hole = \
                config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']

            # Add extra space to buffer due to non-standard number of rollouts steps in each episode
            # Also, if any constraint is applied, add an extra tensor to the buffer for the constraint values of the next states
            num_constraints_values = 0
            if config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']:
                num_constraints_values += 1
            elif config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']:
                num_constraints_values += 1
            rollout_storage = RolloutStorage(num_steps=config['coGAIL']['num_rollouts_steps'] + 500,
                                             obs_shape=env.observation_shape,
                                             random_seed_shape=config['coGAIL']['code_size'],
                                             device=self.device,
                                             opt_robot_w_env_rewards=config['coGAIL']['opt_robot_w_env_rewards'],
                                             num_constraints_values=num_constraints_values)

            self.pi_co = PPO(actor_critic=self.actor_critic,
                             clip_param=config['coGAIL']['ppo_clip_param'],
                             ppo_epoch=config['coGAIL']['ppo_epoch'],
                             num_mini_batch=config['coGAIL']['ppo_num_mini_batch'],
                             value_loss_coef=config['coGAIL']['ppo_value_loss_coef'],
                             entropy_coef=config['coGAIL']['ppo_entropy_coef'],
                             lr=config['coGAIL']['ppo_lr'],
                             eps=config['coGAIL']['ppo_eps'],
                             max_grad_norm=config['coGAIL']['ppo_max_grad_norm'],
                             device=self.device,
                             constr_ball_only_at_the_right_side_wrt_hole=self.constr_ball_only_at_the_right_side_wrt_hole,
                             constr_ball_only_at_the_up_side_wrt_hole=self.constr_ball_only_at_the_up_side_wrt_hole,
                             initial_lambda_constraint=config['coGAIL']['initial_lambda_constraint'],
                             eps_constraint=config['coGAIL']['eps_constraint'],
                             delta_constraint=config['coGAIL']['delta_constraint'],
                             w_actor_critic_gradient_clipping=config['coGAIL']['w_actor_critic_gradient_clipping'],
                             rollout_storage=rollout_storage)

            self.code_variable_train = latent_code_variable(self.device)

            self.gail_batch_size = config['coGAIL']['gail_batch_size']
            self.gail_epoch = config['coGAIL']['gail_epoch']
            self.gail_warm_up_epoch = config['coGAIL']['gail_warm_up_epoch']
            self.gail_warm_up_num_episodes = config['coGAIL']['gail_warm_up_num_episodes']
            self.ppo_gamma = config['coGAIL']['ppo_gamma']
            self.ppo_gae_lambda = config['coGAIL']['ppo_gae_lambda']

    def learn(self, episode):

        gail_epoch = self.gail_epoch
        if episode < self.gail_warm_up_num_episodes:
            gail_epoch = self.gail_warm_up_epoch  # Warm up phase

        # Update Discriminator network
        discr_BCE_loss_list = []
        discr_grad_pen_loss_list = []
        for _ in tqdm(range(gail_epoch), desc='GAIL training epochs'):
            discr_BCE_loss, discr_grad_pen = self.discr.update(self.pi_co.rollout_storage)
            discr_BCE_loss_list.append(discr_BCE_loss)
            discr_grad_pen_loss_list.append(discr_grad_pen)
        avg_discr_BCE_loss = mean(discr_BCE_loss_list)
        avg_discr_grad_pen_loss = mean(discr_grad_pen_loss_list)

        # Get rewards from Discriminator
        self.pi_co.rollout_storage.rewards[:self.pi_co.rollout_storage.step] = \
            self.discr.predict_reward(self.pi_co.rollout_storage.obs[:self.pi_co.rollout_storage.step],
                                      self.pi_co.rollout_storage.actions[:self.pi_co.rollout_storage.step],
                                      self.ppo_gamma,
                                      self.pi_co.rollout_storage.masks[:self.pi_co.rollout_storage.step])

        if self.pi_co.rollout_storage.opt_robot_w_env_rewards:
            # Normalize environment rewards (as in case of Discriminator rewards) to stabilize Actor's and Critic's training
            self.pi_co.rollout_storage.normalize_env_rewards(self.ppo_gamma)

        # Compute value targets based on the Discriminator's rewards and the predicted next values
        self.pi_co.rollout_storage.compute_returns(self.ppo_gamma, self.ppo_gae_lambda)

        # Update Actor and Critic
        value_loss, action_loss, dist_entropy, code_loss, inv_loss, actor_critic_avg_grad_norm_clipped_value, \
            discr_value_loss, env_value_loss, human_action_loss, robot_action_loss, \
            robot_final_constraint_term_loss, constraint_lambda_loss, constraint_lambda = \
            self.pi_co.update(self.pi_co.rollout_storage, self.discr.train_loader)

        return avg_discr_BCE_loss, \
               avg_discr_grad_pen_loss, \
               self.pi_co.rollout_storage.rewards[:self.pi_co.rollout_storage.step].detach().cpu().numpy().copy().squeeze(1), \
               value_loss, \
               action_loss, \
               dist_entropy, \
               code_loss, \
               inv_loss, \
               actor_critic_avg_grad_norm_clipped_value, \
               discr_value_loss, \
               env_value_loss, \
               human_action_loss, \
               robot_action_loss, \
               robot_final_constraint_term_loss, \
               constraint_lambda_loss, \
               constraint_lambda


