import os
import csv
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean
import itertools
import torch
from datetime import timedelta
from typing import List, Union

# Game utility file
from game.game_utils import get_env_action, get_distance_traveled,\
                            get_row_to_store, test_print_logs, \
                            column_names

from maze3D_new.utils import normalize_features
from rl_models.utils.algo_utils import transform_action_to_actions, transform_actions_to_action


class Experiment:
    def __init__(self, environment, agents=None, config=None):
        # retrieve parameters
        self.config = config  # configuration file dictionary
        self.env = environment  # environment to play in
        self.test_model = config['game']['test_model']  # check if only test
        if agents is not None:
            assert len(config['game']['checkpoint_path']) == len(config['game']['checkpoint_name'])
            self.load_checkpoint_path_name = [
                os.path.join(config['game']['checkpoint_path'][path_name], config['game']['checkpoint_name'][path_name])
                for path_name in range(len(config['game']['checkpoint_path']))
            ]
            self.load_checkpoint = config['game']['load_checkpoint']
        self.algo = None

        if 'SAC' in list(config.items())[1] or 'PPO' in list(config.items())[1]:
            self.algo = 'SAC' if 'SAC' in list(config.items())[1] else 'PPO'
            if len(agents) == 2:
                self.X_agent = agents[0]  # X_agent controls X-axis (up/down)
                self.Y_agent = agents[1]  # Y_agent controls Y-axis (right/left)
                self.X_Y_agent = None
            elif len(agents) == 1:
                self.X_agent = None
                self.Y_agent = None
                self.X_Y_agent = agents[0]  # A single agent controls both axes
            else:
                raise NotImplementedError
            if self.load_checkpoint:
                print(
                    "Loading " + self.algo + " model(s) from checkpoint(s): \n{}\n\n and\n\n {}\n".format(
                        self.load_checkpoint_path_name[0] if self.load_checkpoint_path_name[0] != ""
                                                                else
                            "NO FIRST PATH",
                            self.load_checkpoint_path_name[1] if self.load_checkpoint_path_name[1] != ""
                                                              else
                            "NO SECOND PATH"
                    )
                )
                self.X_Y_agent_pretrained = False if 'X_Y_agent_pretrained' not in self.config['game'].keys() \
                                                  else \
                                            config['game']['X_Y_agent_pretrained']
                if self.X_agent is not None and \
                   (not (self.X_Y_agent_pretrained and self.Y_agent.axis_agent == 'X_Y') or self.test_model):
                    self.X_agent.load_models(
                        self.load_checkpoint_path_name[1 if self.X_agent.axis_agent == 'X_Y'
                                                         else
                                                       0]
                    )
                if self.Y_agent is not None and \
                   (not (self.X_Y_agent_pretrained and self.X_agent.axis_agent == 'X_Y') or self.test_model):
                    self.Y_agent.load_models(
                        self.load_checkpoint_path_name[1 if self.Y_agent.axis_agent == 'X_Y'
                                                         else
                                                       0]
                    )
                if self.X_Y_agent is not None:
                    self.X_Y_agent.load_models(self.load_checkpoint_path_name[1])
        elif 'coGAIL' in list(config.items())[1]:
            self.algo = 'coGAIL'
            self.agents = agents
            self.actor_critic = agents.actor_critic
            self.code_variable_train = agents.code_variable_train
            self.code_variable_test = agents.code_variable_test
            self.pi_co = agents.pi_co
            self.discr = agents.discr
            self.X_agent = None
            self.Y_agent = None
            if config['game']['X_agent']:
                self.X_agent = True
            if config['game']['Y_agent']:
                self.Y_agent = True
            assert self.X_agent is not None or self.Y_agent is not None
        elif 'NO_ALGO' in list(config.items())[1]:
            print('\nPlay the game without an algorithm !')
            self.X_agent = None
            self.Y_agent = None
        else:
            raise NotImplementedError

        ## retrieve information from the config file
        # train and test
        self.goal = config["game"]["goal"]
        self.action_iterations = config['game']['action_iterations']
        self.test_window_size_moving_avg = config['Experiment']['test_window_size_moving_avg']
        self.debug_ = config['Experiment']['debug_']
        self.test_seed = config['Experiment']['test_seed']

        self.constr_ball_only_at_the_right_side_wrt_hole = \
            config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']
        self.constr_ball_only_at_the_up_side_wrt_hole = \
            config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']
        self.constr_ball_not_in_circle = \
            config['Experiment']['constraint_ball_not_in_circle']
        if self.constr_ball_only_at_the_right_side_wrt_hole is True or \
           self.constr_ball_only_at_the_up_side_wrt_hole is True or \
           self.constr_ball_not_in_circle is True:
            self.satisfied_constraint_value = config['Experiment']['satisfied_constraint_value']
            self.not_satisfied_constraint_value = config['Experiment']['not_satisfied_constraint_value']

        if self.algo == 'SAC' or self.algo == 'PPO' or self.algo == 'coGAIL':
            self.normalize_features = config[self.algo]['normalize_features']
            if self.algo == 'coGAIL':
                self.device = self.agents.device
            elif self.algo == 'PPO':
                self.icrl = config['PPO']['ICRL']
                # Define the device (CPU or GPU)
                self.device = None
                if self.X_agent is not None or self.Y_agent is not None:
                    if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                        self.device = self.X_agent.device
                    if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                        if isinstance(self.device, torch.device):
                            assert self.device == self.Y_agent.device
                        else:
                            self.device = self.Y_agent.device
                elif self.X_Y_agent is not None:
                    self.device = self.X_Y_agent.device
                else:
                    raise NotImplementedError
        else:
            self.normalize_features = False

        self.freeze_motion = False if 'freeze_motion' not in self.config['Experiment'].keys() \
                                   else \
                             self.config['Experiment']['freeze_motion']
        assert not self.freeze_motion or (self.freeze_motion and self.algo is None)

        # train
        if not self.test_model:
            if self.algo == 'SAC':
                self.start_steps = config['SAC']['start_steps']
                self.num_steps = config['SAC']['num_steps']
                self.n_episodes = config['SAC']['n_episodes']
                self.n_train_episodes = config['SAC']['n_train_episodes']
                self.batch_size = config['SAC']['batch_size']
                self.updates_per_step = config['SAC']['updates_per_step']
                self.test_every_episodes = config['Experiment']['test_every_episodes']
                self.SL_finetune = config['SAC']['SL_finetuning']
                self.SL_finetuning_epochs = config['SAC']['SL_finetuning_epochs']
                self.w_constraint_optimization = config['SAC']['w_constraint_optimization']
                if self.SL_finetune is True:
                    self.save_all = self.config['game']['save_all']
            elif self.algo == 'PPO':
                self.n_iters = config['PPO']['n_iters']
                self.forward_steps = config['PPO']['forward_steps']
                self.n_steps = config['PPO']['n_steps']
                self.ICRL_backward_iterations = config['PPO']['ICRL_backward_iterations']
                self.ICRL_backward_n_rollouts = config['PPO']['ICRL_backward_n_rollouts']
                self.lagrangian = config['PPO']['lagrangian']
                self.test_every_games = config['Experiment']['test_every_games']
                self.SL_finetune = config['PPO']['SL_finetuning']
                self.SL_finetuning_epochs = config['PPO']['SL_finetuning_epochs']
                if self.SL_finetune is True:
                    self.save_all = self.config['game']['save_all']
            elif self.algo == 'coGAIL':
                self.num_steps = config['coGAIL']['num_env_steps']
                self.num_rollouts_steps = config['coGAIL']['num_rollouts_steps']
                self.bc_pretrain_steps = config['coGAIL']['bc_pretrain_steps']
                self.test_every_games = config['Experiment']['test_every_games']
                # Calculate the approximate number of episodes which is necessary for the learning rate scheduler
                self.num_updates = int(self.num_steps) // self.num_rollouts_steps

            self.max_timesteps_per_game = config['Experiment']['max_timesteps_per_game']
            self.log_interval = self.config['Experiment']['log_interval']
            self.save = self.config['game']['save']
            self.save_last = self.config['game']['save_last']
            self.save_constraints = self.config['game']['save_constraints']

        # test
        self.test_max_games = config['Experiment']['test_max_games']
        self.test_max_timesteps_per_game = config['Experiment']['test_max_timesteps_per_game']

        ## initialize lists to keep track of information and variables
        # train
        if self.test_model is False:
            self.test_avg_best_score = -np.inf
            self.test_avg_lowest_num_constr = np.inf
            self.max_games = 0
            self.total_steps = 0
            self.start_game_time = 0
            self.end_game_time = 0
            self.redundant_end_duration = 0
            self.timed_out = False
            self.game_reward = 0
            self.dist_travel = 0
            self.train_step_counter = 0
            self.done = False
            self.fixed_done = 0.
            self.train_step_start_time = 0
            self.observation = None
            self.next_observation = None
            self.reward = None
            self.duration_pause = 0.
            self.setting_up_duration = 0
            self.action_list = None
            self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list = None
            self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list = None
            self.ball_not_in_circle_constraint_violation_list = None
            self.distance_travel_list, self.reward_list, self.game_duration_list = [], [], []
            self.train_step_duration_list = []
            self.episode_duration_list, self.length_list = [], []
            self.game_duration_list_avg_per_log_interval = []
            self.reward_list_avg_per_log_interval = []
            self.distance_travel_list_avg_per_log_interval = []
            self.length_list_avg_per_log_interval = []
            self.episodes_model_saved = []
            if self.debug_ is True:
                self.action_history = []
            if self.algo == 'SAC':
                self.i_train_episode = 0
                if self.X_agent is not None or self.Y_agent is not None:
                    if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                        if not self.SL_finetune:
                            self.X_q1_loss_per_step_list = []
                            self.X_q2_loss_per_step_list = []
                            self.X_entropies_per_step_list = []
                            self.X_entropy_loss_per_step_list = []
                            self.X_policy_loss_per_step_list = []
                            self.X_entropy_coef_per_step_list = []
                            self.X_q1_grad_norm_clipped_value_per_step_list = []
                            self.X_q2_grad_norm_clipped_value_per_step_list = []
                            self.X_actor_grad_norm_clipped_value_per_step_list = []
                            self.X_q1_loss_cur_game_per_step_list = []
                            self.X_q2_loss_cur_game_per_step_list = []
                            self.X_entropies_cur_game_per_step_list = []
                            self.X_entropy_loss_cur_game_per_step_list = []
                            self.X_policy_loss_cur_game_per_step_list = []
                            self.X_entropy_coef_cur_game_per_step_list = []
                            self.X_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                            self.X_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                            self.X_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                        else:
                            self.X_constraint_policy_loss_term_list, self.X_actor_grad_norm_clipped_value_list = [], []
                    if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                        if self.SL_finetune is False:
                            self.Y_q1_loss_per_step_list = []
                            self.Y_q2_loss_per_step_list = []
                            self.Y_entropies_per_step_list = []
                            self.Y_entropy_loss_per_step_list = []
                            self.Y_policy_loss_per_step_list = []
                            self.Y_entropy_coef_per_step_list = []
                            self.Y_q1_grad_norm_clipped_value_per_step_list = []
                            self.Y_q2_grad_norm_clipped_value_per_step_list = []
                            self.Y_actor_grad_norm_clipped_value_per_step_list = []
                            self.Y_q1_loss_cur_game_per_step_list = []
                            self.Y_q2_loss_cur_game_per_step_list = []
                            self.Y_entropies_cur_game_per_step_list = []
                            self.Y_entropy_loss_cur_game_per_step_list = []
                            self.Y_policy_loss_cur_game_per_step_list = []
                            self.Y_entropy_coef_cur_game_per_step_list = []
                            self.Y_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                            self.Y_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                            self.Y_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                        else:
                            self.Y_constraint_policy_loss_term_list, self.Y_actor_grad_norm_clipped_value_list = [], []
                    if self.w_constraint_optimization is True and self.SL_finetune is False:
                        if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                            self.X_constraint_policy_loss_term_value_per_step_list = []
                            self.X_constraint_lambda_loss_value_per_step_list = []
                            self.X_policy_loss_value_wo_constraint_term_per_step_list = []
                            self.X_constraint_lambda_per_step_list = []
                            self.X_constraint_policy_loss_term_value_cur_game_per_step_list = []
                            self.X_constraint_lambda_loss_value_cur_game_per_step_list = []
                            self.X_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                            self.X_constraint_lambda_cur_game_per_step_list = []
                        if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                            self.Y_constraint_policy_loss_term_value_per_step_list = []
                            self.Y_constraint_lambda_loss_value_per_step_list = []
                            self.Y_policy_loss_value_wo_constraint_term_per_step_list = []
                            self.Y_constraint_lambda_per_step_list = []
                            self.Y_constraint_policy_loss_term_value_cur_game_per_step_list = []
                            self.Y_constraint_lambda_loss_value_cur_game_per_step_list = []
                            self.Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                            self.Y_constraint_lambda_cur_game_per_step_list = []
                elif self.X_Y_agent is not None:
                    if self.SL_finetune is False:
                        self.X_Y_q1_loss_per_step_list = []
                        self.X_Y_q2_loss_per_step_list = []
                        self.X_Y_entropies_per_step_list = []
                        self.X_Y_entropy_loss_per_step_list = []
                        self.X_Y_policy_loss_per_step_list = []
                        self.X_Y_entropy_coef_per_step_list = []
                        self.X_Y_q1_grad_norm_clipped_value_per_step_list = []
                        self.X_Y_q2_grad_norm_clipped_value_per_step_list = []
                        self.X_Y_actor_grad_norm_clipped_value_per_step_list = []
                        self.X_Y_q1_loss_cur_game_per_step_list = []
                        self.X_Y_q2_loss_cur_game_per_step_list = []
                        self.X_Y_entropies_cur_game_per_step_list = []
                        self.X_Y_entropy_loss_cur_game_per_step_list = []
                        self.X_Y_policy_loss_cur_game_per_step_list = []
                        self.X_Y_entropy_coef_cur_game_per_step_list = []
                        self.X_Y_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                        self.X_Y_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                        self.X_Y_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                        if self.w_constraint_optimization is True:
                            self.X_Y_constraint_policy_loss_term_value_per_step_list = []
                            self.X_Y_constraint_lambda_loss_value_per_step_list = []
                            self.X_Y_policy_loss_value_wo_constraint_term_per_step_list = []
                            self.X_Y_constraint_lambda_per_step_list = []
                            self.X_Y_constraint_policy_loss_term_value_cur_game_per_step_list = []
                            self.X_Y_constraint_lambda_loss_value_cur_game_per_step_list = []
                            self.X_Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                            self.X_Y_constraint_lambda_cur_game_per_step_list = []
                    else:
                        self.X_Y_constraint_policy_loss_term_list, self.X_Y_actor_grad_norm_clipped_value_list = [], []

            elif self.algo == 'coGAIL':
                self.i_episode = 0
                self.episode_game_counter = 0
                self.total_games = 0
                self.episode_step_counter = 0
                self.episode_reward = 0
                self.code_variable_per_game_list = []
                self.bc_loss_per_update = []
                self.discr_loss_per_episode_list = []
                self.discr_grad_pen_loss_per_episode_list = []
                self.discr_rewards_per_step = []
                self.discr_rewards_per_episode_avg_over_games = []
                self.value_loss_per_episode_list = []
                self.action_loss_per_episode_list = []
                self.dist_entropy_per_episode_list = []
                self.code_loss_per_episode_list = []
                self.inv_loss_per_episode_list = []
                self.actor_critic_grad_norm_clipped_value_per_episode_list = []
                self.code_variable_per_episode_list = []
                if self.pi_co.opt_robot_w_env_rewards:
                    self.discr_value_loss_per_episode_list = []
                    self.env_value_loss_per_episode_list = []
                    self.human_action_loss_per_episode_list = []
                    self.robot_action_loss_per_episode_list = []
                if self.constr_ball_only_at_the_right_side_wrt_hole is True or \
                   self.constr_ball_only_at_the_up_side_wrt_hole is True or \
                   self.constr_ball_not_in_circle is True:
                    self.robot_final_constraint_term_loss_per_episode_list = []
                    self.constraint_lambda_loss_per_episode_list = []
                    self.constraint_lambda_per_episode_list = []
                    if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                        self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list = []
                    if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                        self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list = []
                    if self.constr_ball_not_in_circle is True:
                        self.ball_not_in_circle_episode_constraint_violation_list = []
            elif self.algo == 'PPO':
                self.i_episode = 0
                self.episode_game_counter = 0
                self.total_games = 0
                self.episode_step_counter = 0
                self.episode_reward = 0
                if self.icrl is True:
                    self.current_progress_remaining = None
                if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                    self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list = []
                if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                    self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list = []
                if self.constr_ball_not_in_circle is True:
                    self.ball_not_in_circle_episode_constraint_violation_list = []
                if self.X_agent is not None or self.Y_agent is not None:
                    if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                        if self.SL_finetune is False:
                            self.X_total_loss_per_episode_list = []
                            self.X_policy_loss_per_episode_list = []
                            self.X_reward_value_loss_per_episode_list = []
                            self.X_approx_kl_divs_per_episode_list = []
                            self.X_entropy_loss_per_episode_list = []
                            self.X_clip_fraction_per_episode_list = []
                            self.X_reward_advantage_per_episode_list = []
                            self.X_explained_rew_var_per_episode_list = []
                            self.X_early_stop_epoch_per_episode_list = []
                            if self.icrl is True or self.lagrangian is True:
                                self.X_total_policy_loss_per_episode_list = []
                                self.X_dual_nu_per_episode_list = []
                                self.X_dual_loss_per_episode_list = []
                                if self.icrl is True:
                                    self.X_cost_value_loss_per_episode_list = []
                                    self.X_cost_advantage_per_episode_list = []
                                    self.X_explained_cost_var_per_episode_list = []
                                    self.X_constr_net_cost_per_episode_avg_over_games = []
                                    self.X_constr_net_cost_per_step = []
                                    self.X_cost_advantage_ratio_term_per_episode_list = []
                                    self.X_cost_loss_per_episode_list = []
                                    self.X_total_loss_constr_net_per_iter_list = []
                                    self.X_expert_loss_constr_net_per_iter_list = []
                                    self.X_policy_loss_constr_net_wo_is_per_iter_list = []
                                    self.X_policy_loss_constr_net_per_iter_list = []
                                    self.X_regularizer_loss_constr_net_per_iter_list = []
                                    self.X_is_weights_mean_constr_net_per_iter_list = []
                                    self.X_is_weights_max_constr_net_per_iter_list = []
                                    self.X_is_weights_min_constr_net_per_iter_list = []
                                    self.X_policy_preds_max_constr_net_per_iter_list = []
                                    self.X_policy_preds_min_constr_net_per_iter_list = []
                                    self.X_policy_preds_mean_constr_net_per_iter_list = []
                                    self.X_expert_preds_max_constr_net_per_iter_list = []
                                    self.X_expert_preds_min_constr_net_per_iter_list = []
                                    self.X_expert_preds_mean_constr_net_per_iter_list = []
                                    self.X_kl_old_new_constr_net_per_iter_list = []
                                    self.X_kl_new_old_constr_net_per_iter_list = []
                                    self.X_early_stop_itr_constr_net_per_iter_list = []
                                elif self.lagrangian is True:
                                    self.X_lagrangian_constraint_policy_term_loss_per_episode_list = []
                        else:
                            self.X_constraint_policy_loss_term_list = []
                    if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                        if self.SL_finetune is False:
                            self.Y_total_loss_per_episode_list = []
                            self.Y_policy_loss_per_episode_list = []
                            self.Y_reward_value_loss_per_episode_list = []
                            self.Y_approx_kl_divs_per_episode_list = []
                            self.Y_entropy_loss_per_episode_list = []
                            self.Y_clip_fraction_per_episode_list = []
                            self.Y_reward_advantage_per_episode_list = []
                            self.Y_explained_rew_var_per_episode_list = []
                            self.Y_early_stop_epoch_per_episode_list = []
                            if self.icrl is True or self.lagrangian is True:
                                self.Y_total_policy_loss_per_episode_list = []
                                self.Y_dual_nu_per_episode_list = []
                                self.Y_dual_loss_per_episode_list = []
                                if self.icrl is True:
                                    self.Y_cost_value_loss_per_episode_list = []
                                    self.Y_cost_advantage_per_episode_list = []
                                    self.Y_explained_cost_var_per_episode_list = []
                                    self.Y_constr_net_cost_per_episode_avg_over_games = []
                                    self.Y_constr_net_cost_per_step = []
                                    self.Y_cost_advantage_ratio_term_per_episode_list = []
                                    self.Y_cost_loss_per_episode_list = []
                                    self.Y_total_loss_constr_net_per_iter_list = []
                                    self.Y_expert_loss_constr_net_per_iter_list = []
                                    self.Y_policy_loss_constr_net_wo_is_per_iter_list = []
                                    self.Y_policy_loss_constr_net_per_iter_list = []
                                    self.Y_regularizer_loss_constr_net_per_iter_list = []
                                    self.Y_is_weights_mean_constr_net_per_iter_list = []
                                    self.Y_is_weights_max_constr_net_per_iter_list = []
                                    self.Y_is_weights_min_constr_net_per_iter_list = []
                                    self.Y_policy_preds_max_constr_net_per_iter_list = []
                                    self.Y_policy_preds_min_constr_net_per_iter_list = []
                                    self.Y_policy_preds_mean_constr_net_per_iter_list = []
                                    self.Y_expert_preds_max_constr_net_per_iter_list = []
                                    self.Y_expert_preds_min_constr_net_per_iter_list = []
                                    self.Y_expert_preds_mean_constr_net_per_iter_list = []
                                    self.Y_kl_old_new_constr_net_per_iter_list = []
                                    self.Y_kl_new_old_constr_net_per_iter_list = []
                                    self.Y_early_stop_itr_constr_net_per_iter_list = []
                                elif self.lagrangian is True:
                                    self.Y_lagrangian_constraint_policy_term_loss_per_episode_list = []
                        else:
                            self.Y_constraint_policy_loss_term_list = []
                if self.X_Y_agent is not None:
                    if self.SL_finetune is False:
                        self.X_Y_total_loss_per_episode_list = []
                        self.X_Y_policy_loss_per_episode_list = []
                        self.X_Y_reward_value_loss_per_episode_list = []
                        self.X_Y_approx_kl_divs_per_episode_list = []
                        self.X_Y_entropy_loss_per_episode_list = []
                        self.X_Y_clip_fraction_per_episode_list = []
                        self.X_Y_reward_advantage_per_episode_list = []
                        self.X_Y_explained_rew_var_per_episode_list = []
                        self.X_Y_early_stop_epoch_per_episode_list = []
                        if self.icrl is True or self.lagrangian is True:
                            self.X_Y_total_policy_loss_per_episode_list = []
                            self.X_Y_dual_nu_per_episode_list = []
                            self.X_Y_dual_loss_per_episode_list = []
                            if self.icrl is True:
                                self.X_Y_cost_value_loss_per_episode_list = []
                                self.X_Y_cost_advantage_per_episode_list = []
                                self.X_Y_explained_cost_var_per_episode_list = []
                                self.X_Y_constr_net_cost_per_episode_avg_over_games = []
                                self.X_Y_constr_net_cost_per_step = []
                                self.X_Y_cost_advantage_ratio_term_per_episode_list = []
                                self.X_Y_cost_loss_per_episode_list = []
                                self.X_Y_total_loss_constr_net_per_iter_list = []
                                self.X_Y_expert_loss_constr_net_per_iter_list = []
                                self.X_Y_policy_loss_constr_net_wo_is_per_iter_list = []
                                self.X_Y_policy_loss_constr_net_per_iter_list = []
                                self.X_Y_regularizer_loss_constr_net_per_iter_list = []
                                self.X_Y_is_weights_mean_constr_net_per_iter_list = []
                                self.X_Y_is_weights_max_constr_net_per_iter_list = []
                                self.X_Y_is_weights_min_constr_net_per_iter_list = []
                                self.X_Y_policy_preds_max_constr_net_per_iter_list = []
                                self.X_Y_policy_preds_min_constr_net_per_iter_list = []
                                self.X_Y_policy_preds_mean_constr_net_per_iter_list = []
                                self.X_Y_expert_preds_max_constr_net_per_iter_list = []
                                self.X_Y_expert_preds_min_constr_net_per_iter_list = []
                                self.X_Y_expert_preds_mean_constr_net_per_iter_list = []
                                self.X_Y_kl_old_new_constr_net_per_iter_list = []
                                self.X_Y_kl_new_old_constr_net_per_iter_list = []
                                self.X_Y_early_stop_itr_constr_net_per_iter_list = []
                            elif self.lagrangian is True:
                                self.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list = []
                    else:
                        self.X_Y_constraint_policy_loss_term_list = []
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                self.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list = []
                self.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list = []
                self.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval = []
                self.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval = []
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                self.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list = []
                self.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list = []
                self.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval = []
                self.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval = []
            if self.constr_ball_not_in_circle is True:
                self.ball_not_in_circle_num_constraint_violated_list = []
                self.ball_not_in_circle_freq_constraint_violated_list = []
                self.ball_not_in_circle_num_constraint_violated_per_log_interval = []
                self.ball_not_in_circle_freq_constraint_violated_per_log_interval = []

        # test
        self.test_game_number, self.duration_pause_total = 0, 0
        self.test_reward_list = []
        self.test_step_duration_list = []
        self.test_length_list = []
        self.test_distance_travel_list = []
        self.test_game_duration_list = []
        if self.test_model is True or self.debug_ is True:
            self.test_action_history = []
        if self.test_model is False:
            self.test_game_duration_list_avg_per_test = []
            self.test_reward_list_avg_per_test = []
            self.test_distance_travel_list_avg_per_test = []
            self.test_length_list_avg_per_test = []
        if self.algo == 'coGAIL':
            self.code_variable_per_game_list_of_np_test = []
            self.discr_reward_per_game_list_test = []
        elif self.algo == 'PPO' and self.icrl is True:
            self.constraint_net_cost_per_game_list_test = []
        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
            self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list = []
            self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list = []
            if self.test_model is False:
                self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test = []
                self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test = []
        if self.constr_ball_only_at_the_up_side_wrt_hole is True:
            self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list = []
            self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list = []
            if self.test_model is False:
                self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test = []
                self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test = []
        if self.constr_ball_not_in_circle is True:
            self.test_ball_not_in_circle_num_constraint_violated_list = []
            self.test_ball_not_in_circle_freq_constraint_violated_list = []
            if self.test_model is False:
                self.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test = []
                self.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test = []

        ## initialize train and test dataframes
        # train
        if self.test_model is False and self.debug_ is True:
            self.df = pd.DataFrame(columns=column_names)
        # test
        if self.test_model is True or self.debug_ is True:
            self.df_test = pd.DataFrame(columns=column_names)

    def SL_finetuning_agent0s0(self):
        assert self.SL_finetune is True

        # Supervised Learning finetuning loop
        for self.epoch in range(self.SL_finetuning_epochs):

            print('\nEpoch: ' + str(self.epoch))

            # Train the networks and store the corresponding info
            self.train_store_and_print_info()

            # Testing
            self.test_during_training()

    def train_agent0s0_ppo(self):

        if self.SL_finetune is True:
            self.SL_finetuning_agent0s0()

        else:
            """
            Code based on: https://github.com/shehryar-malik/icrl/blob/master/icrl/icrl.py
            """

            # In this experiment, we train an (or two) agent(s) using
            # PPO or ICRL (with demonstrations for the latter).
            # The loop iterating over "n_iters" is used for alternating
            # between PPO updates and constraint-network updates (when ICRL is used).
            # The inner-loop represents the episodes each of which consists of
            # (around) "forward_steps" environment steps.
            # The total number of the experiment is given by n_iters*forward_steps.

            for iter_ in range(self.n_iters):

                if self.icrl is True:
                    # Used to update the learning rate of the constraint-network.
                    self.current_progress_remaining = 1 - float(iter_) / float(self.n_iters)

                cur_iter_steps = 0

                # Loop of episodes where samples are collected and used for PPO updates
                while cur_iter_steps < self.forward_steps:

                    self.i_episode += 1
                    self.episode_step_counter = 0
                    self.episode_game_counter = 0
                    self.episode_reward = 0
                    if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                        self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list = []
                    if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                        self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list = []
                    if self.constr_ball_not_in_circle is True:
                        self.ball_not_in_circle_episode_constraint_violation_list = []

                    self.reset_buffer()

                    print("\nEpisode: " + str(self.i_episode))

                    # Loop of steps for each episode-game. At the end of each episode
                    # we perform "epochs" number of PPO updates
                    while self.episode_step_counter < self.n_steps:

                        self.episode_game_counter += 1
                        self.total_games += 1

                        # In the beginning of each game, initialize the environment and the variables
                        self.initialize_game_var_train()

                        observation_tensor = self.obs_to_FT(self.observation)

                        # Loop of the game
                        while not self.done:

                            self.train_step_start_time = time.time()
                            self.total_steps += 1
                            cur_iter_steps += 1
                            self.episode_step_counter += 1
                            self.train_step_counter += 1

                            # Get actions
                            env_X_agent_action, env_Y_agent_action, extra_returns = \
                                self.get_agent_action(observation_tensor, greedy=False, random_=None)

                            pred_cost = [None, None]
                            if self.icrl:
                                # Get predicted cost from the constraint-net
                                pred_cost = self.get_predicted_cost(observation_tensor, extra_returns)

                            # Environment step
                            transition = self.env.step(
                                [env_X_agent_action, env_Y_agent_action],
                                self.timed_out,
                                self.goal,
                                self.action_iterations
                            )
                            (
                                self.next_observation,
                                self.reward,
                                self.done,
                                train_fps,
                                self.duration_pause,
                                self.action_list
                            ) = transition

                            # Update variables after step
                            self.train_update_per_step_vars()

                            # add experiences to buffers
                            self.save_experience([observation_tensor, self.done, self.reward, pred_cost] + extra_returns)

                            # set the observation for the next step
                            self.observation = self.next_observation.copy()
                            observation_tensor = self.obs_to_FT(self.next_observation)

                        ## End of game

                        # Update and store info for the current game, print avg logs
                        self.train_game_logging()

                        # Testing
                        self.test_during_training()

                    ## End of the episode

                    # At the end of each episode rollouts compute discounted returns and advantages
                    self.handle_buffer(end_game=True)
                    # PPO update networks, store and print useful info
                    self.train_store_and_print_info()

                # End of PPO episodes for the current iteration

                if self.icrl is True:

                    observations_bw, actions_bw, lengths_bw = [], [], []

                    # Loop of backward games (to collect samples for training constraint-net)
                    for bw_games in tqdm(range(self.ICRL_backward_n_rollouts), desc='Backward game: '):

                        done_bw, timed_out_bw, step_counter_bw = False, False, 0
                        observation_bw, _ = self.env.reset(seed=None, initialize_seed=True if bw_games == 0 else False)
                        observation_tensor_bw = self.obs_to_FT(observation_bw)

                        # Loop of a game of backward iterations
                        while not done_bw:

                            step_counter_bw += 1

                            # Get actions
                            env_X_agent_action_bw, env_Y_agent_action_bw, extra_returns_bw = \
                                self.get_agent_action(observation_tensor_bw, greedy=False, random_=None)

                            # Store samples
                            observations_bw.append(
                                normalize_features(observation_bw.copy()[:8]) if self.normalize_features \
                                                                              else
                                observation_bw.copy()[:8]
                            )
                            actions_bw.append(
                                extra_returns_bw[0].item() if self.X_Y_agent is not None
                                                           else
                                [extra_returns_bw[0], extra_returns_bw[1]]
                            )

                            # Environment step
                            transition = self.env.step(
                                [env_X_agent_action_bw, env_Y_agent_action_bw],
                                timed_out_bw,
                                self.goal,
                                self.action_iterations
                            )
                            next_observation_bw, _, done_bw, _, _, _ = transition

                            # check if the game has exceeded the maximum timesteps per game
                            if step_counter_bw >= self.max_timesteps_per_game:
                                timed_out_bw = True

                            # set the observation for the next step
                            observation_bw = next_observation_bw.copy()
                            observation_tensor_bw = self.obs_to_FT(next_observation_bw)

                        lengths_bw.append(step_counter_bw)

                    # End of backwards rollouts
                    self.train_constraint_net(observations_bw, actions_bw, lengths_bw)

            # End of experiment
            self.max_games = self.total_games

            # Stop visualization
            if self.env.render:
                self.env.pg.quit()

    def train_constraint_net(self, obs, acts, lens):

        assert self.algo == 'PPO', ""

        obs = np.array(obs)

        if self.X_agent is not None or self.Y_agent is not None:
            if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                X_acts = np.array([actions[0].item() for actions in acts])
                X_total_loss_constr_net, \
                X_expert_loss_constr_net, \
                X_policy_loss_constr_net_wo_is, \
                X_policy_loss_constr_net, \
                X_regularizer_loss_constr_net, \
                X_is_weights_mean_constr_net, \
                X_is_weights_max_constr_net, \
                X_is_weights_min_constr_net, \
                X_policy_preds_max_constr_net, \
                X_policy_preds_min_constr_net, \
                X_policy_preds_mean_constr_net, \
                X_expert_preds_max_constr_net, \
                X_expert_preds_min_constr_net, \
                X_expert_preds_mean_constr_net, \
                X_kl_old_new_constr_net, \
                X_kl_new_old_constr_net, \
                X_early_stop_itr_constr_net = \
                    self.X_agent.constraint_net.train_(
                        self.ICRL_backward_iterations,
                        obs,
                        X_acts,
                        lens,
                        self.current_progress_remaining
                    )
                print("\nX Total Loss Constraint-net: {}\n"
                      "X Expert Loss Constraint-net: {}\n"
                      "X Policy Loss Constraint-net wo IS: {}\n"
                      "X Policy Loss Constraint-net: {}\n"
                      "X Regularizer Loss: {}\n"
                      "X IS weights Constraint-net Mean: {}\n"
                      "X IS weights Constraint-net Max: {}\n"
                      "X IS weights Constraint-net Mix: {}\n"
                      "X Policy Preds Constraint-net Mean: {}\n"
                      "X Policy Preds Constraint-net Max: {}\n"
                      "X Policy Preds Constraint-net Min: {}\n"
                      "X Experts Preds Constraint-net Mean: {}\n"
                      "X Experts Preds Constraint-net Max: {}\n"
                      "X Experts Preds Constraint-net Min: {}\n"
                      "X KL Div Old-New: {}\n"
                      "X KL Div New-Old: {}\n"
                      "X Early Stop Iter: {}\n"
                      .format(round(X_total_loss_constr_net, 2),
                              round(X_expert_loss_constr_net, 2),
                              round(X_policy_loss_constr_net_wo_is, 2),
                              round(X_policy_loss_constr_net, 2),
                              round(X_regularizer_loss_constr_net, 2),
                              round(X_is_weights_mean_constr_net, 2),
                              round(X_is_weights_max_constr_net, 2),
                              round(X_is_weights_min_constr_net, 2),
                              round(X_policy_preds_mean_constr_net, 2),
                              round(X_policy_preds_max_constr_net, 2),
                              round(X_policy_preds_min_constr_net, 2),
                              round(X_expert_preds_mean_constr_net, 2),
                              round(X_expert_preds_max_constr_net, 2),
                              round(X_expert_preds_min_constr_net, 2),
                              round(X_kl_old_new_constr_net, 2),
                              round(X_kl_new_old_constr_net, 2),
                              round(X_early_stop_itr_constr_net, 2)))
                self.X_total_loss_constr_net_per_iter_list.append(X_total_loss_constr_net)
                self.X_expert_loss_constr_net_per_iter_list.append(X_expert_loss_constr_net)
                self.X_policy_loss_constr_net_wo_is_per_iter_list.append(X_policy_loss_constr_net_wo_is)
                self.X_policy_loss_constr_net_per_iter_list.append(X_policy_loss_constr_net)
                self.X_regularizer_loss_constr_net_per_iter_list.append(X_regularizer_loss_constr_net)
                self.X_is_weights_mean_constr_net_per_iter_list.append(X_is_weights_mean_constr_net)
                self.X_is_weights_max_constr_net_per_iter_list.append(X_is_weights_max_constr_net)
                self.X_is_weights_min_constr_net_per_iter_list.append(X_is_weights_min_constr_net)
                self.X_policy_preds_max_constr_net_per_iter_list.append(X_policy_preds_max_constr_net)
                self.X_policy_preds_min_constr_net_per_iter_list.append(X_policy_preds_min_constr_net)
                self.X_policy_preds_mean_constr_net_per_iter_list.append(X_policy_preds_mean_constr_net)
                self.X_expert_preds_max_constr_net_per_iter_list.append(X_expert_preds_max_constr_net)
                self.X_expert_preds_min_constr_net_per_iter_list.append(X_expert_preds_min_constr_net)
                self.X_expert_preds_mean_constr_net_per_iter_list.append(X_expert_preds_mean_constr_net)
                self.X_kl_old_new_constr_net_per_iter_list.append(X_kl_old_new_constr_net)
                self.X_kl_new_old_constr_net_per_iter_list.append(X_kl_new_old_constr_net)
                self.X_early_stop_itr_constr_net_per_iter_list.append(X_early_stop_itr_constr_net)
            if self.Y_agent is not None and self.Y_agent is not None:
                Y_acts = np.array([actions[1].item() for actions in acts])
                Y_total_loss_constr_net, \
                Y_expert_loss_constr_net, \
                Y_policy_loss_constr_net_wo_is, \
                Y_policy_loss_constr_net, \
                Y_regularizer_loss_constr_net, \
                Y_is_weights_mean_constr_net, \
                Y_is_weights_max_constr_net, \
                Y_is_weights_min_constr_net, \
                Y_policy_preds_max_constr_net, \
                Y_policy_preds_min_constr_net, \
                Y_policy_preds_mean_constr_net, \
                Y_expert_preds_max_constr_net, \
                Y_expert_preds_min_constr_net, \
                Y_expert_preds_mean_constr_net, \
                Y_kl_old_new_constr_net, \
                Y_kl_new_old_constr_net, \
                Y_early_stop_itr_constr_net = \
                    self.Y_agent.constraint_net.train_(
                        self.ICRL_backward_iterations,
                        obs,
                        Y_acts,
                        lens,
                        self.current_progress_remaining
                    )
                print("\nY Total Loss Constraint-net: {}\n"
                      "Y Expert Loss Constraint-net: {}\n"
                      "Y Policy Loss Constraint-net wo IS: {}\n"
                      "Y Policy Loss Constraint-net: {}\n"
                      "Y Regularizer Loss: {}\n"
                      "Y IS weights Constraint-net Mean: {}\n"
                      "Y IS weights Constraint-net Max: {}\n"
                      "Y IS weights Constraint-net Mix: {}\n"
                      "Y Policy Preds Constraint-net Mean: {}\n"
                      "Y Policy Preds Constraint-net Max: {}\n"
                      "Y Policy Preds Constraint-net Min: {}\n"
                      "Y Experts Preds Constraint-net Mean: {}\n"
                      "Y Experts Preds Constraint-net Max: {}\n"
                      "Y Experts Preds Constraint-net Min: {}\n"
                      "Y KL Div Old-New: {}\n"
                      "Y KL Div New-Old: {}\n"
                      "Y Early Stop Iter: {}\n"
                      .format(round(Y_total_loss_constr_net, 2),
                              round(Y_expert_loss_constr_net, 2),
                              round(Y_policy_loss_constr_net_wo_is, 2),
                              round(Y_policy_loss_constr_net, 2),
                              round(Y_regularizer_loss_constr_net, 2),
                              round(Y_is_weights_mean_constr_net, 2),
                              round(Y_is_weights_max_constr_net, 2),
                              round(Y_is_weights_min_constr_net, 2),
                              round(Y_policy_preds_mean_constr_net, 2),
                              round(Y_policy_preds_max_constr_net, 2),
                              round(Y_policy_preds_min_constr_net, 2),
                              round(Y_expert_preds_mean_constr_net, 2),
                              round(Y_expert_preds_max_constr_net, 2),
                              round(Y_expert_preds_min_constr_net, 2),
                              round(Y_kl_old_new_constr_net, 2),
                              round(Y_kl_new_old_constr_net, 2),
                              round(Y_early_stop_itr_constr_net, 2))
                      )
                self.Y_total_loss_constr_net_per_iter_list.append(Y_total_loss_constr_net)
                self.Y_expert_loss_constr_net_per_iter_list.append(Y_expert_loss_constr_net)
                self.Y_policy_loss_constr_net_wo_is_per_iter_list.append(Y_policy_loss_constr_net_wo_is)
                self.Y_policy_loss_constr_net_per_iter_list.append(Y_policy_loss_constr_net)
                self.Y_regularizer_loss_constr_net_per_iter_list.append(Y_regularizer_loss_constr_net)
                self.Y_is_weights_mean_constr_net_per_iter_list.append(Y_is_weights_mean_constr_net)
                self.Y_is_weights_max_constr_net_per_iter_list.append(Y_is_weights_max_constr_net)
                self.Y_is_weights_min_constr_net_per_iter_list.append(Y_is_weights_min_constr_net)
                self.Y_policy_preds_max_constr_net_per_iter_list.append(Y_policy_preds_max_constr_net)
                self.Y_policy_preds_min_constr_net_per_iter_list.append(Y_policy_preds_min_constr_net)
                self.Y_policy_preds_mean_constr_net_per_iter_list.append(Y_policy_preds_mean_constr_net)
                self.Y_expert_preds_max_constr_net_per_iter_list.append(Y_expert_preds_max_constr_net)
                self.Y_expert_preds_min_constr_net_per_iter_list.append(Y_expert_preds_min_constr_net)
                self.Y_expert_preds_mean_constr_net_per_iter_list.append(Y_expert_preds_mean_constr_net)
                self.Y_kl_old_new_constr_net_per_iter_list.append(Y_kl_old_new_constr_net)
                self.Y_kl_new_old_constr_net_per_iter_list.append(Y_kl_new_old_constr_net)
                self.Y_early_stop_itr_constr_net_per_iter_list.append(Y_early_stop_itr_constr_net)
        elif self.X_Y_agent is not None:
                X_Y_acts = np.array(acts)
                X_Y_total_loss_constr_net, \
                X_Y_expert_loss_constr_net, \
                X_Y_policy_loss_constr_net_wo_is, \
                X_Y_policy_loss_constr_net, \
                X_Y_regularizer_loss_constr_net, \
                X_Y_is_weights_mean_constr_net, \
                X_Y_is_weights_max_constr_net, \
                X_Y_is_weights_min_constr_net, \
                X_Y_policy_preds_max_constr_net, \
                X_Y_policy_preds_min_constr_net, \
                X_Y_policy_preds_mean_constr_net, \
                X_Y_expert_preds_max_constr_net, \
                X_Y_expert_preds_min_constr_net, \
                X_Y_expert_preds_mean_constr_net, \
                X_Y_kl_old_new_constr_net, \
                X_Y_kl_new_old_constr_net, \
                X_Y_early_stop_itr_constr_net = \
                    self.X_Y_agent.constraint_net.train_(
                        self.ICRL_backward_iterations,
                        obs,
                        X_Y_acts,
                        lens,
                        self.current_progress_remaining
                    )
                print("\nX_Y Total Loss Constraint-net: {}\n"
                      "X_Y Expert Loss Constraint-net: {}\n"
                      "X_Y Policy Loss Constraint-net wo IS: {}\n"
                      "X_Y Policy Loss Constraint-net: {}\n"
                      "X_Y Regularizer Loss: {}\n"
                      "X_Y IS weights Constraint-net Mean: {}\n"
                      "X_Y IS weights Constraint-net Max: {}\n"
                      "X_Y IS weights Constraint-net Mix: {}\n"
                      "X_Y Policy Preds Constraint-net Mean: {}\n"
                      "X_Y Policy Preds Constraint-net Max: {}\n"
                      "X_Y Policy Preds Constraint-net Min: {}\n"
                      "X_Y Experts Preds Constraint-net Mean: {}\n"
                      "X_Y Experts Preds Constraint-net Max: {}\n"
                      "X_Y Experts Preds Constraint-net Min: {}\n"
                      "X_Y KL Div Old-New: {}\n"
                      "X_Y KL Div New-Old: {}\n"
                      "X_Y Early Stop Iter: {}\n"
                      .format(round(X_Y_total_loss_constr_net, 2),
                              round(X_Y_expert_loss_constr_net, 2),
                              round(X_Y_policy_loss_constr_net_wo_is, 2),
                              round(X_Y_policy_loss_constr_net, 2),
                              round(X_Y_regularizer_loss_constr_net, 2),
                              round(X_Y_is_weights_mean_constr_net, 2),
                              round(X_Y_is_weights_max_constr_net, 2),
                              round(X_Y_is_weights_min_constr_net, 2),
                              round(X_Y_policy_preds_mean_constr_net, 2),
                              round(X_Y_policy_preds_max_constr_net, 2),
                              round(X_Y_policy_preds_min_constr_net, 2),
                              round(X_Y_expert_preds_mean_constr_net, 2),
                              round(X_Y_expert_preds_max_constr_net, 2),
                              round(X_Y_expert_preds_min_constr_net, 2),
                              round(X_Y_kl_old_new_constr_net, 2),
                              round(X_Y_kl_new_old_constr_net, 2),
                              round(X_Y_early_stop_itr_constr_net, 2)))
                self.X_Y_total_loss_constr_net_per_iter_list.append(X_Y_total_loss_constr_net)
                self.X_Y_expert_loss_constr_net_per_iter_list.append(X_Y_expert_loss_constr_net)
                self.X_Y_policy_loss_constr_net_wo_is_per_iter_list.append(X_Y_policy_loss_constr_net_wo_is)
                self.X_Y_policy_loss_constr_net_per_iter_list.append(X_Y_policy_loss_constr_net)
                self.X_Y_regularizer_loss_constr_net_per_iter_list.append(X_Y_regularizer_loss_constr_net)
                self.X_Y_is_weights_mean_constr_net_per_iter_list.append(X_Y_is_weights_mean_constr_net)
                self.X_Y_is_weights_max_constr_net_per_iter_list.append(X_Y_is_weights_max_constr_net)
                self.X_Y_is_weights_min_constr_net_per_iter_list.append(X_Y_is_weights_min_constr_net)
                self.X_Y_policy_preds_max_constr_net_per_iter_list.append(X_Y_policy_preds_max_constr_net)
                self.X_Y_policy_preds_min_constr_net_per_iter_list.append(X_Y_policy_preds_min_constr_net)
                self.X_Y_policy_preds_mean_constr_net_per_iter_list.append(X_Y_policy_preds_mean_constr_net)
                self.X_Y_expert_preds_max_constr_net_per_iter_list.append(X_Y_expert_preds_max_constr_net)
                self.X_Y_expert_preds_min_constr_net_per_iter_list.append(X_Y_expert_preds_min_constr_net)
                self.X_Y_expert_preds_mean_constr_net_per_iter_list.append(X_Y_expert_preds_mean_constr_net)
                self.X_Y_kl_old_new_constr_net_per_iter_list.append(X_Y_kl_old_new_constr_net)
                self.X_Y_kl_new_old_constr_net_per_iter_list.append(X_Y_kl_new_old_constr_net)
                self.X_Y_early_stop_itr_constr_net_per_iter_list.append(X_Y_early_stop_itr_constr_net)

    def obs_to_FT(self, obs):
        """
        Transforms 'observations' numpy array to FloatTensor after normalizing them (if 'self.normalize_features' is True).
        @param obs: Observation in numpy array format.
        @return: Observations in FloatTensor Format, normalized if needed.
        """

        obs_ = normalize_features(obs.copy()) if self.normalize_features else obs.copy()
        FT_obs = torch.from_numpy(np.array([obs_], dtype=np.float32)).float().to(self.device)
        return FT_obs

    def train_cogail_agents(self):
        """
        Code based on: https://github.com/j96w/cogail/blob/main/scripts/train.py
        """

        # Perform BC to initialize the policy as close as possible to that of demonstrators
        self.pi_co.actor_critic.train()
        for j in tqdm(range(self.bc_pretrain_steps), desc='GAIL pre-training epochs'):
            loss = self.pi_co.pretrain(self.discr.train_loader)
            self.bc_loss_per_update.append(loss)
            print("\nPretrain round {}: loss {}".format(j, loss))

        ## co-GAIL Training Loop

        # In this experiment, each episode consists of 'num_rollouts_steps',
        # so we have an inner loop iterating over games.
        for self.i_episode in itertools.count(1):

            self.episode_game_counter = 0
            self.episode_step_counter = 0
            self.episode_reward = 0
            self.code_variable_per_game_list = []
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list = []
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list = []
            if self.constr_ball_not_in_circle is True:
                self.ball_not_in_circle_episode_constraint_violation_list = []

            print("\nEpisode: " + str(self.i_episode))

            # Update learning rate of policy
            self.pi_co.update_linear_schedule(self.pi_co.optimizer, self.i_episode, self.num_updates, self.pi_co.lr)

            # Games loop
            for _ in itertools.count(1):

                self.episode_game_counter += 1
                self.total_games += 1

                # At the beginning of each game, initialize the environment and the variables
                self.initialize_game_var_train()

                observation_tensor = self.obs_to_FT(self.observation)
                # In each environment reset, we get a new gode to generate trajectories.
                random_seed_tensor = self.code_variable_train.get_next_code().unsqueeze(dim=0).to(self.agents.device)
                # save the first observations and random seed of each game to storage using the right index
                self.handle_buffer(start_game=True, data=[observation_tensor, random_seed_tensor])
                # store the game code at the corresponding list
                self.code_variable_per_game_list.append(random_seed_tensor.detach().cpu().numpy())

                # Start to play a game.
                # When the ball has either reached the goal or the game has timed out, stop the current game.
                while not self.done:

                    self.train_step_start_time = time.time()
                    self.total_steps += 1  # keep track of the overall step number
                    self.train_step_counter += 1  # keep track of the step number for each game
                    self.episode_step_counter += 1  # keep track of the step number for each episode

                    # Get actions
                    env_X_agent_action, env_Y_agent_action, real_X_agent_action, real_Y_agent_action, \
                        value_tensor, actions_tensor, action_log_prob_tensor = \
                        self.get_agent_action(
                            [observation_tensor, random_seed_tensor],
                            False,
                            None
                        )

                    # Environment step
                    transition = self.env.step(
                        [env_X_agent_action, env_Y_agent_action],
                        self.timed_out,
                        self.goal,
                        self.action_iterations
                    )
                    self.next_observation, self.reward, self.done, train_fps, self.duration_pause, self.action_list = \
                        transition

                    # Update variables after step
                    self.train_update_per_step_vars()

                    # add experiences to buffers
                    data = [
                        self.next_observation,
                        self.done,
                        self.fixed_done,
                        self.reward,
                        value_tensor,
                        actions_tensor,
                        action_log_prob_tensor,
                        random_seed_tensor
                    ]
                    self.save_experience(data)

                    # set the observation for the next step
                    self.observation = self.next_observation.copy()
                    observation_tensor = self.obs_to_FT(self.next_observation)

                ## End of game

                # Update and store info for the current game, print avg logs
                self.train_game_logging()

                # Testing
                self.test_during_training()

                # End of episode rollouts
                if self.episode_step_counter >= self.num_rollouts_steps:
                    # At the end of each episode, rollouts compute and store the last predicted value
                    self.handle_buffer(end_game=True)
                    break

            ### End of the episode

            # Update networks, store and print useful info
            self.train_store_and_print_info()

            # Reset buffer
            self.reset_buffer()

            if self.total_steps >= self.num_steps:
                break

        self.max_games = self.total_games

        # Stop visualization
        if self.env.render:
            self.env.pg.quit()

    def train_agent0s0_sac(self):

        assert not ((self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y' and
                     self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y') and
                    (self.X_agent.memory.memory_size != self.Y_agent.memory.memory_size)
                    )

        if self.SL_finetune is True:
            self.SL_finetuning_agent0s0()

        else:
            """
            Code based on: https://github.com/pranz24/pytorch-soft-actor-critic/blob/SAC_V/main.py
            """

            # RL training Loop
            for self.i_episode in itertools.count(1):

                # At the beginning of each game, initialize the environment and the variables
                self.initialize_game_var_train()

                print("\nEpisode: " + str(self.i_episode))

                # Start to play a game.
                # When the ball has either reached the goal or the game has timed out, stop the current game.
                while not self.done:

                    self.train_step_start_time = time.time()
                    # keep track of the overall step number
                    self.total_steps += 1
                    # keep track of the step number for each game
                    self.train_step_counter += 1

                    # Get actions
                    env_X_agent_action, env_Y_agent_action, real_X_agent_action, real_Y_agent_action = \
                        self.get_agent_action(
                            normalize_features(self.observation) if self.normalize_features
                                                                 else
                                               self.observation,
                            False,
                            None if self.start_steps < self.total_steps
                                 else
                            True
                        )

                    # Environment step
                    transition = self.env.step(
                        [env_X_agent_action, env_Y_agent_action],
                        self.timed_out,
                        self.goal,
                        self.action_iterations
                    )
                    self.next_observation, self.reward, self.done, train_fps, self.duration_pause, self.action_list = \
                        transition

                    # Update variables after step
                    self.train_update_per_step_vars()

                    # add experiences to buffers
                    data = [
                        self.observation,
                        self.next_observation,
                        real_X_agent_action,
                        real_Y_agent_action,
                        self.reward,
                        self.fixed_done
                    ]
                    self.save_experience(data)

                    # set the observation for the next step
                    self.observation = self.next_observation.copy()

                    # Train the networks and store the corresponding info
                    self.train_store_and_print_info()

                ## End of game

                # Update, store and print info for the current game. Also, print avg logs
                self.train_game_logging()

                # Testing
                self.test_during_training()

                # Stop training based on the specified condition
                if (isinstance(self.num_steps, int) and self.total_steps >= self.num_steps) or \
                   (isinstance(self.n_episodes, int) and self.i_episode >= self.n_episodes) or \
                   (isinstance(self.n_train_episodes, int) and self.i_train_episode >= self.n_train_episodes):
                    break

            self.max_games = self.i_episode

            # Stop visualization
            if self.env.render:
                self.env.pg.quit()

    def handle_buffer(self, start_game=False, end_game=False, data=None):
        assert (start_game is True or end_game is True) and not (start_game is True and end_game is True), ""

        if self.algo == 'coGAIL':
            if start_game is True:
                observation_tensor = data[0]
                random_seed_tensor = data[1]
                self.pi_co.rollout_storage.obs[self.pi_co.rollout_storage.step].copy_(
                    observation_tensor[0, :8]
                )
                self.pi_co.rollout_storage.random_seed[self.pi_co.rollout_storage.step].copy_(
                    random_seed_tensor.squeeze(dim=0)
                )
                if self.pi_co.rollout_storage.step == 0:
                    self.pi_co.rollout_storage.masks[0] = \
                        torch.from_numpy(np.array([0.0], dtype=np.float32)).float()
                    self.pi_co.rollout_storage.bad_masks[0] = \
                        torch.from_numpy(np.array([0.0], dtype=np.float32)).float()
            elif end_game is True:
                with torch.no_grad():
                    next_value = self.actor_critic.get_value(
                        self.pi_co.rollout_storage.obs[self.pi_co.rollout_storage.step].unsqueeze(dim=0),
                        self.pi_co.rollout_storage.random_seed[self.pi_co.rollout_storage.step].unsqueeze(dim=0)
                    )
                    env_next_value_tensor = None if not self.pi_co.rollout_storage.opt_robot_w_env_rewards \
                                                 else \
                                            next_value[:, 1]
                    next_value_tensor = next_value.squeeze(0) if not self.pi_co.rollout_storage.opt_robot_w_env_rewards \
                                                              else \
                                        next_value[:, 0]
                    self.pi_co.rollout_storage.value_preds[self.pi_co.rollout_storage.step].copy_(
                        next_value_tensor
                    )
                    self.pi_co.rollout_storage.env_value_preds[self.pi_co.rollout_storage.step].copy_(
                        env_next_value_tensor
                    )
            else:
                raise NotImplementedError

        elif self.algo == 'PPO':
            if end_game is True:
                if self.X_agent is not None or self.Y_agent is not None:
                    if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                        self.X_agent.ppo.rollout_buffer.compute_returns_and_advantage()
                    if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                        self.Y_agent.ppo.rollout_buffer.compute_returns_and_advantage()
                elif self.X_Y_agent is not None:
                    self.X_Y_agent.ppo.rollout_buffer.compute_returns_and_advantage()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def get_predicted_cost(self, observations: torch.Tensor, action_list: List) \
            -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Estimate and return the cost
        @param observations: Observations in torch.Tensor format with shape=[1, 8]
        @param action_list: List of action(s). In case of a single agent, we should get the item with index 0.
                            In case of two agents, we should get the items with indices 0 and 1. Each action is
                            a torch.Tensor with shape=[1, 1].
        @return: Predicted cost in np.array format with shape=[1, 1] if only one agent,
                 else a list with two np.array each with shape=[1, 1].
        """

        assert len(observations.size()) == 2 and observations.size(0) == 1, ""
        observations = observations.detach().cpu().numpy()[:, :8]

        if self.algo == 'PPO' and self.icrl is True:
            if self.X_agent is not None or self.Y_agent is not None:
                X_agent_cost = None
                Y_agent_cost = None
                if self.X_agent is not None:
                    if self.X_agent.axis_agent == 'X_Y':
                        assert len(action_list[0].size()) == 1 and action_list[0].size(0) == 1 and \
                               len(action_list[1].size()) == 1 and action_list[1].size(0) == 1, ""
                        # In this case, we should get the action of X-axis from this agent and
                        # the action of Y-axis from the other agent, combine them, and feed them
                        # to the constraint-net.
                        X_axis_action = transform_action_to_actions(
                            action_list[0].item(),
                            self.env.action_space.actions_number
                        )[0]
                        Y_axis_action = action_list[1].item()
                        action = transform_actions_to_action(
                            [X_axis_action, Y_axis_action],
                            self.env.action_space.actions_number
                        )
                        X_agent_cost = self.X_agent.constraint_net.cost_function(observations, np.array([action]))
                    else:
                        assert len(action_list[0].size()) == 1 and action_list[0].size(0) == 1, ""
                        X_agent_cost = self.X_agent.constraint_net.cost_function(
                            observations,
                            action_list[0].detach().cpu().numpy()
                        )
                if self.Y_agent is not None:
                    if self.Y_agent.axis_agent == 'X_Y':
                        assert len(action_list[0].size()) == 1 and action_list[0].size(0) == 1 and \
                               len(action_list[1].size()) == 1 and action_list[1].size(0) == 1, ""
                        # In this case, we should get the action of Y-axis from this agent and
                        # the action of X-axis from the other agent, combine them, and feed them
                        # to the constraint-net.
                        X_axis_action = action_list[0].item()
                        Y_axis_action = transform_action_to_actions(
                            action_list[1].item(),
                            self.env.action_space.actions_number
                        )[0]
                        action = transform_actions_to_action(
                            [X_axis_action, Y_axis_action],
                            self.env.action_space.actions_number
                        )
                        Y_agent_cost = self.Y_agent.constraint_net.cost_function(
                            observations,
                            np.array([action])
                        )
                    else:
                        assert len(action_list[1].size()) == 1 and action_list[1].size(0) == 1, ""
                        Y_agent_cost = self.Y_agent.constraint_net.cost_function(
                            observations,
                            action_list[1].detach().cpu().numpy()
                        )
                return [X_agent_cost, Y_agent_cost]

            elif self.X_Y_agent is not None:
                assert len(action_list[0].size()) == 1 and action_list[0].size(0) == 1, ""
                X_Y_agent_cost = self.X_Y_agent.constraint_net.cost_function(
                    observations,
                    action_list[0].detach().cpu().numpy()
                )
                return X_Y_agent_cost
        else:
            raise NotImplementedError

    def train_update_per_step_vars(self):

        # calculate game duration
        train_step_duration = time.time() - self.train_step_start_time - self.duration_pause
        self.train_step_duration_list.append(train_step_duration)

        # keep track of the total paused time
        self.redundant_end_duration += self.duration_pause

        # keep track of the total game reward
        self.game_reward += self.reward
        if self.algo == 'coGAIL' or self.algo == 'PPO':
            # keep track of the total episode reward
            self.episode_reward += self.reward

        # check if the game has exceeded the maximum timesteps per game
        if self.train_step_counter >= self.max_timesteps_per_game:
            self.timed_out = True

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)
        self.fixed_done = 0. if self.train_step_counter == self.max_timesteps_per_game else float(self.done)

        # compute travelled distance
        self.dist_travel = get_distance_traveled(self.dist_travel, self.observation, self.next_observation)

        # Update number of constraint violations
        if self.constr_ball_only_at_the_right_side_wrt_hole is True or \
           self.constr_ball_only_at_the_up_side_wrt_hole is True or \
           self.constr_ball_not_in_circle is True:
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                assert self.next_observation.shape[0] >= 9, ""
                ball_only_at_the_right_side_wrt_hole_constraint_violation = \
                    1 if self.next_observation[8] == self.not_satisfied_constraint_value \
                      else \
                    0
                self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list.append(
                    ball_only_at_the_right_side_wrt_hole_constraint_violation
                )
                if self.algo == 'coGAIL' or self.algo == 'PPO':
                    self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list.append(
                        ball_only_at_the_right_side_wrt_hole_constraint_violation
                    )
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                    assert self.next_observation.shape[0] >= 10, ""
                    ball_only_at_the_up_side_wrt_hole_constraint_violation = \
                        1 if self.next_observation[9] == self.not_satisfied_constraint_value \
                          else \
                        0
                else:
                    assert self.next_observation.shape[0] >= 9, ""
                    ball_only_at_the_up_side_wrt_hole_constraint_violation = \
                        1 if self.next_observation[8] == self.not_satisfied_constraint_value \
                          else \
                        0
                self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list.append(
                    ball_only_at_the_up_side_wrt_hole_constraint_violation
                )
                if self.algo == 'coGAIL' or self.algo == 'PPO':
                    self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list.append(
                        ball_only_at_the_up_side_wrt_hole_constraint_violation
                    )
            if self.constr_ball_not_in_circle is True:
                if self.constr_ball_only_at_the_right_side_wrt_hole is True and \
                   self.constr_ball_only_at_the_up_side_wrt_hole is True:
                    assert self.next_observation.shape[0] >= 11, ""
                    ball_not_in_circle_constraint_violation = \
                        1 if self.next_observation[10] == self.not_satisfied_constraint_value \
                          else \
                        0
                elif (self.constr_ball_only_at_the_right_side_wrt_hole is True and
                      self.constr_ball_only_at_the_up_side_wrt_hole is False) or \
                     (self.constr_ball_only_at_the_right_side_wrt_hole is False and
                      self.constr_ball_only_at_the_up_side_wrt_hole is True):
                    assert self.next_observation.shape[0] >= 10, ""
                    ball_not_in_circle_constraint_violation = \
                        1 if self.next_observation[9] == self.not_satisfied_constraint_value \
                          else \
                        0
                elif self.constr_ball_only_at_the_right_side_wrt_hole is False and \
                     self.constr_ball_only_at_the_up_side_wrt_hole is False:
                    assert self.next_observation.shape[0] >= 9, ""
                    ball_not_in_circle_constraint_violation = \
                        1 if self.next_observation[8] == self.not_satisfied_constraint_value \
                          else \
                        0
                else:
                    raise NotImplementedError
                self.ball_not_in_circle_constraint_violation_list.append(
                    ball_not_in_circle_constraint_violation
                )
                if self.algo == 'coGAIL' or self.algo == 'PPO':
                    self.ball_not_in_circle_episode_constraint_violation_list.append(
                        ball_not_in_circle_constraint_violation
                    )

        if self.debug_ is True:
            # append row to the dataframe
            new_row = get_row_to_store(self.next_observation, float(self.done), self.fixed_done)
            self.df = self.df.append(new_row, ignore_index=True)
            # Store the actions
            self.action_history = self.action_history + self.action_list

    def test_during_training(self):

        if self.algo == "SAC":
            flag_for_test = self.SL_finetune is True or self.i_episode % self.test_every_episodes == 0 or self.i_episode == 1
        elif self.algo == "coGAIL" or self.algo == "PPO":
            flag_for_test = \
                self.total_games % self.test_every_games == 0 or \
                self.total_games == 1 or \
                (self.algo == "PPO" and self.SL_finetune is True)
            if flag_for_test is True and self.algo == "coGAIL":
                # Reset code
                self.code_variable_test.reset_pivot()
        else:
            raise NotImplementedError

        ## The following is common for all algos
        if flag_for_test is True:

            self.test_game_number += 1  # keep track of the testing session number
            print('\nTest {}'.format(self.test_game_number) + '\n')

            ## Run test
            # Evaluation mode
            with torch.no_grad():
                self.eval_mode()
                # Test
                self.test_process()
            # Return to train mode
            self.train_mode()

            # Update per test metrics
            self.test_game_duration_list_avg_per_test.append(
                mean(self.test_game_duration_list[-self.test_max_games:])
            )
            test_avg_reward = mean(self.test_reward_list[-self.test_max_games:])
            self.test_reward_list_avg_per_test.append(test_avg_reward)
            self.test_distance_travel_list_avg_per_test.append(
                mean(self.test_distance_travel_list[-self.test_max_games:])
            )
            self.test_length_list_avg_per_test.append(
                mean(self.test_length_list[-self.test_max_games:])
            )
            test_avg_num_constr = 0
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg = mean(
                    self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:]
                )
                test_avg_num_constr += test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg
                self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list_avg_per_test.append(
                    test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg
                )
                self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_test.append(
                    mean(self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
                )
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg = mean(
                    self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:]
                )
                test_avg_num_constr += test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg
                self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list_avg_per_test.append(
                    test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg
                )
                self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_test.append(
                    mean(self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
                )
            if self.constr_ball_not_in_circle is True:
                test_ball_not_in_circle_num_constraint_violated_avg = mean(
                    self.test_ball_not_in_circle_num_constraint_violated_list[-self.test_max_games:]
                )
                test_avg_num_constr += test_ball_not_in_circle_num_constraint_violated_avg
                self.test_ball_not_in_circle_num_constraint_violated_list_avg_per_test.append(
                    test_ball_not_in_circle_num_constraint_violated_avg
                )
                self.test_ball_not_in_circle_freq_constraint_violated_list_avg_per_test.append(
                    mean(self.test_ball_not_in_circle_freq_constraint_violated_list[-self.test_max_games:])
                )

            ## Save the models
            if self.save is True:
                if (self.save_last is False and self.save_constraints is False and
                    ((self.algo != "SAC" and not (self.algo == "PPO" and self.SL_finetune is True)) or
                     (self.algo in ["SAC", "PPO"] and
                      (self.SL_finetune is False or (self.SL_finetune is True and self.save_all is False))
                     )
                    ) and test_avg_reward > self.test_avg_best_score
                ) or \
                    self.save_last is True or \
                    (self.save_constraints is True and test_avg_num_constr < self.test_avg_lowest_num_constr) or \
                    (self.algo in ["SAC", "PPO"] and self.SL_finetune is True and self.save_all is True):
                    # If 'save_last' is False and 'save_constraints' is False and 'save_all' is False,
                    # check if new best avg reward is achieved and if so, save the models.
                    # Else if 'save_constraints' is True, check if new lowest average number of constraints is achieved and if so, save the models.
                    # Otherwise, if 'save_last' or 'save_all' is True, save the last models.
                    if self.save_last is False and self.save_constraints is False and \
                       ((self.algo != "SAC" and not (self.algo == "PPO" and self.SL_finetune is True)) or
                        (self.algo == "SAC" and
                         (self.SL_finetune is False or (self.SL_finetune is True and self.save_all is False))
                        )
                       ) and \
                       test_avg_reward > self.test_avg_best_score:
                        print('Saving model... \nHighest reward achieved: ' + str(test_avg_reward))
                        self.test_avg_best_score = test_avg_reward
                    elif self.save_constraints is True and test_avg_num_constr < self.test_avg_lowest_num_constr:
                        print('Saving model... \nLowest number of constraints achieved: ' + str(test_avg_num_constr))
                        self.test_avg_lowest_num_constr = test_avg_num_constr
                    elif self.save_last is True or \
                         (self.algo in ["SAC", "PPO"] and self.SL_finetune is True and self.save_all is True):
                        print('Saving last models...')
                    else:
                        raise NotImplementedError
                    self.save_agents_models()
                    if (self.algo == "coGAIL") or (self.algo in ["SAC", "PPO"] and self.SL_finetune is False):
                        episode_model_saved = self.i_episode
                    elif self.algo in ["SAC", "PPO"] and self.SL_finetune is True:
                        episode_model_saved = self.epoch
                    else:
                        raise NotImplementedError
                    self.episodes_model_saved.append(episode_model_saved)

            # test logging
            test_print_logs(
                test_avg_reward,
                mean(self.test_length_list[-self.test_max_games:]),
                mean(self.test_game_duration_list[-self.test_max_games:]),
                self.update_test_constraint_violated_list()
            )

    def update_test_constraint_violated_list(self):
        test_ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated = None
        test_ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated = None
        test_ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated = None
        test_ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated = None
        test_ball_not_in_circle_avg_num_constraint_violated = None
        test_ball_not_in_circle_avg_freq_constraint_violated = None
        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
            test_ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated = \
                mean(self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:])
            test_ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated = \
                mean(self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
        if self.constr_ball_only_at_the_up_side_wrt_hole is True:
            test_ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated = \
                mean(self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:])
            test_ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated = \
                mean(self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
        if self.constr_ball_not_in_circle is True:
            test_ball_not_in_circle_avg_num_constraint_violated = \
                mean(self.test_ball_not_in_circle_num_constraint_violated_list[-self.test_max_games:])
            test_ball_not_in_circle_avg_freq_constraint_violated = \
                mean(self.test_ball_not_in_circle_freq_constraint_violated_list[-self.test_max_games:])

        return [test_ball_only_at_the_right_side_wrt_hole_avg_num_constraint_violated,
                test_ball_only_at_the_right_side_wrt_hole_avg_freq_constraint_violated,
                test_ball_only_at_the_up_side_wrt_hole_avg_num_constraint_violated,
                test_ball_only_at_the_up_side_wrt_hole_avg_freq_constraint_violated,
                test_ball_not_in_circle_avg_num_constraint_violated,
                test_ball_not_in_circle_avg_freq_constraint_violated]

    def human_agent_test(self):

        self.test_process()

        # test logging
        test_print_logs(
            mean(self.test_reward_list),
            mean(self.test_length_list),
            mean(self.test_game_duration_list),
            self.update_test_constraint_violated_list()
        )

    def human_alone_test(self):

        self.test_process()

        # test logging
        test_print_logs(
            mean(self.test_reward_list),
            mean(self.test_length_list),
            mean(self.test_game_duration_list),
            self.update_test_constraint_violated_list()
        )

    def agent0s0_test(self):

        self.test_process()

        # test logging
        test_print_logs(
            mean(self.test_reward_list),
            mean(self.test_length_list),
            mean(self.test_game_duration_list),
            self.update_test_constraint_violated_list()
        )

    def test_process(self):

        replay_game = False

        test_game_i = 0
        while test_game_i < self.test_max_games:

            if self.test_model or self.debug_:
                print('Test game: ' + str(test_game_i) + '\n')

            # the timestamp that the game starts
            test_start_game_time = time.time()

            # Reset environment
            test_observation, test_setting_up_duration = \
                self.env.reset(
                    seed=self.test_seed,
                    initialize_seed=False if test_game_i > 0 or (test_game_i == 0 and replay_game) else True,
                    replay_game=replay_game
                )

            if self.algo == 'coGAIL' or self.algo == 'PPO':
                observation_tensor_test = self.obs_to_FT(test_observation)
                if self.algo == 'coGAIL':
                    if self.test_model is False:
                        # In each environment reset, we get a new gode to generate trajectories.
                        code_tensor_test = \
                            self.code_variable_test.get_next_code().unsqueeze(dim=0).to(self.agents.device)
                    else:
                        raise NotImplementedError
                    # store the first code at the corresponding list
                    code_variable_per_step_list_test = [code_tensor_test.detach().cpu().numpy()]
                    # initialize Discriminator's game reward
                    test_game_discriminator_reward = 0
                elif self.algo == 'PPO' and self.icrl is True:
                    test_game_constraint_net_cost = []

            if self.debug_ is True or self.test_model is True:
                # store the first observation at the test dataframe
                test_new_row = get_row_to_store(test_observation, 0., 0.)
                self.df_test = self.df_test.append(test_new_row, ignore_index=True)

            test_step_counter = 0  # keep track of the step number for each game
            test_timed_out = False  # turn to false when the game has been timed out
            test_game_reward = 0  # the cumulative game reward
            test_dist_travel = 0  # the distance that the ball travels
            test_done = False
            # Keep track of the number of constraint violations
            test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list = None
            test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list = None
            test_ball_not_in_circle_constraint_violation_list = None
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list = []
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list = []
            if self.constr_ball_not_in_circle is True:
                test_ball_not_in_circle_constraint_violation_list = []

            # duration in the train_game_number that is not playable by the user
            test_redundant_end_duration = test_setting_up_duration

            # Start to play a game.
            # When the ball has either reached the goal or the game has timed out, stop the current game.
            while not test_done:

                test_game_step_start_time = time.time()  # the step start time
                test_step_counter += 1  # keep track of the step number for each game

                ## Get actions
                if self.algo == 'coGAIL':
                    test_env_X_agent_action, test_env_Y_agent_action, _, _, _, action_tensor_test, _ = \
                        self.get_agent_action(
                            [observation_tensor_test, code_tensor_test],
                            True,
                            None
                        )
                    # Get Discriminator's reward
                    discr_reward_test = self.discr.predict_reward_test(
                        observation_tensor_test[:, :8],
                        action_tensor_test
                    )
                    test_game_discriminator_reward += discr_reward_test.item()
                elif self.algo == 'PPO':
                    test_env_X_agent_action, test_env_Y_agent_action, test_extra_returns = \
                        self.get_agent_action(
                            observation_tensor_test,
                            True,
                            None
                        )
                    if self.icrl is True:
                        test_game_constraint_net_cost.append(
                            self.get_predicted_cost(observation_tensor_test, test_extra_returns)
                        )
                else:
                    test_env_X_agent_action, test_env_Y_agent_action, _, _ = \
                        self.get_agent_action(
                            normalize_features(test_observation) if self.normalize_features
                                                                 else
                            test_observation,
                            True,
                            None
                        )

                # Environment step
                test_transition = self.env.step(
                    [test_env_X_agent_action, test_env_Y_agent_action],
                    test_timed_out,
                    self.goal,
                    self.action_iterations
                )
                test_next_observation, test_reward, test_done, test_fps, test_duration_pause, test_action_list = \
                    test_transition

                test_redundant_end_duration += test_duration_pause  # keep track of the total paused time

                # check if the game has exceeded the maximum timesteps per test game
                if test_step_counter >= self.test_max_timesteps_per_game:
                    test_timed_out = True

                # compute travelled distance
                test_dist_travel = get_distance_traveled(test_dist_travel, test_observation, test_next_observation)

                # Keep track of whether the game ended due to success or due to hitting the time horizon.
                test_fixed_done = 0. if test_step_counter == self.test_max_timesteps_per_game else float(test_done)

                if self.debug_ is True or self.test_model is True:
                    # append row to the dataframe
                    test_new_row = get_row_to_store(test_next_observation, float(test_done), test_fixed_done)
                    self.df_test = self.df_test.append(test_new_row, ignore_index=True)
                    # keep track of the action history.
                    # action_list contains every agent-human pair sent to the environment
                    self.test_action_history = self.test_action_history + test_action_list

                if self.algo == 'coGAIL':
                    # store the code of the current step
                    code_variable_per_step_list_test.append(code_tensor_test.detach().cpu().numpy())

                # calculate game step duration
                test_step_duration = time.time() - test_game_step_start_time - test_duration_pause
                self.test_step_duration_list.append(test_step_duration)

                test_game_reward += test_reward  # keep track of the total game reward

                # Update number of constraint violations
                if self.constr_ball_only_at_the_right_side_wrt_hole is True or \
                   self.constr_ball_only_at_the_up_side_wrt_hole is True or \
                   self.constr_ball_not_in_circle is True:
                    flag_to_print_constr_viol_info = False
                    constr_info_to_print = ""
                    if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                        assert test_next_observation.shape[0] >= 9, ""
                        test_ball_only_at_the_right_side_wrt_hole_constraint_violation = \
                            1 if test_next_observation[8] == self.not_satisfied_constraint_value \
                              else \
                            0
                        test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list.append(
                            test_ball_only_at_the_right_side_wrt_hole_constraint_violation
                        )
                        if self.test_model is True and self.debug_ is True and \
                           test_ball_only_at_the_right_side_wrt_hole_constraint_violation == 1:
                            flag_to_print_constr_viol_info = True
                            constr_info_to_print += "Violation of 'ball_only_at_the_right_side_wrt_hole' constraint!\n"
                    if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                            assert test_next_observation.shape[0] >= 10, ""
                            test_ball_only_at_the_up_side_wrt_hole_constraint_violation = \
                                1 if test_next_observation[9] == self.not_satisfied_constraint_value \
                                  else \
                                0
                        else:
                            assert test_next_observation.shape[0] >= 9, ""
                            test_ball_only_at_the_up_side_wrt_hole_constraint_violation = \
                                1 if test_next_observation[8] == self.not_satisfied_constraint_value \
                                  else \
                                0
                        test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list.append(
                            test_ball_only_at_the_up_side_wrt_hole_constraint_violation
                        )
                        if self.test_model is True and \
                           self.debug_ is True and \
                           test_ball_only_at_the_up_side_wrt_hole_constraint_violation == 1:
                            flag_to_print_constr_viol_info = True
                            constr_info_to_print += "Violation of 'ball_only_at_the_up_side_wrt_hole' constraint!\n"
                    if self.constr_ball_not_in_circle is True:
                        if self.constr_ball_only_at_the_right_side_wrt_hole is True and \
                           self.constr_ball_only_at_the_up_side_wrt_hole is True:
                            assert test_next_observation.shape[0] >= 11, ""
                            test_ball_not_in_circle_constraint_violation = \
                                1 if test_next_observation[10] == self.not_satisfied_constraint_value \
                                  else \
                                0
                        elif (self.constr_ball_only_at_the_right_side_wrt_hole is True and
                              self.constr_ball_only_at_the_up_side_wrt_hole is False) or \
                             (self.constr_ball_only_at_the_right_side_wrt_hole is False and
                              self.constr_ball_only_at_the_up_side_wrt_hole is True):
                            assert test_next_observation.shape[0] >= 10, ""
                            test_ball_not_in_circle_constraint_violation = \
                                1 if test_next_observation[9] == self.not_satisfied_constraint_value \
                                  else \
                                0
                        elif self.constr_ball_only_at_the_right_side_wrt_hole is False and \
                             self.constr_ball_only_at_the_up_side_wrt_hole is False:
                            assert test_next_observation.shape[0] >= 9, ""
                            test_ball_not_in_circle_constraint_violation = \
                                1 if test_next_observation[8] == self.not_satisfied_constraint_value \
                                  else \
                                0
                        else:
                            raise NotImplementedError
                        test_ball_not_in_circle_constraint_violation_list.append(
                            test_ball_not_in_circle_constraint_violation
                        )
                        if self.test_model is True and \
                           self.debug_ is True and \
                           test_ball_not_in_circle_constraint_violation == 1:
                            flag_to_print_constr_viol_info = True
                            constr_info_to_print += "Violation of 'ball_not_in_circle' constraint!\n"
                        if flag_to_print_constr_viol_info is True:
                            print(constr_info_to_print)

                # set the observation for the next step
                test_observation = test_next_observation.copy()
                if self.algo == 'coGAIL' or self.algo == 'PPO':
                    observation_tensor_test = self.obs_to_FT(test_next_observation)

            ## End of test game

            test_game_i += 1

            if self.freeze_motion:
                # NOTE that this case is only implemented when a human (or two humans)
                # plays alone (without an algorithm).
                print('Replay game?')
                answer = input('Y/N: ')
                print()  # Just for leaving an empty line
                while answer != 'Y' and answer != 'N':
                    answer = input('Invalid answer! Please answer with Y or N: ')
                replay_game = False if answer == 'N' else (True if answer == 'Y' else None)
                if replay_game:
                    # Roll back variable values to those before the last game
                    test_game_i -= 1
                    rows_to_be_deleted_list = list(
                        range(len(self.df_test.index)-(test_step_counter+1), len(self.df_test.index))
                    )
                    self.df_test.drop(rows_to_be_deleted_list, axis=0, inplace=True)
                    del self.test_action_history[-test_step_counter:]
                    del self.test_step_duration_list[-test_step_counter:]

            if not replay_game:
                # update the testing metrics
                self.update_test_metrics(
                    test_redundant_end_duration,
                    test_start_game_time,
                    test_game_reward,
                    test_dist_travel,
                    test_step_counter,
                    test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list,
                    test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list,
                    test_ball_not_in_circle_constraint_violation_list
                )

            if self.algo == 'coGAIL':
                # Store all codes of the game
                self.code_variable_per_game_list_of_np_test.append(code_variable_per_step_list_test.copy())
                # Store Discriminator's game reward
                self.discr_reward_per_game_list_test.append(test_game_discriminator_reward)
            elif self.algo == 'PPO' and self.icrl is True:
                self.constraint_net_cost_per_game_list_test.append(test_game_constraint_net_cost)

    def initialize_game_var_train(self):

        # Train mode
        self.train_mode()

        self.start_game_time = time.time()
        self.timed_out = False  # used to check if we hit the maximum train_game_number duration
        self.game_reward = 0  # keeps track of the rewards for each train_game_number
        self.dist_travel = 0  # keeps track of the ball's travelled distance
        self.train_step_counter = 0
        self.done = False
        # Keep track of the number of constraint violations
        self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list = None
        self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list = None
        self.ball_not_in_circle_constraint_violation_list = None
        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
            self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list = []
        if self.constr_ball_only_at_the_up_side_wrt_hole is True:
            self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list = []
        if self.constr_ball_not_in_circle is True:
            self.ball_not_in_circle_constraint_violation_list = []

        if self.algo == "SAC":
            initialize_env_seed = False if ((self.i_episode - 1) % self.test_every_episodes) != 0 else True

            # Initialize lists for training details
            if self.X_agent is not None and self.Y_agent is not None:
                if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                    self.X_q1_loss_cur_game_per_step_list = []
                    self.X_q2_loss_cur_game_per_step_list = []
                    self.X_entropies_cur_game_per_step_list = []
                    self.X_entropy_loss_cur_game_per_step_list = []
                    self.X_policy_loss_cur_game_per_step_list = []
                    self.X_entropy_coef_cur_game_per_step_list = []
                    self.X_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                    self.X_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                    self.X_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_q1_loss_cur_game_per_step_list = []
                    self.Y_q2_loss_cur_game_per_step_list = []
                    self.Y_entropies_cur_game_per_step_list = []
                    self.Y_entropy_loss_cur_game_per_step_list = []
                    self.Y_policy_loss_cur_game_per_step_list = []
                    self.Y_entropy_coef_cur_game_per_step_list = []
                    self.Y_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                    self.Y_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                    self.Y_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                if self.w_constraint_optimization is True:
                    if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                        self.X_constraint_policy_loss_term_value_cur_game_per_step_list = []
                        self.X_constraint_lambda_loss_value_cur_game_per_step_list = []
                        self.X_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                        self.X_constraint_lambda_cur_game_per_step_list = []
                    if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                        self.Y_constraint_policy_loss_term_value_cur_game_per_step_list = []
                        self.Y_constraint_lambda_loss_value_cur_game_per_step_list = []
                        self.Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                        self.Y_constraint_lambda_cur_game_per_step_list = []
            elif self.X_Y_agent is not None:
                self.X_Y_q1_loss_cur_game_per_step_list = []
                self.X_Y_q2_loss_cur_game_per_step_list = []
                self.X_Y_entropies_cur_game_per_step_list = []
                self.X_Y_entropy_loss_cur_game_per_step_list = []
                self.X_Y_policy_loss_cur_game_per_step_list = []
                self.X_Y_entropy_coef_cur_game_per_step_list = []
                self.X_Y_q1_grad_norm_clipped_value_cur_game_per_step_list = []
                self.X_Y_q2_grad_norm_clipped_value_cur_game_per_step_list = []
                self.X_Y_actor_grad_norm_clipped_value_cur_game_per_step_list = []
                if self.w_constraint_optimization is True:
                    self.X_Y_constraint_policy_loss_term_value_cur_game_per_step_list = []
                    self.X_Y_constraint_lambda_loss_value_cur_game_per_step_list = []
                    self.X_Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list = []
                    self.X_Y_constraint_lambda_cur_game_per_step_list = []
            else:
                raise NotImplementedError

        elif self.algo == 'coGAIL' or self.algo == 'PPO':
            initialize_env_seed = False if ((self.total_games - 1) % self.test_every_games != 0) else True

        else:
            raise ValueError

        # reset environment
        self.observation, self.setting_up_duration = self.env.reset(seed=None, initialize_seed=initialize_env_seed)

        if self.debug_:
            # store the first observation at the dataframe
            new_row = get_row_to_store(self.observation, 0., 0.)
            self.df = self.df.append(new_row, ignore_index=True)

        # duration in the game that is not playable by the user
        self.redundant_end_duration = self.setting_up_duration

    def train_mode(self):
        if self.algo == 'SAC':
            if self.X_agent is not None and self.Y_agent is not None:
                if not self.X_agent.axis_agent == 'X_Y':
                    self.X_agent.actor.train()
                    self.X_agent.critic.train()
                if not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_agent.actor.train()
                    self.Y_agent.critic.train()
            elif self.X_Y_agent is not None:
                self.X_Y_agent.actor.train()
                self.X_Y_agent.critic.train()
            else:
                raise NotImplementedError

        elif self.algo == 'coGAIL':
            self.actor_critic.train()
            self.discr.train()

        elif self.algo == 'PPO':
            if self.X_agent is not None and self.Y_agent is not None:
                if not self.X_agent.axis_agent == 'X_Y':
                    self.X_agent.ppo.policy.train()
                    if self.icrl:
                        self.X_agent.constraint_net.train()
                if not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_agent.ppo.policy.train()
                    if self.icrl:
                        self.Y_agent.constraint_net.train()
            elif self.X_Y_agent is not None:
                self.X_Y_agent.ppo.policy.train()
                if self.icrl:
                    self.X_Y_agent.constraint_net.train()
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def eval_mode(self):
        if self.algo == 'SAC':
            if self.X_agent is not None and self.Y_agent is not None:
                if not self.X_agent.axis_agent == 'X_Y':
                    self.X_agent.actor.eval()
                    self.X_agent.critic.eval()
                if not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_agent.actor.eval()
                    self.Y_agent.critic.eval()
            elif self.X_Y_agent is not None:
                self.X_Y_agent.actor.eval()
                self.X_Y_agent.critic.eval()
            else:
                raise NotImplementedError

        elif self.algo == 'coGAIL':
            self.actor_critic.eval()
            self.discr.eval()

        elif self.algo == 'PPO':
            if self.X_agent is not None and self.Y_agent is not None:
                if not self.X_agent.axis_agent == 'X_Y':
                    self.X_agent.ppo.policy.eval()
                    if self.icrl:
                        self.X_agent.constraint_net.eval()
                if not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_agent.ppo.policy.eval()
                    if self.icrl:
                        self.Y_agent.constraint_net.eval()
            elif self.X_Y_agent is not None:
                self.X_Y_agent.ppo.policy.eval()
                if self.icrl:
                    self.X_Y_agent.constraint_net.eval()
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def save_info(self, chkpt_dir, experiment_duration):
        """
        Saves experiment additional information in a file
        :param chkpt_dir: the checkpoint directory to store the file
        :param experiment_duration: the total duration of the experiment
        """

        info = {
            'goal': self.goal,
            'experiment_duration': experiment_duration,
            'total_games': self.test_max_games if self.test_model is True else self.max_games
        }
        if not self.test_model:
            info['total_steps'] = self.total_steps

        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    def reset_buffer(self):

        if self.algo == 'coGAIL':
            self.pi_co.rollout_storage.after_update()

        elif self.algo == 'PPO':
            if self.X_agent is not None and self.Y_agent is not None:
                if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                    self.X_agent.ppo.rollout_buffer.reset()
                if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                    self.Y_agent.ppo.rollout_buffer.reset()
            elif self.X_Y_agent is not None:
                self.X_Y_agent.ppo.rollout_buffer.reset()

        else:
            raise NotImplementedError

    def save_experience(self, data):
        """
        Saves an interaction to the replay buffer of the agent.
        :param data: list with data of the interaction to be stored in the Replay Buffer.
        """

        if self.algo == "SAC":

            observation, next_observation, real_X_agent_action, real_Y_agent_action, reward, fixed_done = data

            if self.X_agent is not None and self.Y_agent is not None:
                if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                    self.X_agent.memory.add(
                        normalize_features(observation.copy()[:8]) if self.normalize_features
                                                                   else
                        observation.copy()[:8],
                        real_X_agent_action,
                        reward,
                        normalize_features(next_observation.copy()[:8]) if self.normalize_features
                                                                        else
                        next_observation.copy()[:8],
                        fixed_done
                    )
                if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                    self.Y_agent.memory.add(
                        normalize_features(observation.copy()[:8]) if self.normalize_features
                                                                   else
                        observation.copy()[:8],
                        real_Y_agent_action, reward,
                        normalize_features(next_observation.copy()[:8]) if self.normalize_features
                                                                        else
                        next_observation.copy()[:8],
                        fixed_done
                    )

            elif self.X_Y_agent is not None:
                self.X_Y_agent.memory.add(
                    observation.copy()[:8] if not self.normalize_features
                                           else
                    normalize_features(observation.copy()[:8]),
                    transform_actions_to_action(
                        [real_X_agent_action, real_Y_agent_action],
                        self.env.action_space.actions_number
                    ),
                    reward,
                    next_observation.copy()[:8] if not self.normalize_features
                                                else
                    normalize_features(next_observation.copy()[:8]),
                    fixed_done
                )
            else:
                raise NotImplementedError

        elif self.algo == "PPO":
            observation = data[0][:, :8].squeeze(0)
            done = np.array([data[1]], dtype=float)
            reward = np.array([data[2]])

            if self.X_agent is not None or self.Y_agent is not None:
                if self.X_agent is not None and self.X_agent.axis_agent == "X":
                    X_action = data[4].unsqueeze(0)
                    X_cost = None if self.icrl is False else data[3][0]
                    X_values = data[6].squeeze(0)
                    X_cost_values = None if self.icrl is False else data[7].squeeze(0)
                    X_log_probs = data[8]
                    self.X_agent.ppo.rollout_buffer.add(
                        observation,
                        X_action,
                        reward,
                        X_cost,
                        done,
                        X_values,
                        X_cost_values,
                        X_log_probs
                    )
                if self.Y_agent is not None and self.Y_agent.axis_agent == "Y":
                    Y_action = data[5].unsqueeze(0)
                    Y_cost = None if self.icrl is False else data[3][1]
                    Y_values = data[9].squeeze(0)
                    Y_cost_values = None if self.icrl is False else data[10].squeeze(0)
                    Y_log_probs = data[11]
                    self.Y_agent.ppo.rollout_buffer.add(
                        observation,
                        Y_action,
                        reward,
                        Y_cost,
                        done,
                        Y_values,
                        Y_cost_values,
                        Y_log_probs
                    )
            elif self.X_Y_agent is not None or self.X_Y_agent.axis_agent == "X_Y":
                X_Y_action = data[4].unsqueeze(0)
                X_Y_cost = None if self.icrl is False else data[3]
                X_Y_values = data[5].squeeze(0)
                X_Y_cost_values = None if self.icrl is False else data[6].squeeze(0)
                X_Y_log_probs = data[7]
                self.X_Y_agent.ppo.rollout_buffer.add(
                    observation,
                    X_Y_action,
                    reward,
                    X_Y_cost,
                    done,
                    X_Y_values,
                    X_Y_cost_values,
                    X_Y_log_probs
                )
            else:
                raise NotImplementedError

        elif self.algo == "coGAIL":

            next_observation, done, fixed_done, reward, value_tensor, \
                actions_tensor, action_log_prob_tensor, random_seed_tensor = data

            masks_tensor = torch.from_numpy(
                np.array([1.0 - float(done)], dtype=np.float32)
            ).float().to(self.agents.device)
            bad_masks_tensor = torch.from_numpy(
                np.array([1.0 - fixed_done], dtype=np.float32)
            ).float().to(self.agents.device)
            next_observation_tensor = self.obs_to_FT(next_observation)
            dummy_reward_tensor = torch.from_numpy(
                np.array([np.nan], dtype=np.float32)
            ).float().to(self.agents.device)
            env_reward_tensor = torch.from_numpy(
                np.array([reward], dtype=np.float32)
            ).float().to(self.agents.device)
            env_value_tensor = None if not self.pi_co.rollout_storage.opt_robot_w_env_rewards \
                                    else \
                               value_tensor[:, 1]
            value_tensor = value_tensor.squeeze(0) if not self.pi_co.rollout_storage.opt_robot_w_env_rewards \
                                                   else \
                           value_tensor[:, 0]

            self.pi_co.rollout_storage.insert(
                next_observation_tensor,
                actions_tensor.squeeze(0),
                action_log_prob_tensor.squeeze(0),
                value_tensor.squeeze(0),
                dummy_reward_tensor,
                masks_tensor,
                bad_masks_tensor,
                random_seed_tensor.squeeze(0),
                env_reward_tensor,
                env_value_tensor
            )

        else:
            raise NotImplementedError

    def train_game_logging(self):

        # keep track of total pause duration
        self.end_game_time = time.time()

        # update time metrics about the experiment
        self.update_time_metrics()

        # update training metrics about the experiment
        self.update_train_metrics()

        ## logging per log interval
        if self.algo == 'SAC':
            log_flag = self.i_episode % self.log_interval == 0 or self.i_episode == 1
        elif self.algo == 'coGAIL' or self.algo == 'PPO':
            log_flag = self.total_games % self.test_every_games == 0 or self.total_games == 1
        else:
            raise NotImplementedError

        if log_flag is True:

            # Calculate per_log_interval values
            reward_avg_per_log_interval = mean(self.reward_list[-self.test_max_games:])
            length_avg_per_log_interval = mean(self.length_list[-self.test_max_games:])
            game_duration_avg_per_log_interval = mean(self.game_duration_list[-self.test_max_games:])
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg_per_log_interval = \
                    mean(self.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:])
                ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval = \
                    mean(self.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg_per_log_interval = \
                    mean(self.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list[-self.test_max_games:])
                ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval = \
                    mean(self.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list[-self.test_max_games:])
            if self.constr_ball_not_in_circle is True:
                ball_not_in_circle_num_constraint_violated_avg_per_log_interval = \
                    mean(self.ball_not_in_circle_num_constraint_violated_list[-self.test_max_games:])
                ball_not_in_circle_freq_constraint_violated_list_avg_per_log_interval = \
                    mean(self.ball_not_in_circle_freq_constraint_violated_list[-self.test_max_games:])

            # Store per_log_interval values
            self.game_duration_list_avg_per_log_interval.append(game_duration_avg_per_log_interval)
            self.reward_list_avg_per_log_interval.append(reward_avg_per_log_interval)
            self.distance_travel_list_avg_per_log_interval.append(mean(self.distance_travel_list[-self.test_max_games:]))
            self.length_list_avg_per_log_interval.append(length_avg_per_log_interval)
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                self.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_per_log_interval.append(
                    ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg_per_log_interval
                )
                self.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_per_log_interval.append(
                    ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval
                )
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                self.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_per_log_interval.append(
                    ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg_per_log_interval
                )
                self.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_per_log_interval.append(
                    ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval
                )
            if self.constr_ball_not_in_circle is True:
                self.ball_not_in_circle_num_constraint_violated_per_log_interval.append(
                    ball_not_in_circle_num_constraint_violated_avg_per_log_interval
                )
                self.ball_not_in_circle_freq_constraint_violated_per_log_interval.append(
                    ball_not_in_circle_freq_constraint_violated_list_avg_per_log_interval
                )

            # Print per_log_interval values
            print('\n##########Average stats for training##########')
            print('Total episodes until now: {}\n'
                  'Total timesteps: {}\n'
                  '##Avg over the last {} games##\n'
                  'Avg per length: {}\n'
                  'Avg per game reward: {}\n'
                  'Avg episode duration: {}'
                  .format(self.i_episode,
                          self.total_steps,
                          self.test_max_games,
                          round(length_avg_per_log_interval, 2),
                          round(reward_avg_per_log_interval, 2),
                          timedelta(seconds=game_duration_avg_per_log_interval))
                  )
            if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                print("\nAvg number of 'ball_only_at_the_right_side_wrt_hole' constraint violations: {}\n"
                      "Avg frequency of 'ball_only_at_the_right_side_wrt_hole' constraint violations: {}"
                      .format(
                    round(ball_only_at_the_right_side_wrt_hole_num_constraint_violated_avg_per_log_interval, 2),
                    round(ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval, 2)
                ))
            if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                print("\nAvg number of 'ball_only_at_the_up_side_wrt_hole' constraint violations: {}\n"
                      "Avg frequency of 'ball_only_at_the_up_side_wrt_hole' constraint violations: {}"
                      .format(
                    round(ball_only_at_the_up_side_wrt_hole_num_constraint_violated_avg_per_log_interval, 2),
                    round(ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list_avg_per_log_interval, 2)
                ))
            if self.constr_ball_not_in_circle is True:
                print("\nAvg number of 'ball_not_in_circle' constraint violations: {}\n"
                      "Avg frequency of 'ball_not_in_circle' constraint violations: {}"
                      .format(
                    round(ball_not_in_circle_num_constraint_violated_avg_per_log_interval, 2),
                    round(ball_not_in_circle_freq_constraint_violated_list_avg_per_log_interval, 2)
                ))

        if self.algo == "SAC":

            # Store all game losses
            if (self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y' and
                self.X_agent.memory.get_size() > self.batch_size) or \
                (self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y' and
                 self.Y_agent.memory.get_size() > self.batch_size) or \
                    (self.X_Y_agent is not None and self.X_Y_agent.memory.get_size() > self.batch_size):

                if self.X_agent is not None and self.Y_agent is not None:
                    if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                        self.X_q1_loss_per_step_list.append(self.X_q1_loss_cur_game_per_step_list)
                        self.X_q2_loss_per_step_list.append(self.X_q2_loss_cur_game_per_step_list)
                        self.X_entropies_per_step_list.append(self.X_entropies_cur_game_per_step_list)
                        self.X_entropy_loss_per_step_list.append(self.X_entropy_loss_cur_game_per_step_list)
                        self.X_policy_loss_per_step_list.append(self.X_policy_loss_cur_game_per_step_list)
                        self.X_entropy_coef_per_step_list.append(self.X_entropy_coef_cur_game_per_step_list)
                        self.X_q1_grad_norm_clipped_value_per_step_list.append(
                            self.X_q1_grad_norm_clipped_value_cur_game_per_step_list
                        )
                        self.X_q2_grad_norm_clipped_value_per_step_list.append(
                            self.X_q2_grad_norm_clipped_value_cur_game_per_step_list
                        )
                        self.X_actor_grad_norm_clipped_value_per_step_list.append(
                            self.X_actor_grad_norm_clipped_value_cur_game_per_step_list
                        )
                    if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                        self.Y_q1_loss_per_step_list.append(self.Y_q1_loss_cur_game_per_step_list)
                        self.Y_q2_loss_per_step_list.append(self.Y_q2_loss_cur_game_per_step_list)
                        self.Y_entropies_per_step_list.append(self.Y_entropies_cur_game_per_step_list)
                        self.Y_entropy_loss_per_step_list.append(self.Y_entropy_loss_cur_game_per_step_list)
                        self.Y_policy_loss_per_step_list.append(self.Y_policy_loss_cur_game_per_step_list)
                        self.Y_entropy_coef_per_step_list.append(self.Y_entropy_coef_cur_game_per_step_list)
                        self.Y_q1_grad_norm_clipped_value_per_step_list.append(
                            self.Y_q1_grad_norm_clipped_value_cur_game_per_step_list
                        )
                        self.Y_q2_grad_norm_clipped_value_per_step_list.append(
                            self.Y_q2_grad_norm_clipped_value_cur_game_per_step_list
                        )
                        self.Y_actor_grad_norm_clipped_value_per_step_list.append(
                            self.Y_actor_grad_norm_clipped_value_cur_game_per_step_list
                        )
                    if self.w_constraint_optimization is True:
                        if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                            self.X_constraint_policy_loss_term_value_per_step_list.append(
                                self.X_constraint_policy_loss_term_value_cur_game_per_step_list
                            )
                            self.X_constraint_lambda_loss_value_per_step_list.append(
                                self.X_constraint_lambda_loss_value_cur_game_per_step_list
                            )
                            self.X_policy_loss_value_wo_constraint_term_per_step_list.append(
                                self.X_policy_loss_value_wo_constraint_term_cur_game_per_step_list
                            )
                            self.X_constraint_lambda_per_step_list.append(
                                self.X_constraint_lambda_cur_game_per_step_list
                            )
                        if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                            self.Y_constraint_policy_loss_term_value_per_step_list.append(
                                self.Y_constraint_policy_loss_term_value_cur_game_per_step_list
                            )
                            self.Y_constraint_lambda_loss_value_per_step_list.append(
                                self.Y_constraint_lambda_loss_value_cur_game_per_step_list
                            )
                            self.Y_policy_loss_value_wo_constraint_term_per_step_list.append(
                                self.Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list
                            )
                            self.Y_constraint_lambda_per_step_list.append(
                                self.Y_constraint_lambda_cur_game_per_step_list
                            )
                elif self.X_Y_agent is not None:
                    self.X_Y_q1_loss_per_step_list.append(self.X_Y_q1_loss_cur_game_per_step_list)
                    self.X_Y_q2_loss_per_step_list.append(self.X_Y_q2_loss_cur_game_per_step_list)
                    self.X_Y_entropies_per_step_list.append(self.X_Y_entropies_cur_game_per_step_list)
                    self.X_Y_entropy_loss_per_step_list.append(self.X_Y_entropy_loss_cur_game_per_step_list)
                    self.X_Y_policy_loss_per_step_list.append(self.X_Y_policy_loss_cur_game_per_step_list)
                    self.X_Y_entropy_coef_per_step_list.append(self.X_Y_entropy_coef_cur_game_per_step_list)
                    self.X_Y_q1_grad_norm_clipped_value_per_step_list.append(
                        self.X_Y_q1_grad_norm_clipped_value_cur_game_per_step_list
                    )
                    self.X_Y_q2_grad_norm_clipped_value_per_step_list.append(
                        self.X_Y_q2_grad_norm_clipped_value_cur_game_per_step_list
                    )
                    self.X_Y_actor_grad_norm_clipped_value_per_step_list.append(
                        self.X_Y_actor_grad_norm_clipped_value_cur_game_per_step_list
                    )
                    if self.w_constraint_optimization is True:
                        self.X_Y_constraint_policy_loss_term_value_per_step_list.append(
                            self.X_Y_constraint_policy_loss_term_value_cur_game_per_step_list
                        )
                        self.X_Y_constraint_lambda_loss_value_per_step_list.append(
                            self.X_Y_constraint_lambda_loss_value_cur_game_per_step_list
                        )
                        self.X_Y_policy_loss_value_wo_constraint_term_per_step_list.append(
                            self.X_Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list
                        )
                        self.X_Y_constraint_lambda_per_step_list.append(
                            self.X_Y_constraint_lambda_cur_game_per_step_list
                        )
                else:
                    raise NotImplementedError

                ## logging per interval
                if self.i_episode % self.log_interval == 0:

                    # Calculate and print per_log_interval values
                    if (self.X_agent is not None and
                        not self.X_agent.axis_agent == 'X_Y' and
                        len(self.X_q1_loss_per_step_list) >= self.test_max_games) or \
                       (self.Y_agent is not None and
                        not self.Y_agent.axis_agent == 'X_Y' and
                        len(self.Y_q1_loss_per_step_list) >= self.test_max_game):
                        if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                            X_q1_loss_avg_per_log_interval = np.mean([
                                np.mean(self.X_q1_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_q2_loss_avg_per_log_interval = np.mean([
                                np.mean(self.X_q2_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_entropies_avg_per_log_interval = np.mean([
                                np.mean(self.X_entropies_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_entropy_loss_avg_per_log_interval = np.mean([
                                np.mean(self.X_entropy_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_policy_loss_avg_per_log_interval = np.mean([
                                np.mean(self.X_policy_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_entropy_coef_avg_per_log_interval = np.mean([
                                np.mean(self.X_entropy_coef_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_q1_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.X_q1_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_q2_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.X_q2_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            X_actor_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.X_actor_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            print("\nAvg X_q1_loss: {}\n"
                                  "Avg X_q2_loss: {}\n"
                                  "Avg X_entropies: {}\n"
                                  "Avg X_entropy_loss: {}\n"
                                  "Avg X_policy: {}\n"
                                  "Avg X_entropy_coefficient: {}"
                                  .format(round(float(X_q1_loss_avg_per_log_interval), 2),
                                          round(float(X_q2_loss_avg_per_log_interval), 2),
                                          round(float(X_entropies_avg_per_log_interval), 2),
                                          round(float(X_entropy_loss_avg_per_log_interval), 2),
                                          round(float(X_policy_loss_avg_per_log_interval), 2),
                                          round(float(X_entropy_coef_avg_per_log_interval), 2)
                                          )
                                  )
                            if self.X_agent.clip_grad_norm:
                                print("\nAvg X_q1_gradient_norm_clipped_value: {}\n"
                                      "Avg X_q2_gradient_norm_clipped_value: {}\n"
                                      "Avg X_actor_gradient_norm_clipped_value: {}"
                                      .format(round(float(X_q1_grad_norm_clipped_value_avg_per_log_interval), 2),
                                              round(float(X_q2_grad_norm_clipped_value_avg_per_log_interval), 2),
                                              round(float(X_actor_grad_norm_clipped_value_avg_per_log_interval), 2)
                                              )
                                      )
                        if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                            Y_q1_loss_avg_per_log_interval = np.mean([
                                np.mean(self.Y_q1_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_q2_loss_avg_per_log_interval = np.mean([
                                np.mean(self.Y_q2_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_entropies_avg_per_log_interval = np.mean([
                                np.mean(self.Y_entropies_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_entropy_loss_avg_per_log_interval = np.mean([
                                np.mean(self.Y_entropy_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_policy_loss_avg_per_log_interval = np.mean([
                                np.mean(self.Y_policy_loss_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_entropy_coef_avg_per_log_interval = np.mean([
                                np.mean(self.Y_entropy_coef_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_q1_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.Y_q1_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_q2_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.Y_q2_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            Y_actor_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                                np.mean(self.Y_actor_grad_norm_clipped_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                            ])
                            print("\nAvg Y_q1_loss: {}\n"
                                  "Avg Y_q2_loss: {}\n"
                                  "Avg Y_entropies: {}\n"
                                  "Avg Y_entropy_loss: {}\n"
                                  "Avg Y_policy_loss: {}\n"
                                  "Avg Y_entropy_coefficient: {}"
                                  .format(round(float(Y_q1_loss_avg_per_log_interval), 2),
                                          round(float(Y_q2_loss_avg_per_log_interval), 2),
                                          round(float(Y_entropies_avg_per_log_interval), 2),
                                          round(float(Y_entropy_loss_avg_per_log_interval), 2),
                                          round(float(Y_policy_loss_avg_per_log_interval), 2),
                                          round(float(Y_entropy_coef_avg_per_log_interval), 2)
                                          )
                                  )
                            if self.Y_agent.clip_grad_norm:
                                print("\nAvg Y_q1_gradient_norm_clipped_value: {}\n"
                                      "Avg Y_q2_gradient_norm_clipped_value: {}\n"
                                      "Avg Y_actor_gradient_norm_clipped_value: {}"
                                      .format(round(float(Y_q1_grad_norm_clipped_value_avg_per_log_interval), 2),
                                              round(float(Y_q2_grad_norm_clipped_value_avg_per_log_interval), 2),
                                              round(float(Y_actor_grad_norm_clipped_value_avg_per_log_interval), 2)
                                              )
                                      )
                        if self.w_constraint_optimization is True:
                            if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                                X_constraint_policy_loss_term_value_avg_per_log_interval = np.mean([
                                    np.mean(self.X_constraint_policy_loss_term_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                X_constraint_lambda_loss_value_avg_per_log_interval = np.mean([
                                    np.mean(self.X_constraint_lambda_loss_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                X_policy_loss_value_wo_constraint_term_avg_per_log_interval = np.mean([
                                    np.mean(self.X_policy_loss_value_wo_constraint_term_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                X_constraint_lambda_avg_per_log_interval = np.mean([
                                    np.mean(self.X_constraint_lambda_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                print("\nAvg X_constraint_policy_loss_term_value: {}\n"
                                      "Avg X_constraint_lambda_loss_value: {}\n"
                                      "Avg X_policy_loss_value_wo_constraint_term: {}\n"
                                      "Avg X_constraint_lambda: {}"
                                      .format(round(float(X_constraint_policy_loss_term_value_avg_per_log_interval), 2),
                                              round(float(X_constraint_lambda_loss_value_avg_per_log_interval), 2),
                                              round(float(X_policy_loss_value_wo_constraint_term_avg_per_log_interval), 2),
                                              round(float(X_constraint_lambda_avg_per_log_interval), 2)
                                              )
                                      )
                            if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                                Y_constraint_policy_loss_term_value_avg_per_log_interval = np.mean([
                                    np.mean(self.Y_constraint_policy_loss_term_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                Y_constraint_lambda_loss_value_avg_per_log_interval = np.mean([
                                    np.mean(self.Y_constraint_lambda_loss_value_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                Y_policy_loss_value_wo_constraint_term_avg_per_log_interval = np.mean([
                                    np.mean(self.Y_policy_loss_value_wo_constraint_term_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                Y_constraint_lambda_avg_per_log_interval = np.mean([
                                    np.mean(self.Y_constraint_lambda_per_step_list[-(i+1)]) for i in range(self.test_max_games)
                                ])
                                print("\nAvg Y_constraint_policy_loss_term_value: {}\n"
                                      "Avg Y_constraint_lambda_loss_value: {}\n"
                                      "Avg Y_policy_loss_value_wo_constraint_term: {}\n"
                                      "Avg Y_constraint_lambda: {}"
                                      .format(round(float(Y_constraint_policy_loss_term_value_avg_per_log_interval), 2),
                                              round(float(Y_constraint_lambda_loss_value_avg_per_log_interval), 2),
                                              round(float(Y_policy_loss_value_wo_constraint_term_avg_per_log_interval), 2),
                                              round(float(Y_constraint_lambda_avg_per_log_interval), 2)
                                              )
                                      )
                    elif self.X_Y_agent is not None and len(self.X_Y_q1_loss_per_step_list) >= self.test_max_games:
                        X_Y_q1_loss_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_q1_loss_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_q2_loss_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_q2_loss_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_entropies_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_entropies_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_entropy_loss_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_entropy_loss_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_policy_loss_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_policy_loss_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_entropy_coef_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_entropy_coef_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_q1_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_q1_grad_norm_clipped_value_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_q2_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_q2_grad_norm_clipped_value_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        X_Y_actor_grad_norm_clipped_value_avg_per_log_interval = np.mean([
                            np.mean(self.X_Y_actor_grad_norm_clipped_value_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                        ])
                        print("Avg X_Y_q1_loss: {}\n"
                              "Avg X_Y_q2_loss: {}\n"
                              "Avg X_Y_entropies: {}\n"
                              "Avg X_Y_entropy_loss: {}\n"
                              "Avg X_Y_policy_loss: {}\n"
                              "Avg X_Y_entropy_coefficient: {}"
                              .format(round(float(X_Y_q1_loss_avg_per_log_interval), 2),
                                      round(float(X_Y_q2_loss_avg_per_log_interval), 2),
                                      round(float(X_Y_entropies_avg_per_log_interval), 2),
                                      round(float(X_Y_entropy_loss_avg_per_log_interval), 2),
                                      round(float(X_Y_policy_loss_avg_per_log_interval), 2),
                                      round(float(X_Y_entropy_coef_avg_per_log_interval), 2)
                                      )
                              )
                        if self.X_Y_agent.clip_grad_norm:
                            print("X_Y_q1_gradient_norm_clipped_value: {}\n"
                                  "X_Y_q2_gradient_norm_clipped_value: {}\n"
                                  "X_Y_actor_gradient_norm_clipped_value: {}"
                                  .format(round(float(X_Y_q1_grad_norm_clipped_value_avg_per_log_interval), 2),
                                          round(float(X_Y_q2_grad_norm_clipped_value_avg_per_log_interval), 2),
                                          round(float(X_Y_actor_grad_norm_clipped_value_avg_per_log_interval), 2)
                                          )
                                  )
                        if self.w_constraint_optimization is True:
                            X_Y_constraint_policy_loss_term_value_avg_per_log_interval = np.mean([
                                np.mean(self.X_Y_constraint_policy_loss_term_value_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                            ])
                            X_Y_constraint_lambda_loss_value_avg_per_log_interval = np.mean([
                                np.mean(self.X_Y_constraint_lambda_loss_value_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                            ])
                            X_Y_policy_loss_value_wo_constraint_term_avg_per_log_interval = np.mean([
                                np.mean(self.X_Y_policy_loss_value_wo_constraint_term_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                            ])
                            X_Y_constraint_lambda_avg_per_log_interval = np.mean([
                                np.mean(self.X_Y_constraint_lambda_per_step_list[-(i + 1)]) for i in range(self.test_max_games)
                            ])
                            print("\nConstraint_policy_loss_term_value: {}\n"
                                  "Constraint_lambda_loss_value: {}\n"
                                  "Policy_loss_value_wo_constraint_term: {}\n"
                                  "Constraint_lambda: {}".
                                  format(round(float(X_Y_constraint_policy_loss_term_value_avg_per_log_interval), 2),
                                         round(float(X_Y_constraint_lambda_loss_value_avg_per_log_interval), 2),
                                         round(float(X_Y_policy_loss_value_wo_constraint_term_avg_per_log_interval), 2),
                                         round(float(X_Y_constraint_lambda_avg_per_log_interval), 2)
                                         )
                                  )

            if self.debug_:
                # Print useful information
                print("\nEpisode: {}\n"
                      "Total num_steps: {}\n"
                      "Episode steps: {}\n"
                      "Reward: {}".
                      format(self.i_episode,
                             self.total_steps,
                             self.train_step_counter,
                             round(self.game_reward, 2)
                             )
                      )
                if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                    print("\nFrequency of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}\n"
                          "Number of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}".
                          format(round(mean(self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list), 2),
                                 sum(self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
                                 )
                          )
                if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                    print("\nFrequency of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n"
                          "Number of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}".
                          format(round(mean(self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list), 2),
                                 sum(self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
                                 )
                          )
                if self.constr_ball_not_in_circle is True:
                    print("\nFrequency of 'ball_not_in_circle' Constraint Violations: {}\n"
                          "Number of 'ball_not_in_circle' Constraint Violations: {}".
                          format(round(mean(self.ball_not_in_circle_constraint_violation_list), 2),
                                 sum(self.ball_not_in_circle_constraint_violation_list)
                                 )
                          )

                if (self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y' and
                     self.X_agent.memory.get_size() > self.batch_size) or \
                    (self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y' and
                     self.Y_agent.memory.get_size() > self.batch_size):
                    if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                        print("\nX_q1_loss: {}\n"
                              "X_q2_loss: {}\n"
                              "X_entropies: {}\n"
                              "X_entropy_loss: {}\n"
                              "X_policy: {}\n"
                              "X_entropy_coefficient: {}"
                              .format(round(float(np.mean(self.X_q1_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.X_q2_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.X_entropies_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.X_entropy_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.X_policy_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.X_entropy_coef_cur_game_per_step_list)), 2)
                                      )
                              )
                        if self.X_agent.clip_grad_norm:
                            print("\nX_q1_gradient_norm_clipped_value: {}\n"
                                  "X_q2_gradient_norm_clipped_value: {}\n"
                                  "X_actor_gradient_norm_clipped_value: {}"
                                  .format(round(float(np.mean(self.X_q1_grad_norm_clipped_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.X_q2_grad_norm_clipped_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.X_actor_grad_norm_clipped_value_cur_game_per_step_list)), 2)
                                          )
                                  )
                        if self.w_constraint_optimization is True:
                            print("\nX_constraint_policy_loss_term_value: {}\n"
                                  "X_constraint_lambda_loss_value: {}\n"
                                  "X_policy_loss_value_wo_constraint_term: {}\n"
                                  "X_constraint_lambda: {}"
                                  .format(round(float(np.mean(self.X_constraint_policy_loss_term_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.X_constraint_lambda_loss_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.X_policy_loss_value_wo_constraint_term_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.X_constraint_lambda_cur_game_per_step_list)), 2)
                                          )
                                  )
                    if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                        print("\nY_q1_loss: {}\n"
                              "Y_q2_loss: {}\n"
                              "Y_entropies: {}\n"
                              "Y_entropy_loss: {}\n"
                              "Y_policy_loss: {}\n"
                              "Y_entropy_coefficient: {}"
                              .format(round(float(np.mean(self.Y_q1_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.Y_q2_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.Y_entropies_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.Y_entropy_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.Y_policy_loss_cur_game_per_step_list)), 2),
                                      round(float(np.mean(self.Y_entropy_coef_cur_game_per_step_list)), 2)
                                      )
                              )
                        if self.Y_agent.clip_grad_norm:
                            print("\nY_q1_gradient_norm_clipped_value: {}\n"
                                  "Y_q2_gradient_norm_clipped_value: {}\n"
                                  "Y_actor_gradient_norm_clipped_value: {}"
                                  .format(round(float(np.mean(self.Y_q1_grad_norm_clipped_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.Y_q2_grad_norm_clipped_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.Y_actor_grad_norm_clipped_value_cur_game_per_step_list)), 2)
                                          )
                                  )
                        if self.w_constraint_optimization is True:
                            print("\nY_constraint_policy_loss_term_value: {}\n"
                                  "Y_constraint_lambda_loss_value: {}\n"
                                  "Y_policy_loss_value_wo_constraint_term: {}\n"
                                  "Y_constraint_lambda: {}"
                                  .format(round(float(np.mean(self.Y_constraint_policy_loss_term_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.Y_constraint_lambda_loss_value_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list)), 2),
                                          round(float(np.mean(self.Y_constraint_lambda_cur_game_per_step_list)), 2)
                                          )
                                  )

                elif self.X_Y_agent is not None and self.X_Y_agent.memory.get_size() > self.batch_size:
                    print_info = [
                        self.X_Y_q1_loss_cur_game_per_step_list, self.X_Y_q2_loss_cur_game_per_step_list,
                        self.X_Y_entropies_cur_game_per_step_list, self.X_Y_entropy_loss_cur_game_per_step_list,
                        self.X_Y_policy_loss_cur_game_per_step_list, self.X_Y_entropy_coef_cur_game_per_step_list
                    ]
                    print_info = [round(float(np.mean(elem)), 2) for elem in print_info]
                    print("\nX_Y_q1_loss: {}\n"
                          "X_Y_q2_loss: {}\n"
                          "X_Y_entropies: {}\n"
                          "X_Y_entropy_loss: {}\n"
                          "X_Y_policy_loss: {}\n"
                          "X_Y_entropy_coefficient: {}".format(*print_info))
                    if self.X_Y_agent.clip_grad_norm:
                        clipping_print_info = [
                            self.X_Y_q1_grad_norm_clipped_value_cur_game_per_step_list,
                            self.X_Y_q2_grad_norm_clipped_value_cur_game_per_step_list,
                            self.X_Y_actor_grad_norm_clipped_value_cur_game_per_step_list
                        ]
                        clipping_print_info = [round(float(np.mean(elem)), 2) for elem in clipping_print_info]
                        print("\nX_Y_q1_gradient_norm_clipped_value: {}\n"
                              "X_Y_q2_gradient_norm_clipped_value: {}\n"
                              "X_Y_actor_gradient_norm_clipped_value: {}".format(*clipping_print_info)
                              )
                    if self.w_constraint_optimization is True:
                        constraint_print_info = [
                            self.X_Y_constraint_policy_loss_term_value_cur_game_per_step_list,
                            self.X_Y_constraint_lambda_loss_value_cur_game_per_step_list,
                            self.X_Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list,
                            self.X_Y_constraint_lambda_cur_game_per_step_list
                        ]
                        constraint_print_info = [round(float(np.mean(elem)), 2) for elem in constraint_print_info]
                        print("\nConstraint_policy_loss_term_value: {}\n"
                              "Constraint_lambda_loss_value: {}\n"
                              "Policy_loss_value_wo_constraint_term: {}\n"
                              "Constraint_lambda: {}".format(*constraint_print_info)
                              )

    def update_time_metrics(self):
        """
        Updates experiment time duration metrics (duration_pause_total, game_duration_list).
        """
        # the net total duration of the experiment
        self.duration_pause_total += self.redundant_end_duration
        game_duration = self.end_game_time - self.start_game_time - self.redundant_end_duration

        # keep track of the game duration
        self.game_duration_list.append(game_duration)

    def update_train_metrics(self):
        """
        Updates train metrics:
            reward_list,
            distance_travel_list,
            length_list,
            <type_of_constraint>_num_constraint_violated_list,
            <type_of_constraint>_freq_constraint_violated_list
        """
        # keep track of the game reward history
        self.reward_list.append(self.game_reward)

        # keep track of the ball's travelled distance
        self.distance_travel_list.append(self.dist_travel)

        # keep track of the game length in steps
        self.length_list.append(self.train_step_counter)

        # keep track of constraint violations
        if self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list is not None:
            self.ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list.append(
                sum(self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
            )
            self.ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list.append(
                sum(self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list) /
                len(self.ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
            )
        if self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list is not None:
            self.ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list.append(
                sum(self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
            )
            self.ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list.append(
                sum(self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list) /
                len(self.ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
            )
        if self.ball_not_in_circle_constraint_violation_list is not None:
            self.ball_not_in_circle_num_constraint_violated_list.append(
                sum(self.ball_not_in_circle_constraint_violation_list)
            )
            self.ball_not_in_circle_freq_constraint_violated_list.append(
                sum(self.ball_not_in_circle_constraint_violation_list) /
                len(self.ball_not_in_circle_constraint_violation_list)
            )

    def update_test_metrics(
            self,
            duration_pause,
            start_game_time,
            game_reward,
            dist_travel,
            step_counter,
            test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list=None,
            test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list=None,
            test_ball_not_in_circle_constraint_violation_list=None
    ):
        """
        Updates test metrics
        """
        end = time.time()

        # Keep track of test game duration
        self.duration_pause_total += duration_pause
        game_duration = end - start_game_time - duration_pause
        self.test_game_duration_list.append(game_duration)

        # keep track of the test game reward history
        self.test_reward_list.append(game_reward)

        # keep track of the ball's travelled distance during testing
        self.test_distance_travel_list.append(dist_travel)

        # keep track of the test game length in steps
        self.test_length_list.append(step_counter)

        # keep track of constraint violations
        flag_to_print_cur_env_layout_index = False
        constr_info_to_print = ""
        if test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list is not None:
            self.test_ball_only_at_the_right_side_wrt_hole_num_constraint_violated_list.append(
                sum(test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
            )
            self.test_ball_only_at_the_right_side_wrt_hole_freq_constraint_violated_list.append(
                sum(test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list) /
                len(test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list)
            )
            if self.test_model is True and \
               self.debug_ is True and \
               sum(test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list) > 0:
                constr_info_to_print += "{} violations of constraint 'ball_only_at_the_right_side_wrt_hole'\n".\
                                        format(sum(test_ball_only_at_the_right_side_wrt_hole_constraint_violation_list))
                flag_to_print_cur_env_layout_index = True
        if test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list is not None:
            self.test_ball_only_at_the_up_side_wrt_hole_num_constraint_violated_list.append(
                sum(test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
            )
            self.test_ball_only_at_the_up_side_wrt_hole_freq_constraint_violated_list.append(
                sum(test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list) /
                len(test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list)
            )
            if self.test_model is True and \
               self.debug_ is True and \
               sum(test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list) > 0:
                constr_info_to_print += "{} violations of constraint 'ball_only_at_the_up_side_wrt_hole'\n".\
                                        format(sum(test_ball_only_at_the_up_side_wrt_hole_constraint_violation_list))
                flag_to_print_cur_env_layout_index = True
        if test_ball_not_in_circle_constraint_violation_list is not None:
            self.test_ball_not_in_circle_num_constraint_violated_list.append(
                sum(test_ball_not_in_circle_constraint_violation_list)
            )
            self.test_ball_not_in_circle_freq_constraint_violated_list.append(
                sum(test_ball_not_in_circle_constraint_violation_list) /
                len(test_ball_not_in_circle_constraint_violation_list)
            )
            if self.test_model is True and \
               self.debug_ is True and \
               sum(test_ball_not_in_circle_constraint_violation_list) > 0:
                constr_info_to_print += "{} violations of constraint 'ball_not_in_circle'\n".\
                                        format(sum(test_ball_not_in_circle_constraint_violation_list))
                flag_to_print_cur_env_layout_index = True
        if flag_to_print_cur_env_layout_index is True:
            constr_info_to_print += "The current env layout index is: {}\n".format(self.env.layout_index)
            print(constr_info_to_print)

    def save_agents_models(self):
        if self.algo == 'SAC' or self.algo == 'PPO':
            # If a new higher score in testing is achieved, save the models
            if self.X_agent is not None and self.Y_agent is not None:
                if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                    self.X_agent.save_models(override=True if self.SL_finetune is False else not self.save_all)
                if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                    self.Y_agent.save_models(override=True if self.SL_finetune is False else not self.save_all)
            elif self.X_Y_agent is not None:
                self.X_Y_agent.save_models(override=True if self.SL_finetune is False else not self.save_all)
            else:
                raise NotImplementedError
        elif self.algo == 'coGAIL':
            self.pi_co.save_model()
            self.discr.save_model()
        else:
            raise NotImplementedError

    def train_store_and_print_info(self):

        if self.algo == "SAC":

            ## Perform 'updates_per_step' number of network updates
            assert not (self.X_agent is not None and self.X_agent.axis_agent == 'X' and
                        self.Y_agent is not None and self.Y_agent.axis_agent == 'Y' and
                        self.X_agent.memory.get_size() != self.Y_agent.memory.get_size())

            if ((self.X_agent is not None and self.X_agent.axis_agent == 'X'
                and self.X_agent.memory.get_size() > self.batch_size) or
                (self.Y_agent is not None and self.Y_agent.axis_agent == 'Y'
                 and self.Y_agent.memory.get_size() > self.batch_size) or
                 (self.X_Y_agent is not None and self.X_Y_agent.memory.get_size() > self.batch_size)
            ) or \
                self.SL_finetune is True:

                # Increase 'i_train_episode' var
                if self.done:
                    self.i_train_episode += 1

                for _ in range(self.updates_per_step):  # Number of updates per step in environment

                    # Update parameters of all the networks
                    training_returns = self.train_networks()

                    # Store losses
                    if self.X_agent is not None and self.Y_agent is not None:
                        if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                            if self.SL_finetune is False:
                                X_q1_loss, \
                                X_q2_loss, \
                                X_entropies, \
                                X_entropy_loss, \
                                X_policy_loss, \
                                X_entropy_coef, \
                                X_q1_grad_norm_clipped_value, \
                                X_q2_grad_norm_clipped_value, \
                                X_actor_grad_norm_clipped_value, \
                                X_constraint_policy_loss_term_value, \
                                X_constraint_lambda_loss_value, \
                                X_policy_loss_value_wo_constraint_term, \
                                X_constraint_lambda = \
                                    training_returns[0]
                                self.X_q1_loss_cur_game_per_step_list.append(X_q1_loss)
                                self.X_q2_loss_cur_game_per_step_list.append(X_q2_loss)
                                self.X_entropies_cur_game_per_step_list.append(X_entropies)
                                self.X_entropy_loss_cur_game_per_step_list.append(X_entropy_loss)
                                self.X_policy_loss_cur_game_per_step_list.append(X_policy_loss)
                                self.X_entropy_coef_cur_game_per_step_list.append(X_entropy_coef)
                                self.X_q1_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    X_q1_grad_norm_clipped_value
                                )
                                self.X_q2_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    X_q2_grad_norm_clipped_value
                                )
                                self.X_actor_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    X_actor_grad_norm_clipped_value
                                )
                                if self.w_constraint_optimization is True:
                                    self.X_constraint_policy_loss_term_value_cur_game_per_step_list.append(
                                        X_constraint_policy_loss_term_value
                                    )
                                    self.X_constraint_lambda_loss_value_cur_game_per_step_list.append(
                                        X_constraint_lambda_loss_value
                                    )
                                    self.X_policy_loss_value_wo_constraint_term_cur_game_per_step_list.append(
                                        X_policy_loss_value_wo_constraint_term
                                    )
                                    self.X_constraint_lambda_cur_game_per_step_list.append(X_constraint_lambda)
                            else:
                                X_constraint_policy_loss_term, X_actor_grad_norm_clipped_value = training_returns[0]
                                self.X_constraint_policy_loss_term_list.append(X_constraint_policy_loss_term)
                                self.X_actor_grad_norm_clipped_value_list.append(X_actor_grad_norm_clipped_value)
                                # print X_agent Supervised Learning finetuning results
                                print("\nX Constraint Policy Loss Term: {}\n"
                                      "X Actor Grad Norm Clipped Value: {}\n"
                                      .format(round(X_constraint_policy_loss_term, 2),
                                              X_actor_grad_norm_clipped_value
                                              )
                                      )
                        if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                            if self.SL_finetune is False:
                                Y_q1_loss, \
                                Y_q2_loss, \
                                Y_entropies, \
                                Y_entropy_loss, \
                                Y_policy_loss, \
                                Y_entropy_coef, \
                                Y_q1_grad_norm_clipped_value, \
                                Y_q2_grad_norm_clipped_value, \
                                Y_actor_grad_norm_clipped_value, \
                                Y_constraint_policy_loss_term_value, \
                                Y_constraint_lambda_loss_value, \
                                Y_policy_loss_value_wo_constraint_term, \
                                Y_constraint_lambda = training_returns[1]
                                self.Y_q1_loss_cur_game_per_step_list.append(Y_q1_loss)
                                self.Y_q2_loss_cur_game_per_step_list.append(Y_q2_loss)
                                self.Y_entropies_cur_game_per_step_list.append(Y_entropies)
                                self.Y_entropy_loss_cur_game_per_step_list.append(Y_entropy_loss)
                                self.Y_policy_loss_cur_game_per_step_list.append(Y_policy_loss)
                                self.Y_entropy_coef_cur_game_per_step_list.append(Y_entropy_coef)
                                self.Y_q1_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    Y_q1_grad_norm_clipped_value
                                )
                                self.Y_q2_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    Y_q2_grad_norm_clipped_value
                                )
                                self.Y_actor_grad_norm_clipped_value_cur_game_per_step_list.append(
                                    Y_actor_grad_norm_clipped_value
                                )
                                if self.w_constraint_optimization is True:
                                    self.Y_constraint_policy_loss_term_value_cur_game_per_step_list.append(
                                        Y_constraint_policy_loss_term_value
                                    )
                                    self.Y_constraint_lambda_loss_value_cur_game_per_step_list.append(
                                        Y_constraint_lambda_loss_value
                                    )
                                    self.Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list.append(
                                        Y_policy_loss_value_wo_constraint_term
                                    )
                                    self.Y_constraint_lambda_cur_game_per_step_list.append(Y_constraint_lambda)
                            else:
                                Y_constraint_policy_loss_term, Y_actor_grad_norm_clipped_value = training_returns[0]
                                self.Y_constraint_policy_loss_term_list.append(Y_constraint_policy_loss_term)
                                self.Y_actor_grad_norm_clipped_value_list.append(Y_actor_grad_norm_clipped_value)
                                # print Y_agent Supervised Learning finetuning results
                                print("\nY Constraint Policy Loss Term: {}\n"
                                      "Y Actor Grad Norm Clipped Value: {}\n"
                                      .format(round(Y_constraint_policy_loss_term, 2), Y_actor_grad_norm_clipped_value))

                    elif self.X_Y_agent is not None:
                        if self.SL_finetune is False:
                            X_Y_q1_loss, \
                            X_Y_q2_loss, \
                            X_Y_entropies, \
                            X_Y_entropy_loss, \
                            X_Y_policy_loss, \
                            X_Y_entropy_coef, \
                            X_Y_q1_grad_norm_clipped_value, \
                            X_Y_q2_grad_norm_clipped_value, \
                            X_Y_actor_grad_norm_clipped_value, \
                            X_Y_constraint_policy_loss_term_value, \
                            X_Y_constraint_lambda_loss_value, \
                            X_Y_policy_loss_value_wo_constraint_term, \
                            X_Y_constraint_lambda = \
                                training_returns
                            self.X_Y_q1_loss_cur_game_per_step_list.append(X_Y_q1_loss)
                            self.X_Y_q2_loss_cur_game_per_step_list.append(X_Y_q2_loss)
                            self.X_Y_entropies_cur_game_per_step_list.append(X_Y_entropies)
                            self.X_Y_entropy_loss_cur_game_per_step_list.append(X_Y_entropy_loss)
                            self.X_Y_policy_loss_cur_game_per_step_list.append(X_Y_policy_loss)
                            self.X_Y_entropy_coef_cur_game_per_step_list.append(X_Y_entropy_coef)
                            self.X_Y_q1_grad_norm_clipped_value_cur_game_per_step_list.append(
                                X_Y_q1_grad_norm_clipped_value
                            )
                            self.X_Y_q2_grad_norm_clipped_value_cur_game_per_step_list.append(
                                X_Y_q2_grad_norm_clipped_value
                            )
                            self.X_Y_actor_grad_norm_clipped_value_cur_game_per_step_list.append(
                                X_Y_actor_grad_norm_clipped_value
                            )
                            if self.w_constraint_optimization is True:
                                self.X_Y_constraint_policy_loss_term_value_cur_game_per_step_list.append(
                                    X_Y_constraint_policy_loss_term_value
                                )
                                self.X_Y_constraint_lambda_loss_value_cur_game_per_step_list.append(
                                    X_Y_constraint_lambda_loss_value
                                )
                                self.X_Y_policy_loss_value_wo_constraint_term_cur_game_per_step_list.append(
                                    X_Y_policy_loss_value_wo_constraint_term
                                )
                                self.X_Y_constraint_lambda_cur_game_per_step_list.append(X_Y_constraint_lambda)
                        else:
                            X_Y_constraint_policy_loss_term, X_Y_actor_grad_norm_clipped_value = training_returns
                            self.X_Y_constraint_policy_loss_term_list.append(X_Y_constraint_policy_loss_term)
                            self.X_Y_actor_grad_norm_clipped_value_list.append(X_Y_actor_grad_norm_clipped_value)
                            # print X_Y_agent Supervised Learning finetuning results
                            print("\nX_Y Constraint Policy Loss Term: {}\n"
                                  "X_Y Actor Grad Norm Clipped Value: {}\n"
                                  .format(round(X_Y_constraint_policy_loss_term, 2),
                                          X_Y_actor_grad_norm_clipped_value
                                          )
                                  )

                    else:
                        raise NotImplementedError

        elif self.algo == "coGAIL":

            # Update all networks
            discr_loss, \
            discr_grad_pen_loss, \
            discr_rewards, \
            value_loss, \
            action_loss, \
            dist_entropy, \
            code_loss, \
            inv_loss, \
            actor_critic_grad_norm_clipped_value, \
            discr_value_loss, \
            env_value_loss,\
            human_action_loss, \
            robot_action_loss, \
            robot_final_constraint_term_loss, \
            constraint_lambda_loss, \
            constraint_lambda = \
                self.train_networks()

            # Store useful information
            self.discr_loss_per_episode_list.append(discr_loss)
            self.discr_grad_pen_loss_per_episode_list.append(discr_grad_pen_loss)
            self.discr_rewards_per_episode_avg_over_games.append(discr_rewards.sum() / self.episode_game_counter)
            self.discr_rewards_per_step.extend(discr_rewards.tolist())
            self.value_loss_per_episode_list.append(value_loss)
            self.action_loss_per_episode_list.append(action_loss)
            self.dist_entropy_per_episode_list.append(dist_entropy)
            self.code_loss_per_episode_list.append(code_loss)
            self.inv_loss_per_episode_list.append(inv_loss)
            self.actor_critic_grad_norm_clipped_value_per_episode_list.append(actor_critic_grad_norm_clipped_value)
            self.code_variable_per_episode_list.append(self.code_variable_per_game_list)
            if self.pi_co.opt_robot_w_env_rewards:
                self.discr_value_loss_per_episode_list.append(discr_value_loss)
                self.env_value_loss_per_episode_list.append(env_value_loss)
                self.human_action_loss_per_episode_list.append(human_action_loss)
                self.robot_action_loss_per_episode_list.append(robot_action_loss)
            if self.constr_ball_only_at_the_right_side_wrt_hole is True or \
               self.constr_ball_only_at_the_up_side_wrt_hole is True or \
               self.constr_ball_not_in_circle is True:
                self.robot_final_constraint_term_loss_per_episode_list.append(robot_final_constraint_term_loss)
                self.constraint_lambda_loss_per_episode_list.append(constraint_lambda_loss)
                self.constraint_lambda_per_episode_list.append(constraint_lambda)

            # Print useful information
            print("\nEpisode: {}\n"
                  "Total games: {}\n"
                  "Total episode games: {}\n"
                  "Total steps: {}\n"
                  "Episode steps: {}\n"
                  "Mean Reward: {}\n"
                  "Mean Discriminator Reward: {}\n"
                  "Discriminator Loss: {}\n"
                  "Discriminator Gradient Penalty: {}\n"
                  "Value Loss: {}\n"
                  "Action Loss: {}\n"
                  "Distribution Entropy: {}\n"
                  "Code Loss: {}\n"
                  "Action Reconstruction Loss: {}\n"
                  "Actor Critic Gradient Norm Clipped Value: {}\n".
                  format(self.i_episode,
                         self.total_games,
                         self.episode_game_counter,
                         self.total_steps,
                         self.episode_step_counter,
                         round(self.episode_reward / self.episode_step_counter, 2),
                         round(np.sum(discr_rewards) / self.episode_step_counter, 2),
                         round(discr_loss, 2),
                         round(discr_grad_pen_loss, 3),
                         round(value_loss, 2),
                         round(action_loss, 2),
                         round(dist_entropy, 2),
                         round(code_loss, 3),
                         round(inv_loss, 3),
                         round(actor_critic_grad_norm_clipped_value, 2)
                         )
                  )
            if self.pi_co.opt_robot_w_env_rewards:
                print("Discriminator Value Loss: {}\n"
                      "Environment Value Loss: {}\n"
                      "Human Action Loss: {}\n"
                      "Robot Action Loss: {}\n".
                      format(round(discr_value_loss, 2),
                             round(env_value_loss, 2),
                             round(human_action_loss, 4),
                             round(robot_action_loss, 4)
                             )
                      )
            if self.constr_ball_only_at_the_up_side_wrt_hole is True or \
               self.constr_ball_only_at_the_right_side_wrt_hole is True or \
               self.constr_ball_not_in_circle is True:
                print("Robot Final Constraint Term Loss: {}\n"
                      "Constraint Lambda Loss: {}\n"
                      "Constraint Lambda: {}\n".
                      format(round(robot_final_constraint_term_loss, 2),
                             round(constraint_lambda_loss, 3),
                             round(constraint_lambda.tolist(), 2)
                             )
                      )
                if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                    print("Frequency of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}\n"
                          "Number of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}\n".
                          format(round(mean(self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list), 2),
                                 sum(self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list)
                                 )
                          )
                if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                    print("Frequency of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n"
                          "Number of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n".
                          format(round(mean(self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list), 2),
                                 sum(self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list)
                                 )
                          )
                if self.constr_ball_not_in_circle is True:
                    print("Frequency of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n"
                          "Number of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n".
                          format(round(mean(self.ball_not_in_circle_episode_constraint_violation_list), 2),
                                 sum(self.ball_not_in_circle_episode_constraint_violation_list)
                                 )
                          )

        elif self.algo == 'PPO':
            training_returns = self.train_networks()

            # Store useful information
            if self.X_agent is not None and self.Y_agent is not None:
                if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                    if self.SL_finetune is False:
                        X_total_loss = training_returns[0][0]
                        X_policy_loss = training_returns[0][1]
                        X_reward_value_loss = training_returns[0][2]
                        X_approx_kl_divs = training_returns[0][3]
                        X_entropy_loss = training_returns[0][4]
                        X_clip_fraction = training_returns[0][5]
                        X_reward_advantage = training_returns[0][6]
                        X_explained_rew_var = training_returns[0][7]
                        X_early_stop_epoch = training_returns[0][8]
                        self.X_total_loss_per_episode_list.append(X_total_loss)
                        self.X_policy_loss_per_episode_list.append(X_policy_loss)
                        self.X_reward_value_loss_per_episode_list.append(X_reward_value_loss)
                        self.X_approx_kl_divs_per_episode_list.append(X_approx_kl_divs)
                        self.X_entropy_loss_per_episode_list.append(X_entropy_loss)
                        self.X_clip_fraction_per_episode_list.append(X_clip_fraction)
                        self.X_reward_advantage_per_episode_list.append(X_reward_advantage)
                        self.X_explained_rew_var_per_episode_list.append(X_explained_rew_var)
                        self.X_early_stop_epoch_per_episode_list.append(X_early_stop_epoch)
                        if self.icrl is True or self.lagrangian is True:
                            X_total_policy_loss = training_returns[0][9]
                            X_dual_nu = training_returns[0][10]
                            X_dual_loss = training_returns[0][11]
                            self.X_total_policy_loss_per_episode_list.append(X_total_policy_loss)
                            self.X_dual_nu_per_episode_list.append(X_dual_nu)
                            self.X_dual_loss_per_episode_list.append(X_dual_loss)
                            if self.icrl is True:
                                X_cost_value_loss = training_returns[0][12]
                                X_cost_advantage = training_returns[0][13]
                                X_explained_cost_var = training_returns[0][14]
                                X_all_cost_per_step = training_returns[0][15]
                                X_cost_advantage_ratio_term = training_returns[0][16]
                                X_cost_loss = training_returns[0][17]
                                self.X_cost_value_loss_per_episode_list.append(X_cost_value_loss)
                                self.X_cost_advantage_per_episode_list.append(X_cost_advantage)
                                self.X_explained_cost_var_per_episode_list.append(X_explained_cost_var)
                                self.X_constr_net_cost_per_step.extend(X_all_cost_per_step)
                                self.X_constr_net_cost_per_episode_avg_over_games.append(
                                    sum(X_all_cost_per_step) / self.episode_game_counter
                                )
                                self.X_cost_advantage_ratio_term_per_episode_list.append(X_cost_advantage_ratio_term)
                                self.X_cost_loss_per_episode_list.append(X_cost_loss)
                            elif self.lagrangian is True:
                                X_lagrangian_constraint_policy_loss_term = training_returns[0][12]
                                self.X_lagrangian_constraint_policy_term_loss_per_episode_list.append(
                                    X_lagrangian_constraint_policy_loss_term
                                )
                    else:
                        X_constraint_policy_loss_term = training_returns[0]
                        self.X_constraint_policy_loss_term_list.append(X_constraint_policy_loss_term)
                        # print X_agent Supervised Learning finetuning results
                        print("\nX Constraint Policy Loss Term: {}\n".format(round(X_constraint_policy_loss_term, 2)))
                if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                    if self.SL_finetune is False:
                        Y_total_loss = training_returns[0][0]
                        Y_policy_loss = training_returns[0][1]
                        Y_reward_value_loss = training_returns[0][2]
                        Y_approx_kl_divs = training_returns[0][3]
                        Y_entropy_loss = training_returns[0][4]
                        Y_clip_fraction = training_returns[0][5]
                        Y_reward_advantage = training_returns[0][6]
                        Y_explained_rew_var = training_returns[0][7]
                        Y_early_stop_epoch = training_returns[0][8]
                        self.Y_total_loss_per_episode_list.append(Y_total_loss)
                        self.Y_policy_loss_per_episode_list.append(Y_policy_loss)
                        self.Y_reward_value_loss_per_episode_list.append(Y_reward_value_loss)
                        self.Y_approx_kl_divs_per_episode_list.append(Y_approx_kl_divs)
                        self.Y_entropy_loss_per_episode_list.append(Y_entropy_loss)
                        self.Y_clip_fraction_per_episode_list.append(Y_clip_fraction)
                        self.Y_reward_advantage_per_episode_list.append(Y_reward_advantage)
                        self.Y_explained_rew_var_per_episode_list.append(Y_explained_rew_var)
                        self.Y_early_stop_epoch_per_episode_list.append(Y_early_stop_epoch)
                        if self.icrl is True or self.lagrangian is True:
                            Y_total_policy_loss = training_returns[0][9]
                            Y_dual_nu = training_returns[0][10]
                            Y_dual_loss = training_returns[0][11]
                            self.Y_total_policy_loss_per_episode_list.append(Y_total_policy_loss)
                            self.Y_dual_nu_per_episode_list.append(Y_dual_nu)
                            self.Y_dual_loss_per_episode_list.append(Y_dual_loss)
                            if self.icrl is True:
                                Y_cost_value_loss = training_returns[0][12]
                                Y_cost_advantage = training_returns[0][13]
                                Y_explained_cost_var = training_returns[0][14]
                                Y_all_cost_per_step = training_returns[0][15]
                                Y_cost_advantage_ratio_term = training_returns[0][16]
                                Y_cost_loss = training_returns[0][17]
                                self.Y_cost_value_loss_per_episode_list.append(Y_cost_value_loss)
                                self.Y_cost_advantage_per_episode_list.append(Y_cost_advantage)
                                self.Y_explained_cost_var_per_episode_list.append(Y_explained_cost_var)
                                self.Y_constr_net_cost_per_step.extend(Y_all_cost_per_step)
                                self.Y_constr_net_cost_per_episode_avg_over_games.append(
                                    sum(Y_all_cost_per_step) / self.episode_game_counter
                                )
                                self.Y_cost_advantage_ratio_term_per_episode_list.append(Y_cost_advantage_ratio_term)
                                self.Y_cost_loss_per_episode_list.append(Y_cost_loss)
                            elif self.lagrangian is True:
                                Y_lagrangian_constraint_policy_loss_term = training_returns[0][12]
                                self.Y_lagrangian_constraint_policy_term_loss_per_episode_list.append(
                                    Y_lagrangian_constraint_policy_loss_term
                                )
                    else:
                        Y_constraint_policy_loss_term = training_returns[0]
                        self.Y_constraint_policy_loss_term_list.append(Y_constraint_policy_loss_term)
                        # print Y_agent Supervised Learning finetuning results
                        print("\nY Constraint Policy Loss Term: {}\n".format(round(Y_constraint_policy_loss_term, 2)))
            elif self.X_Y_agent is not None:
                if self.SL_finetune is False:
                    X_Y_total_loss = training_returns[0]
                    X_Y_policy_loss = training_returns[1]
                    X_Y_reward_value_loss = training_returns[2]
                    X_Y_approx_kl_divs = training_returns[3]
                    X_Y_entropy_loss = training_returns[4]
                    X_Y_clip_fraction = training_returns[5]
                    X_Y_reward_advantage = training_returns[6]
                    X_Y_explained_rew_var = training_returns[7]
                    X_Y_early_stop_epoch = training_returns[8]
                    self.X_Y_total_loss_per_episode_list.append(X_Y_total_loss)
                    self.X_Y_policy_loss_per_episode_list.append(X_Y_policy_loss)
                    self.X_Y_reward_value_loss_per_episode_list.append(X_Y_reward_value_loss)
                    self.X_Y_approx_kl_divs_per_episode_list.append(X_Y_approx_kl_divs)
                    self.X_Y_entropy_loss_per_episode_list.append(X_Y_entropy_loss)
                    self.X_Y_clip_fraction_per_episode_list.append(X_Y_clip_fraction)
                    self.X_Y_reward_advantage_per_episode_list.append(X_Y_reward_advantage)
                    self.X_Y_explained_rew_var_per_episode_list.append(X_Y_explained_rew_var)
                    self.X_Y_early_stop_epoch_per_episode_list.append(X_Y_early_stop_epoch)
                    if self.icrl is True or self.lagrangian is True:
                        X_Y_total_policy_loss = training_returns[9]
                        X_Y_dual_nu = training_returns[10]
                        X_Y_dual_loss = training_returns[11]
                        self.X_Y_total_policy_loss_per_episode_list.append(X_Y_total_policy_loss)
                        self.X_Y_dual_nu_per_episode_list.append(X_Y_dual_nu)
                        self.X_Y_dual_loss_per_episode_list.append(X_Y_dual_loss)
                        if self.icrl is True:
                            X_Y_cost_value_loss = training_returns[12]
                            X_Y_cost_advantage = training_returns[13]
                            X_Y_explained_cost_var = training_returns[14]
                            X_Y_all_cost_per_step = training_returns[15]
                            X_Y_cost_advantage_ratio_term = training_returns[16]
                            X_Y_cost_loss = training_returns[17]
                            self.X_Y_cost_value_loss_per_episode_list.append(X_Y_cost_value_loss)
                            self.X_Y_cost_advantage_per_episode_list.append(X_Y_cost_advantage)
                            self.X_Y_explained_cost_var_per_episode_list.append(X_Y_explained_cost_var)
                            self.X_Y_constr_net_cost_per_step.extend(X_Y_all_cost_per_step)
                            self.X_Y_constr_net_cost_per_episode_avg_over_games.append(
                                sum(X_Y_all_cost_per_step) / self.episode_game_counter
                            )
                            self.X_Y_cost_advantage_ratio_term_per_episode_list.append(X_Y_cost_advantage_ratio_term)
                            self.X_Y_cost_loss_per_episode_list.append(X_Y_cost_loss)
                        elif self.lagrangian is True:
                            X_Y_lagrangian_constraint_policy_loss_term = training_returns[12]
                            self.X_Y_lagrangian_constraint_policy_term_loss_per_episode_list.append(
                                X_Y_lagrangian_constraint_policy_loss_term
                            )
                else:
                    X_Y_constraint_policy_loss_term = training_returns
                    self.X_Y_constraint_policy_loss_term_list.append(X_Y_constraint_policy_loss_term)
                    # print X_Y_agent Supervised Learning finetuning results
                    print("\nX_Y Constraint Policy Loss Term: {}\n".format(round(X_Y_constraint_policy_loss_term, 2)))

            else:
                raise NotImplementedError

            if self.SL_finetune is False:

                # Print useful information
                print("\nEpisode: {}\n"
                      "Total games: {}\n"
                      "Total episode games: {}\n"
                      "Total steps: {}\n"
                      "Episode steps: {}\n"
                      "Mean Reward: {}\n".
                      format(self.i_episode,
                             self.total_games,
                             self.episode_game_counter,
                             self.total_steps,
                             self.episode_step_counter,
                             round(self.episode_reward / self.episode_step_counter, 2))
                      )
                if self.constr_ball_only_at_the_up_side_wrt_hole is True or \
                   self.constr_ball_only_at_the_right_side_wrt_hole is True or \
                   self.constr_ball_not_in_circle is True:
                    if self.constr_ball_only_at_the_right_side_wrt_hole is True:
                        print("Frequency of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}\n"
                              "Number of 'ball_only_at_the_right_side_wrt_hole' Constraint Violations: {}\n".
                              format(round(mean(self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list), 2),
                                     sum(self.ball_only_at_the_right_side_wrt_hole_episode_constraint_violation_list))
                              )
                    if self.constr_ball_only_at_the_up_side_wrt_hole is True:
                        print("Frequency of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n"
                              "Number of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n".
                              format(round(mean(self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list), 2),
                                     sum(self.ball_only_at_the_up_side_wrt_hole_episode_constraint_violation_list))
                              )
                    if self.constr_ball_not_in_circle is True:
                        print("Frequency of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n"
                              "Number of 'ball_only_at_the_up_side_wrt_hole' Constraint Violations: {}\n".
                              format(round(mean(self.ball_not_in_circle_episode_constraint_violation_list), 2),
                                     sum(self.ball_not_in_circle_episode_constraint_violation_list))
                              )

                if self.X_agent is not None and self.Y_agent is not None:
                    if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                        print("\nX Total Loss: {}\n"
                              "X Policy Loss: {}\n"
                              "X Reward Value Loss: {}\n"
                              "X Approximate KL Div: {}\n"
                              "X Entropy Loss: {}\n"
                              "X Clip Fraction: {}\n"
                              "X Reward Advantage: {}\n"
                              "X Explained Reward Variance: {}\n"
                              "X Early Stop Epoch: {}\n".
                              format(round(X_total_loss, 2),
                                     round(X_policy_loss, 2),
                                     round(X_reward_value_loss, 3),
                                     round(X_approx_kl_divs, 2),
                                     round(X_entropy_loss, 2),
                                     round(X_clip_fraction, 2),
                                     round(X_reward_advantage, 2),
                                     round(X_explained_rew_var, 2),
                                     X_early_stop_epoch)
                              )
                        if self.icrl is True or self.lagrangian is True:
                            print("\nX Total Policy Loss: {}\n"
                                  "X Dual Var: {}\n"
                                  "X Dual Var Loss: {}".
                                  format(round(X_total_policy_loss, 2),
                                         round(X_dual_nu, 2),
                                         round(X_dual_loss, 2))
                                  )
                            if self.icrl is True:
                                print("X Cost Policy Loss Term: {}\n"
                                      "X CostAdvantage-Ratio Term: {}\n"
                                      "X Cost Value Loss: {}\n"
                                      "X Cost Advantage: {}\n"
                                      "X Explained Cost Variance: {}\n"
                                      "X Avg Cost: {}\n"
                                      "X Total Cost: {}\n".
                                      format(round(X_cost_loss, 2),
                                             round(X_cost_advantage_ratio_term, 3),
                                             round(X_cost_value_loss, 2),
                                             round(X_cost_advantage, 2),
                                             round(X_explained_cost_var, 2),
                                             round(mean(X_all_cost_per_step), 2),
                                             round(sum(X_all_cost_per_step), 2))
                                      )
                            elif self.lagrangian is True:
                                print("X Constraint Policy Loss Term: {}\n".format(
                                    round(X_lagrangian_constraint_policy_loss_term, 4)
                                ))
                    if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                        print("\nY Total Loss: {}\n"
                              "Y Policy Loss: {}\n"
                              "Y Reward Value Loss: {}\n"
                              "Y Approximate KL Div: {}\n"
                              "Y Entropy Loss: {}\n"
                              "Y Clip Fraction: {}\n"
                              "Y Reward Advantage: {}\n"
                              "Y Explained Reward Variance: {}\n"
                              "Y Early Stop Epoch: {}\n".
                              format(round(Y_total_loss, 2),
                                     round(Y_policy_loss, 2),
                                     round(Y_reward_value_loss, 3),
                                     round(Y_approx_kl_divs, 2),
                                     round(Y_entropy_loss, 2),
                                     round(Y_clip_fraction, 2),
                                     round(Y_reward_advantage, 2),
                                     round(Y_explained_rew_var, 2),
                                     Y_early_stop_epoch)
                              )
                        if self.icrl is True or self.lagrangian is True:
                            print("\nY Total Policy Loss: {}\n"
                                  "Y Dual Var: {}\n"
                                  "Y Dual Var Loss: {}".
                                  format(round(Y_total_policy_loss, 2),
                                         round(Y_dual_nu, 2),
                                         round(Y_dual_loss, 2))
                                  )
                            if self.icrl is True:
                                print("Y Cost Policy Loss Term: {}\n"
                                      "Y CostAdvantage-Ratio Term: {}\n"
                                      "Y Cost Value Loss: {}\n"
                                      "Y Cost Advantage: {}\n"
                                      "Y Explained Cost Variance: {}\n"
                                      "Y Avg Cost: {}\n"
                                      "Y Total Cost: {}\n".
                                      format(round(Y_cost_loss, 2),
                                             round(Y_cost_advantage_ratio_term, 3),
                                             round(Y_cost_value_loss, 2),
                                             round(Y_cost_advantage, 2),
                                             round(Y_explained_cost_var, 2),
                                             round(mean(Y_all_cost_per_step), 2),
                                             round(sum(Y_all_cost_per_step), 2))
                                      )
                            elif self.lagrangian is True:
                                print("Y Constraint Policy Loss Term: {}\n".format(
                                    round(Y_lagrangian_constraint_policy_loss_term, 4)
                                ))

                if self.X_Y_agent is not None:
                    print("\nX_Y Total Loss: {}\n"
                          "X_Y Policy Loss: {}\n"
                          "X_Y Reward Value Loss: {}\n"
                          "X_Y Approximate KL Div: {}\n"
                          "X_Y Entropy Loss: {}\n"
                          "X_Y Clip Fraction: {}\n"
                          "X_Y Reward Advantage: {}\n"
                          "X_Y Explained Reward Variance: {}\n"
                          "X_Y Early Stop Epoch: {}\n".
                          format(round(X_Y_total_loss, 2),
                                 round(X_Y_policy_loss, 2),
                                 round(X_Y_reward_value_loss, 3),
                                 round(X_Y_approx_kl_divs, 2),
                                 round(X_Y_entropy_loss, 2),
                                 round(X_Y_clip_fraction, 2),
                                 round(X_Y_reward_advantage, 2),
                                 round(X_Y_explained_rew_var, 2),
                                 X_Y_early_stop_epoch
                                 )
                          )
                    if self.icrl is True or self.lagrangian is True:
                        print("\nX_Y Total Policy Loss: {}\n"
                              "X_Y Dual Var: {}\n"
                              "X_Y Dual Var Loss: {}".
                              format(round(X_Y_total_policy_loss, 2),
                                     round(X_Y_dual_nu, 2),
                                     round(X_Y_dual_loss, 2)
                                     )
                              )
                        if self.icrl is True:
                            print("X_Y Cost Policy Loss Term: {}\n"
                                  "X_Y CostAdvantage-Ratio Term: {}\n"
                                  "X_Y Cost Value Loss: {}\n"
                                  "X_Y Cost Advantage: {}\n"
                                  "X_Y Explained Cost Variance: {}\n"
                                  "X_Y Avg Cost: {}\n"
                                  "X_Y Total Cost: {}\n".
                                  format(round(X_Y_cost_loss, 2),
                                         round(X_Y_cost_advantage_ratio_term, 3),
                                         round(X_Y_cost_value_loss, 2),
                                         round(X_Y_cost_advantage, 2),
                                         round(X_Y_explained_cost_var, 2),
                                         round(mean(X_Y_all_cost_per_step), 2),
                                         round(sum(X_Y_all_cost_per_step), 2)
                                         )
                                  )
                        elif self.lagrangian is True:
                            print("X_Y Constraint Policy Loss Term: {}\n".format(
                                round(X_Y_lagrangian_constraint_policy_loss_term, 4)
                            ))

        else:
            raise NotImplementedError

    def train_networks(self):
        """
        Train the networks of agent(s)
        :return: training results
        """

        if self.algo == 'SAC':
            if self.X_agent is not None and self.Y_agent is not None:
                X_agent_training_results = None
                Y_agent_training_results = None
                if self.X_agent is not None and not self.X_agent.axis_agent == 'X_Y':
                    if self.SL_finetune is False:
                        X_agent_training_results = self.X_agent.learn(soft_update_target=True)
                    else:
                        X_agent_training_results = self.X_agent.SL_finetuning()
                if self.Y_agent is not None and not self.Y_agent.axis_agent == 'X_Y':
                    if self.SL_finetune is False:
                        Y_agent_training_results = self.Y_agent.learn(soft_update_target=True)
                    else:
                        Y_agent_training_results = self.Y_agent.SL_finetuning()
                training_results = [X_agent_training_results, Y_agent_training_results]
            elif self.X_Y_agent is not None:
                if self.SL_finetune is False:
                    training_results = self.X_Y_agent.learn(soft_update_target=True)
                else:
                    training_results = self.X_Y_agent.SL_finetuning()
            else:
                raise NotImplementedError

        elif self.algo == 'coGAIL':
            training_results = self.agents.learn(self.i_episode)

        elif self.algo == 'PPO':
            if self.X_agent is not None and self.Y_agent is not None:
                X_agent_training_results = None
                Y_agent_training_results = None
                if self.X_agent is not None and self.X_agent.axis_agent == 'X':
                    if self.SL_finetune is False:
                        X_agent_training_results = self.X_agent.ppo.train_()
                    else:
                        X_agent_training_results = self.X_agent.ppo.SL_finetuning()
                if self.Y_agent is not None and self.Y_agent.axis_agent == 'Y':
                    if self.SL_finetune is False:
                        Y_agent_training_results = self.Y_agent.ppo.train_()
                    else:
                        Y_agent_training_results = self.Y_agent.ppo.SL_finetuning()
                training_results = [X_agent_training_results, Y_agent_training_results]
            elif self.X_Y_agent is not None:
                if self.SL_finetune is False:
                    training_results = self.X_Y_agent.ppo.train_()
                else:
                    training_results = self.X_Y_agent.ppo.SL_finetuning()
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        return training_results

    def get_agent_action(self, prev_observation, greedy=True, random_=None):
        """
        Retrieves the original action from the agent and converts it into an environment-compatible one.
        Especially discrete SAC predicts actions using a categorical distribution, so we have to map these actions into
        both negative and positive numbers.
        :param prev_observation: Observations to be fed to agents. numpy array for SAC,
                                 torch.FloatTensor for co-GAIL
        :param greedy: if True, takes a greedy action, if False, sample an action, if None, random should be True
        :param random_: if Yes, random actions (using np.random) are executed to avoid the initial bias
                        of agents. Otherwise, there should be None.
        :return: the environment_compatible agent's action, the original agent's action
        """

        returns = self.compute_agent_action(prev_observation, greedy=greedy, random_=random_)
        X_agent_action, Y_agent_action = returns[0], returns[1]

        # agents' actions ready to be used from the environment
        env_X_agent_action = None
        env_Y_agent_action = None
        if X_agent_action is not None:
            env_X_agent_action = get_env_action(X_agent_action)
        if Y_agent_action is not None:
            env_Y_agent_action = get_env_action(Y_agent_action)

        if self.algo == "SAC" or self.algo is None:
            return env_X_agent_action, env_Y_agent_action, X_agent_action, Y_agent_action
        elif self.algo == "coGAIL":
            value, action, action_log_prob = returns[2], returns[3], returns[4]
            return env_X_agent_action, env_Y_agent_action, X_agent_action, Y_agent_action, value, action, action_log_prob
        elif self.algo == "PPO":
            return env_X_agent_action, env_Y_agent_action, returns[2]

    def compute_agent_action(self, observation, greedy=True, random_=None):
        """
        Computes agent's next action based on the observation given.
        It returns a random action from the legal ones
        when the randomness criterion (a random number) is lower than the randomness threshold (epsilon).
        :param observation: the observation, based on which to calculate action
        :param greedy: if True, takes a greedy action, if False, sample an action, if None, random should be True
        :param random_: if Yes, random actions (using np.random) will be executed to avoid the initial bias
                        of agents. Otherwise, there should be None.
        :return: agent's action
        """

        X_agent_action = None
        Y_agent_action = None
        X_Y_agent_action = None

        if self.algo == 'SAC':
            obs = observation[:8]
            if random_:
                if self.X_agent is not None:
                    if not self.X_agent.axis_agent == "X_Y":
                        # get a random action for X_agent
                        X_agent_action = np.random.randint(self.X_agent.n_actions)
                    else:
                        # get X_agent's decision on the next action
                        X_agent_action = self.X_agent.actor.greedy_act(obs)
                        X_agent_action = transform_action_to_actions(
                            X_agent_action,
                            self.env.action_space.actions_number
                        )[0]
                if self.Y_agent is not None:
                    if not self.Y_agent.axis_agent == "X_Y":
                        # get a random action for Y_agent
                        Y_agent_action = np.random.randint(self.Y_agent.n_actions)
                    else:
                        # get Y_agent's decision on the next action
                        Y_agent_action = self.Y_agent.actor.greedy_act(obs)
                        Y_agent_action = transform_action_to_actions(
                            Y_agent_action,
                            self.env.action_space.actions_number
                        )[1]
                if self.X_Y_agent is not None:
                    # get a random action for X_Y_agent
                    X_Y_agent_action = np.random.randint(self.X_Y_agent.n_actions)
            # check if greedy is False and sample actions
            elif not greedy and greedy is not None:
                if self.X_agent is not None:
                    if not self.X_agent.axis_agent == "X_Y":
                        # sample X_agent's next action
                        X_agent_action = self.X_agent.actor.sample_act(obs)
                    else:
                        # get X_agent's decision on the next action
                        X_agent_action = self.X_agent.actor.greedy_act(obs)
                        X_agent_action = transform_action_to_actions(
                            X_agent_action,
                            self.env.action_space.actions_number
                        )[0]
                if self.Y_agent is not None:
                    if not self.Y_agent.axis_agent == "X_Y":
                        # sample Y_agent's next action
                        Y_agent_action = self.Y_agent.actor.sample_act(obs)
                    else:
                        # get Y_agent's decision on the next action
                        Y_agent_action = self.Y_agent.actor.greedy_act(obs)
                        Y_agent_action = transform_action_to_actions(
                            Y_agent_action,
                            self.env.action_space.actions_number
                        )[1]
                if self.X_Y_agent is not None:
                    # sample X_Y_agent's next action
                    X_Y_agent_action = self.X_Y_agent.actor.sample_act(obs)
            # use agents' greedy actions
            elif greedy:
                if self.X_agent is not None:
                    # get X_agent's decision on the next action
                    X_agent_action = self.X_agent.actor.greedy_act(obs)
                    if self.X_agent.axis_agent == "X_Y":
                        X_agent_action = transform_action_to_actions(
                            X_agent_action,
                            self.env.action_space.actions_number
                        )[0]
                if self.Y_agent is not None:
                    # get Y_agent's decision on the next action
                    Y_agent_action = self.Y_agent.actor.greedy_act(obs)
                    if self.Y_agent.axis_agent == "X_Y":
                        Y_agent_action = transform_action_to_actions(
                            Y_agent_action,
                            self.env.action_space.actions_number
                        )[1]
                if self.X_Y_agent is not None:
                    # get X_Y_agent's decision on the next action
                    X_Y_agent_action = self.X_Y_agent.actor.greedy_act(obs)
            else:
                print('Not valid condition in action selection. Case 1 !!!')
                exit(0)

            if self.X_agent is not None or self.Y_agent is not None:
                return X_agent_action, Y_agent_action
            elif self.X_Y_agent is not None:
                X_axis_action, Y_axis_action = transform_action_to_actions(
                    X_Y_agent_action,
                    self.env.action_space.actions_number
                )
                return X_axis_action, Y_axis_action
            else:
                raise NotImplementedError

        elif self.algo == 'PPO':

            obs = observation[:, :8]

            if random_ is True:
                print('"random_" is not a valid action selection method for PPO algorithm !!!')

            # check if greedy is not None and sample actions
            elif greedy is True or greedy is False:

                with (torch.no_grad()):

                    if self.X_agent is not None or self.Y_agent is not None:
                        X_agent_action, X_agent_reward_values, X_agent_cost_values, X_agent_log_probs, X_returns = \
                            None, None, None, None, [None]
                        Y_agent_action, Y_agent_reward_values, Y_agent_cost_values, Y_agent_log_probs, Y_returns = \
                            None, None, None, None, [None]

                        if self.X_agent is not None:
                            if self.X_agent.axis_agent == "X":
                                # sample X_agent's next action
                                X_returns = self.X_agent.ppo.policy.forward(obs, deterministic=greedy)
                                X_agent_action = X_returns[0][0].item()
                                X_agent_reward_values, X_agent_cost_values, X_agent_log_probs = X_returns[1:]
                            else:
                                # get X_agent's decision on the next action
                                X_agent_action = self.X_agent.ppo.policy.predict(obs, deterministic=True)
                                X_agent_action = transform_action_to_actions(
                                    X_agent_action[0].item(),
                                    self.env.action_space.actions_number
                                )[0]
                        if self.Y_agent is not None:
                            if self.Y_agent.axis_agent == "Y":
                                # sample Y_agent's next action
                                Y_returns = self.Y_agent.ppo.policy.forward(obs, deterministic=greedy)
                                Y_agent_action = Y_returns[0][0].item()
                                Y_agent_reward_values, Y_agent_cost_values, Y_agent_log_probs = Y_returns[1:]
                            else:
                                # get Y_agent's decision on the next action
                                Y_agent_action = self.Y_agent.ppo.policy.predict(obs, deterministic=True)
                                Y_agent_action = transform_action_to_actions(
                                    Y_agent_action[0].item(),
                                    self.env.action_space.actions_number
                                )[0]

                        return X_agent_action, \
                               Y_agent_action, \
                               [
                                   X_returns[0],
                                   Y_returns[0],
                                   X_agent_reward_values,
                                   X_agent_cost_values,
                                   X_agent_log_probs,
                                   Y_agent_reward_values,
                                   Y_agent_cost_values,
                                   Y_agent_log_probs
                               ]

                    elif self.X_Y_agent is not None:
                        # sample X_Y_agent's next action
                        X_Y_returns = self.X_Y_agent.ppo.policy.forward(obs, deterministic=greedy)
                        X_axis_action, Y_axis_action = transform_action_to_actions(
                            X_Y_returns[0][0].item(),
                            self.env.action_space.actions_number
                        )
                        X_Y_agent_action, X_Y_agent_reward_values, X_Y_agent_cost_values, X_Y_agent_log_probs = \
                            X_Y_returns

                        return X_axis_action, \
                               Y_axis_action, \
                               [
                                   X_Y_agent_action,
                                   X_Y_agent_reward_values,
                                   X_Y_agent_cost_values,
                                   X_Y_agent_log_probs
                               ]

                    else:
                        raise NotImplementedError

            else:
                print('Not valid condition in action selection. Case 2 !!!')
                exit(0)

        elif self.algo == 'coGAIL':

            obs = observation[0][:, :8]
            code = observation[1]

            if random_:
                print('"random_" is not a valid action selection method for co-GAIL algorithm !!!')

            # check if greedy is not None and sample actions
            elif greedy is True or greedy is False:

                with torch.no_grad():

                    value, action, action_log_prob = self.actor_critic.act(obs, code, deterministic=greedy)

                    assert len(action.size()) == 2 and action.size(0) == 1 and action.size(1) == 2

                    X_agent_action = action[0, 0].detach().cpu().numpy() if self.X_agent is not None else None
                    Y_agent_action = action[0, 1].detach().cpu().numpy() if self.Y_agent is not None else None

            else:
                print('Not valid condition in action selection. Case 3 !!!')
                exit(0)

            return X_agent_action, Y_agent_action, value, action, action_log_prob

        else:
            return X_agent_action, Y_agent_action

