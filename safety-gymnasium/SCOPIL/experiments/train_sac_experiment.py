import itertools
from typing import List

import numpy as np

from .train_experiment import TrainExperiment


class SACExperiment(TrainExperiment):
    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):

        super().__init__(environment, agent, config, file_results_dir, seed)

        # Retrieve information from the config file for training
        self.start_steps = config['SAC']['start_steps']
        self.num_steps = config['SAC']['num_steps']
        self.batch_size = config['SAC']['batch_size']
        self.update_every_steps = config['SAC']['update_every_steps']
        self.gradient_steps = config['SAC']['gradient_steps']
        self.use_sde = config['SAC']['use_sde']
        self.sde_sample_freq = config['SAC']['sde_sample_freq']
        self.w_constraint_optimization = config['SAC']['w_constraint_optimization']
        self.clip_grad_norm = config['SAC']['clip_grad_norm']
        self.pretrain = config['SAC']['pretrain']

        # Initialize lists to keep track of information and variables during training
        self.i_train_episode = 0
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.ent_coef_loss_list = []
        self.entr_coef_list = []
        self.actor_loss_cur_game_list = []
        self.critic_loss_cur_game_list = []
        self.ent_coef_loss_cur_game_list = []
        self.entr_coef_cur_game_list = []
        self.actor_loss_avg_per_log_interval = []
        self.critic_loss_avg_per_log_interval = []
        self.ent_coef_loss_avg_per_log_interval = []
        self.entr_coef_avg_per_log_interval = []
        if self.clip_grad_norm is True:
            self.grad_norm_clipped_list = []
            self.grad_norm_clipped_cur_game_list = []
            self.grad_norm_clipped_avg_per_log_interval = []
        if self.w_constraint_optimization is True:
            self.constraint_policy_loss_term_value_list = []
            self.constraint_lambda_loss_value_list = []
            self.policy_loss_value_wo_constraint_term_list = []
            self.constraint_lambda_list = []
            self.constraint_policy_loss_term_value_cur_game_list = []
            self.constraint_lambda_loss_value_cur_game_list = []
            self.policy_loss_value_wo_constraint_term_cur_game_list = []
            self.constraint_lambda_cur_game_list = []
            self.constraint_policy_loss_term_value_avg_per_log_interval = []
            self.constraint_lambda_loss_value_avg_per_log_interval = []
            self.policy_loss_value_wo_constraint_term_avg_per_log_interval = []
            self.constraint_lambda_avg_per_log_interval = []
            if self.pretrain is True:
                self.pretrain_mse_losses = []
                self.pretrain_nll_losses = []
                self.pretrain_losses = []
                self.pretrain_log_probs = []
                self.pretrain_probs = []
                self.pretrain_grad_norms_clipped = []

    def train(self):
        """
        Trains a SAC agent with or without constraints' optimization
        """

        # Pretrain the agent with the given demonstrations
        if self.pretrain is True:
            pretrain_logs = self.agent.pretrain()
            self.pretrain_log(*pretrain_logs)

        # RL training Loop
        for self.i_episode in itertools.count(1):

            # At the beginning of each episode, initialize the environment and the variables
            self.initialize_game_var_train()

            print("\nEpisode: " + str(self.i_episode))

            # Start the episode
            while not self.done:

                # Get action
                action, buffer_action = self.agent.sample_action(self.observation, self.total_steps)

                # Environment step
                self.next_observation, self.reward, self.cost, self.done, self.truncated, self.info = \
                    self.env_step(action)

                # Update variables after step
                self.train_update_per_step_vars()

                # add experiences to buffers
                self.save_experience(
                    [
                        self.observation,
                        self.next_observation,
                        buffer_action,
                        self.reward,
                        self.fixed_done,
                        self.done,
                        self.truncated,
                        self.info
                    ]
                )

                # set the observation for the next step
                self.observation = self.next_observation.copy()

                # Train the networks and store the corresponding info
                self.train_store_and_print_info()

            ## End of game

            # Update, store and print info for the current game. Also, print avg logs
            self.train_game_logging()

            # Testing
            self.test_during_training()

            # Stop training
            if self.total_steps >= self.num_steps:
                break

        # Close the environment
        self.env.close()

    def save_experience(self, data):
        """
        Saves an interaction to the replay buffer of the agent.
        :param data: List with data of the interaction to be stored in the Replay Buffer.
            The list has the following elements:
            1) observation
            2) next_observation
            3) action
            4) reward
            5) fixed done
            6) done
            7) truncated
            8) info
        """
        self.agent.replay_buffer.add(*data)

    def initialize_game_var_train(self):

        super().initialize_game_var_train()

        # Reset the noise for the SDE
        if self.use_sde:
            self.agent.actor.reset_noise(1)

        # Initialize lists for training details
        self.actor_loss_cur_game_list = []
        self.critic_loss_cur_game_list = []
        self.ent_coef_loss_cur_game_list = []
        self.entr_coef_cur_game_list = []
        if self.w_constraint_optimization is True:
            self.constraint_policy_loss_term_value_cur_game_list = []
            self.constraint_lambda_loss_value_cur_game_list = []
            self.policy_loss_value_wo_constraint_term_cur_game_list = []
            self.constraint_lambda_cur_game_list = []
        if self.clip_grad_norm is True:
            self.grad_norm_clipped_cur_game_list = []

    def train_update_per_step_vars(self):
        """
        Update variables after each step
        """

        super().train_update_per_step_vars()

        # keep track of the overall step number
        self.total_steps += 1

        # keep track of the step number for each game
        self.train_step_counter += 1

        # Sample a new noise matrix if needed
        if self.use_sde and self.sde_sample_freq > 0 and self.train_step_counter % self.sde_sample_freq == 0:
            self.agent.actor.reset_noise(1)

    def train_store_and_print_info(self):
        """
        Perform network updates
        """

        if self.total_steps >= self.start_steps:

            # Increase 'i_train_episode' var
            if self.done:
                self.i_train_episode += 1

            # Update networks' parameters
            training_returns = self.train_networks()

            # Store train metrics
            self.actor_loss_cur_game_list.append(training_returns[0])
            self.critic_loss_cur_game_list.append(training_returns[1])
            self.ent_coef_loss_cur_game_list.append(training_returns[2])
            self.entr_coef_cur_game_list.append(training_returns[3])
            if self.w_constraint_optimization is True:
                self.constraint_policy_loss_term_value_cur_game_list.append(training_returns[4])
                self.constraint_lambda_loss_value_cur_game_list.append(training_returns[5])
                self.policy_loss_value_wo_constraint_term_cur_game_list.append(training_returns[6])
                self.constraint_lambda_cur_game_list.append(training_returns[7])
            if self.clip_grad_norm is True:
                self.grad_norm_clipped_cur_game_list.append(training_returns[8])

    def train_mode(self):
        """
        Set the agent in training mode
        """
        self.agent.policy.set_training_mode(True)

    def eval_mode(self):
        """
        Set the agent in evaluation mode
        """
        self.agent.policy.set_training_mode(False)

    def train_game_logging(self):

        super().train_game_logging()

        # Store all game train metrics
        if self.total_steps >= self.start_steps:
            self.actor_loss_list.append(np.mean(self.actor_loss_cur_game_list))
            self.critic_loss_list.append(np.mean(self.critic_loss_cur_game_list))
            self.ent_coef_loss_list.append(np.mean(self.ent_coef_loss_cur_game_list))
            self.entr_coef_list.append(np.mean(self.entr_coef_cur_game_list))
            if self.w_constraint_optimization is True:
                self.constraint_policy_loss_term_value_list.append(
                    np.mean(self.constraint_policy_loss_term_value_cur_game_list)
                )
                self.constraint_lambda_loss_value_list.append(
                    np.mean(self.constraint_lambda_loss_value_cur_game_list)
                )
                self.policy_loss_value_wo_constraint_term_list.append(
                    np.mean(self.policy_loss_value_wo_constraint_term_cur_game_list)
                )
                self.constraint_lambda_list.append(np.mean(self.constraint_lambda_cur_game_list))
            if self.clip_grad_norm is True:
                self.grad_norm_clipped_list.append(np.mean(self.grad_norm_clipped_cur_game_list))

        ## logging per interval
        if (
                self.total_steps >= self.start_steps and
                (self.i_episode % self.log_interval == 0 or self.i_episode == 1)
        ):

            # Calculate and store per_log_interval values
            actor_loss_avg_per_log_interval = np.mean(self.actor_loss_list[-self.log_interval:])
            critic_loss_avg_per_log_interval = np.mean(self.critic_loss_list[-self.log_interval:])
            ent_coef_loss_avg_per_log_interval = np.mean(self.ent_coef_loss_list[-self.log_interval:])
            entr_coef_avg_per_log_interval = np.mean(self.entr_coef_list[-self.log_interval:])
            self.actor_loss_avg_per_log_interval.append(actor_loss_avg_per_log_interval)
            self.critic_loss_avg_per_log_interval.append(critic_loss_avg_per_log_interval)
            self.ent_coef_loss_avg_per_log_interval.append(ent_coef_loss_avg_per_log_interval)
            self.entr_coef_avg_per_log_interval.append(entr_coef_avg_per_log_interval)
            if self.w_constraint_optimization is True:
                constraint_policy_loss_term_avg_per_log_interval = \
                    np.mean(self.constraint_policy_loss_term_value_list[-self.log_interval:])
                constraint_lambda_loss_avg_per_log_interval = \
                    np.mean(self.constraint_lambda_loss_value_list[-self.log_interval:])
                policy_loss_wo_constraint_term_avg_per_log_interval = \
                    np.mean(self.policy_loss_value_wo_constraint_term_list[-self.log_interval:])
                constraint_lambda_avg_per_log_interval = np.mean(self.constraint_lambda_list[-self.log_interval:])
                self.constraint_policy_loss_term_value_avg_per_log_interval.append(
                    constraint_policy_loss_term_avg_per_log_interval
                )
                self.constraint_lambda_loss_value_avg_per_log_interval.append(
                    constraint_lambda_loss_avg_per_log_interval
                )
                self.policy_loss_value_wo_constraint_term_avg_per_log_interval.append(
                    policy_loss_wo_constraint_term_avg_per_log_interval
                )
                self.constraint_lambda_avg_per_log_interval.append(constraint_lambda_avg_per_log_interval)
            if self.clip_grad_norm is True:
                grad_norm_clipped_avg_per_log_interval = np.mean(self.grad_norm_clipped_list[-self.log_interval:])
                self.grad_norm_clipped_avg_per_log_interval.append(grad_norm_clipped_avg_per_log_interval)

            # Print per_log_interval values
            print(
                "\nAvg actor_loss: {}\n"
                "Avg critic_loss: {}\n"
                "Avg ent_coef_loss: {}\n"
                "Avg entr_coef: {}\n".format(
                    round(float(actor_loss_avg_per_log_interval), 2),
                    round(float(critic_loss_avg_per_log_interval), 2),
                    round(float(ent_coef_loss_avg_per_log_interval), 2),
                    round(float(entr_coef_avg_per_log_interval), 2)
                )
            )
            if self.clip_grad_norm is True:
                print("Avg grad_norm_clipped: {}\n".format(round(float(grad_norm_clipped_avg_per_log_interval), 2)))
            if self.w_constraint_optimization is True:
                print(
                    "Avg constraint_policy_loss_term: {}\n"
                    "Avg constraint_lambda_loss: {}\n"
                    "Avg policy_loss_wo_constraint_term: {}\n"
                    "Avg constraint_lambda: {}\n".format(
                        round(float(constraint_policy_loss_term_avg_per_log_interval), 2),
                        round(float(constraint_lambda_loss_avg_per_log_interval), 2),
                        round(float(policy_loss_wo_constraint_term_avg_per_log_interval), 2),
                        round(float(constraint_lambda_avg_per_log_interval), 2)
                    )
                )

        if self.debug_:
            # TODO: Print useful information
            pass

    def train_networks(self):
        training_returns = self.agent.train()

        return training_returns

    def get_agent_action_test(self, obs):
        """
        Get the agent action for testing
        :param obs: np.array, observation of the agent
        :return: np.array, action
        """

        return self.agent.predict(obs, deterministic=True)

    def save_agent_models(self, prefix_model_name):
        """
        Save the agent models
        :param prefix_model_name: str, prefix name for the models
        """
        self.agent.save(prefix_model_name, self.file_results_dir)

    def pretrain_log(
            self,
            mse_losses: List[float],
            nll_losses: List[float],
            losses: List[float],
            log_probs: List[float],
            probs: List[float],
            grad_norms_clipped: List[float]
    ):
        """
        Log the pretraining results
        """

        # Store the pretraining results
        self.pretrain_mse_losses = mse_losses
        self.pretrain_nll_losses = nll_losses
        self.pretrain_losses = losses
        self.pretrain_log_probs = log_probs
        self.pretrain_probs = probs
        self.pretrain_grad_norms_clipped = grad_norms_clipped
