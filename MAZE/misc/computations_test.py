import pylab as pl
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statistics import mean


class DummyParam(nn.Module):
    def __init__(self, input_=1.):
        super(DummyParam, self).__init__()

        self.dummy_tensor = nn.Parameter(torch.from_numpy(np.array([input_])), requires_grad=True)

    def forward(self):
        return self.dummy_tensor


torch.set_printoptions(precision=16)


def Adam_pytorch_opt(var1=5., var2=15., g_dif=0., lr=0.0003, eps=1e-4, n_steps=5):

    print('(##############pytorch Adam################')
    dummy1 = DummyParam(var1)
    dummy2 = DummyParam(var2)
    g_diff = torch.from_numpy(np.array([g_dif]))
    dummy1_optimizer = torch.optim.Adam(dummy1.parameters(), lr=lr, eps=eps)

    for t in range(n_steps):

        dummy1_v = dummy1()
        dummy2_v = dummy2()
        dummy3 = dummy1_v*dummy2_v.detach()
        print()
        print('t: ' + str(t))

        # print('dummy 1 param:')
        # for param in dummy1.parameters():
        #     print(param)

        dummy1_optimizer.zero_grad()
        dummy3.backward()

        # print('dummy 1 grads')
        # for par in dummy1.parameters():
        #     print(par.grad)

        dummy1_optimizer.step()

        print('dummy 1 param: ')
        for param in dummy1.parameters():
            print(param)

        if t == 0:
            state_dict = dummy2.state_dict()
            state_dict['dummy_tensor'] = dummy2_v - g_diff
            dummy2.load_state_dict(state_dict)


def Adam_custom_opt(var1=5., var2=15., g_dif=0., lr=0.0003, eps=1e-4,
                    m_prev=0., u_prev=0., b1=0.9, b2=0.999, t_start=1, t_stop=6):
    print()
    print('###############custom Adam################')
    m_prev = torch.tensor([m_prev], dtype=torch.float64)
    u_prev = torch.tensor([u_prev], dtype=torch.float64)
    b1 = torch.tensor([b1], dtype=torch.float64)
    b2 = torch.tensor([b2], dtype=torch.float64)
    lr = torch.tensor([lr], dtype=torch.float64)
    eps = torch.tensor([eps], dtype=torch.float64)
    one = torch.tensor([1.0], dtype=torch.float64)

    var = torch.tensor([var1], dtype=torch.float64)
    g = torch.tensor([var2], dtype=torch.float64)
    g_dif = torch.tensor([g_dif], dtype=torch.float64)

    t_start = t_start
    t_stop = t_stop
    for t in range(t_start, t_stop):

        print('\nt: ' + str(t))

        m = (b1 * m_prev) + ((one - b1) * g)
        u = (b2 * u_prev) + ((one - b2) * torch.pow(g, 2))
        mhat = m / (one - torch.pow(b1, t))
        uhat = u / (one - torch.pow(b2, t))

        ratio = (mhat / (torch.sqrt(uhat) + eps))
        var = var - (lr * ratio)

        print('m: ' + str(m))
        print('u: ' + str(u))
        print('mhat: ' + str(mhat))
        print('uhat: ' + str(uhat))
        print('ratio: ' + str(ratio))
        print('var: ' + str(var))

        m_prev = m
        u_prev = u

        if t == 1:
            g = g - g_dif


def SGD_pytorch_opt(var1=5., var2=15., g_dif=0., lr=0.0003, n_steps=5):

    print()
    print('############pytorch SGD###############')
    dummy4 = DummyParam(var1)
    dummy5 = DummyParam(var2)
    g_diff_ = torch.from_numpy(np.array([g_dif]))
    dummy4_optimizer = torch.optim.SGD(dummy4.parameters(), lr=lr)

    for t in range(n_steps):

        dummy4_v = dummy4()
        dummy5_v = dummy5()
        dummy6 = dummy4_v*dummy5_v.detach()
        print()
        print('t: ' + str(t))

        # print('dummy 4 param:')
        # for param in dummy1.parameters():
        #     print(param)

        dummy4_optimizer.zero_grad()
        dummy6.backward()

        # print('dummy 4 grads')
        # for par in dummy4.parameters():
        #     print(par.grad)

        dummy4_optimizer.step()

        print('dummy 4 param: ')
        for param in dummy4.parameters():
            print(param)

        if t == 0:
            state_dict = dummy5.state_dict()
            state_dict['dummy_tensor'] = dummy5_v - g_diff_
            dummy5.load_state_dict(state_dict)


def SGD_custom_opt(var1=5., var2=15., g_dif=0., lr=0.0003, t_start=1, t_stop=6):
    print()
    print('#########custom SGD################')
    lr = torch.tensor([lr], dtype=torch.float64)

    var = torch.tensor([var1], dtype=torch.float64)
    g = torch.tensor([var2], dtype=torch.float64)
    g_dif = torch.tensor([g_dif], dtype=torch.float64)
    for t in range(t_start, t_stop):

        print('\nt: ' + str(t))

        var = var - lr*g

        print('var1: ' + str(var))

        if t == 1:
            g = g-g_dif


def plot_lambda_different_increases(var_initial=1.05, m_initial=0.0, u_initial=0.0,
                                    b1=0.9, b2=0.999, eps=1e-4, lr=0.0003, batch_size=256,
                                    start_zoom_in_episode=500,
                                    file_path_constraint_policy_loss_term=None,
                                    file_path_constraint_policy_loss_term_avg_per_game=None,
                                    file_path_lambda_avg_per_game=None,
                                    file_path_episode_length=None):
    ### plot lambda increase ###

    constraint_policy_loss_term = pd.read_csv(file_path_constraint_policy_loss_term)
    constraint_policy_loss_term_avg_per_game = pd.read_csv(file_path_constraint_policy_loss_term_avg_per_game)
    constr_lambda_avg_per_game = pd.read_csv(file_path_lambda_avg_per_game)
    episodes_length = pd.read_csv(file_path_episode_length)

    flagDrop = True
    remaining_batch_size = batch_size
    ind = 0
    while flagDrop:
        if remaining_batch_size >= episodes_length.iloc[ind]['Length']:
            remaining_batch_size -= episodes_length.iloc[ind]['Length']
            episodes_length = episodes_length.drop(axis=0, index=ind)
        else:
            episodes_length.iloc[ind]['Length'] = episodes_length.iloc[ind]['Length'] - remaining_batch_size
            flagDrop = False

    num_episodes = len(episodes_length.index)
    num_steps = episodes_length.sum().values[0]
    episodes = np.arange(1, num_episodes+2)
    steps = np.arange(1, num_steps+1)

    episodes_ratio = np.sqrt(1. - np.power(b2, episodes))/(1-np.power(b1, episodes))
    steps_ratio = np.sqrt(1. - np.power(b2, steps))/(1-np.power(b1, steps))
    steps_ratio_avg_per_episode = []
    adaptive_episode_size = lr * episodes_ratio
    adaptive_ratio_all_steps = []
    adaptive_ratio_avg_per_episode = []
    var_list_only_b1_b2 = [var_initial]
    var_list_b1_b2_length_avg_per_episode = [var_initial]
    var_list_b1_b2_length_all_steps = [[var_initial]]
    m_avg_per_episode = [m_initial]
    m_all_steps = [[m_initial]]
    u_avg_per_episode = [u_initial]
    u_all_steps = [[u_initial]]
    expected_lambda_avg_per_episode = [var_initial]
    expected_lambda_all_steps = [[var_initial]]
    appr_m_avg_per_episode = [m_initial]
    appr_m_all_steps = [[m_initial]]
    appr_u_avg_per_episode = [u_initial]
    appr_u_all_steps = [[u_initial]]
    appr_expected_lambda_avg_per_episode = [var_initial]
    appr_expected_lambda_all_steps = [[var_initial]]

    step_counter = 1
    for episode in episodes[:-1]:
        var_list_only_b1_b2.append(var_list_only_b1_b2[episode-1] + adaptive_episode_size[episode-1])
        var_list_b1_b2_length_all_steps.append([])

        m_all_steps.append([])
        u_all_steps.append([])
        expected_lambda_all_steps.append([])

        appr_m_all_steps.append([])
        appr_u_all_steps.append([])
        appr_expected_lambda_all_steps.append([])

        adaptive_ratio_all_steps.append([])

        episode_start_step = step_counter
        for step in range(step_counter, episodes_length.values[episode-1][0]+step_counter):

            if step == episode_start_step:
                previous_lamda_episode = episode-1
            else:
                previous_lamda_episode = episode

            var_list_b1_b2_length_all_steps[episode].\
                append(var_list_b1_b2_length_all_steps[previous_lamda_episode][-1] +
                       (lr * steps_ratio[step_counter-1]))

            m_all_steps[episode].append(b1 * m_all_steps[previous_lamda_episode][-1] +
                                        ((1. - b1) * constraint_policy_loss_term.values[step_counter-1][0]))
            u_all_steps[episode].append(b2 * u_all_steps[previous_lamda_episode][-1] +
                                        ((1. - b2) * (constraint_policy_loss_term.values[step_counter-1][0]**2)))
            expected_lambda_all_steps[episode].append(expected_lambda_all_steps[previous_lamda_episode][-1] +
                                                      ((lr * steps_ratio[step_counter-1]) *
                                                      (m_all_steps[episode][-1]/(np.sqrt(u_all_steps[episode][-1]) + eps))))

            adaptive_ratio_all_steps[episode-1].append(steps_ratio[step_counter-1] *
                                                       (m_all_steps[episode][-1]/(np.sqrt(u_all_steps[episode][-1]) + eps)))

            appr_m_all_steps[episode].append(b1 * appr_m_all_steps[previous_lamda_episode][-1] +
                                             (1. - b1) * constraint_policy_loss_term_avg_per_game.values[episode - 1][0])
            appr_u_all_steps[episode].append(b2 * appr_u_all_steps[previous_lamda_episode][-1] +
                                             ((1. - b2) * (constraint_policy_loss_term_avg_per_game.values[episode - 1][0] ** 2)))
            appr_expected_lambda_all_steps[episode].append(appr_expected_lambda_all_steps[previous_lamda_episode][-1] +
                                                           ((lr * steps_ratio[step_counter-1]) *
                                                            (appr_m_all_steps[episode][-1] /
                                                             (np.sqrt(appr_u_all_steps[episode][-1]) + eps))))

            step_counter += 1

        var_list_b1_b2_length_avg_per_episode.append(mean(var_list_b1_b2_length_all_steps[episode]))

        m_avg_per_episode.append(mean(m_all_steps[episode]))
        u_avg_per_episode.append(mean(u_all_steps[episode]))
        expected_lambda_avg_per_episode.append((mean(expected_lambda_all_steps[episode])))

        adaptive_ratio_avg_per_episode.append(mean(adaptive_ratio_all_steps[episode-1]))
        steps_ratio_avg_per_episode.append(mean(steps_ratio[episode_start_step-1:step_counter-1]))

        appr_m_avg_per_episode.append(mean(appr_m_all_steps[episode]))
        appr_u_avg_per_episode.append(mean(appr_u_all_steps[episode]))
        appr_expected_lambda_avg_per_episode.append((mean(appr_expected_lambda_all_steps[episode])))

        assert step_counter == step+1

    assert step_counter == num_steps+1, 'step_counter: ' + str(step_counter) + ' , num_steps: ' + str(num_steps)

    min_max_scaler = MinMaxScaler((min(var_list_only_b1_b2), max(var_list_only_b1_b2)))
    var_list_b1_b2_length_avg_per_episode_scaled = min_max_scaler.\
        fit_transform(np.expand_dims(np.array(var_list_b1_b2_length_avg_per_episode), axis=1))
    m_avg_per_episode_scaled = min_max_scaler.fit_transform(np.expand_dims(np.array(m_avg_per_episode), axis=1))
    u_avg_per_episode_scaled = min_max_scaler.fit_transform(np.expand_dims(np.array(u_avg_per_episode), axis=1))
    u_sqrt_avg_per_episode_scaled = min_max_scaler.\
        fit_transform(np.expand_dims(np.sqrt(np.array(u_avg_per_episode)), axis=1))
    ratio_m_u_sqrt_e_avg_per_episode_scaled = min_max_scaler.\
        fit_transform(np.expand_dims(np.array(m_avg_per_episode)/(np.sqrt(np.array(u_avg_per_episode)) + eps), axis=1))
    expected_lambda_avg_per_episode_scaled = min_max_scaler.\
        fit_transform(np.expand_dims(np.array(expected_lambda_avg_per_episode), axis=1))
    appr_expected_lambda_avg_per_episode_scaled = min_max_scaler.\
        fit_transform(np.expand_dims(np.array(appr_expected_lambda_avg_per_episode), axis=1))
    episodes_length_scaled = min_max_scaler.fit_transform(episodes_length.values)
    episodes_length_scaled = np.append(np.array([min(var_list_only_b1_b2)]), episodes_length_scaled.squeeze(axis=1))
    constraint_policy_loss_term_avg_per_game_scaled = min_max_scaler.\
        fit_transform(constraint_policy_loss_term_avg_per_game.values)
    constraint_policy_loss_term_avg_per_game_scaled = \
        np.append(np.array([min(var_list_only_b1_b2)]), constraint_policy_loss_term_avg_per_game_scaled.squeeze(axis=1))
    constr_lambda_scaled = min_max_scaler.fit_transform(constr_lambda_avg_per_game.values)
    constr_lambda_scaled = np.append(np.array([min(var_list_only_b1_b2)]), constr_lambda_scaled.squeeze(axis=1))

    ######################################
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    plt.xlabel('Episodes')
    plt.ylabel('Scaled Values')
    plt.plot(episodes, episodes_length_scaled, zorder=1)
    plt.plot(episodes, var_list_only_b1_b2, zorder=2)
    plt.plot(episodes, var_list_b1_b2_length_avg_per_episode_scaled, zorder=2)
    plt.plot(episodes, constraint_policy_loss_term_avg_per_game_scaled, zorder=1)
    plt.plot(episodes, constr_lambda_scaled, zorder=1)
    plt.plot(episodes, m_avg_per_episode_scaled, zorder=1)
    plt.plot(episodes, u_avg_per_episode_scaled, zorder=1)
    plt.plot(episodes, u_sqrt_avg_per_episode_scaled, zorder=1)
    plt.plot(episodes, ratio_m_u_sqrt_e_avg_per_episode_scaled, zorder=1)
    plt.plot(episodes, expected_lambda_avg_per_episode_scaled, zorder=1)
    plt.plot(episodes, appr_expected_lambda_avg_per_episode_scaled, zorder=1)
    plt.gca().legend(('scaled length', 'b1 and b2 only', 'b1 b2 length',
                      'scaled constraint policy loss term', 'scaled lambda', 'scaled m', 'scaled u',
                      'scaled sqrt u', 'scaled m / (sqrt(u)+e)', 'scaled expected lambda',
                      'scaled approximated expected lambda'),
                     bbox_to_anchor=(0.5, -0.06), loc=9)
    fig.subplots_adjust(bottom=0.25)
    plt.show()
    ######################################

    ######################################
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    plt.plot(episodes, var_list_only_b1_b2, zorder=2)
    plt.plot(episodes, var_list_b1_b2_length_avg_per_episode, zorder=2)
    plt.plot(episodes, np.append(np.array([var_initial]), constr_lambda_avg_per_game.values.squeeze(axis=1)), zorder=1)
    plt.plot(episodes, expected_lambda_avg_per_episode, zorder=1)
    plt.plot(episodes, appr_expected_lambda_avg_per_episode, zorder=1)
    plt.gca().legend(('b1 and b2 only', 'b1 b2 length', 'lambda', 'expected lambda', 'approximated expected lambda'), loc='upper left')
    plt.show()
    ######################################

    ######################################
    fig1 = plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    ax1 = fig1.add_subplot(111)
    ax1.plot(episodes, np.append(np.array([constraint_policy_loss_term_avg_per_game.values[0, 0]]),
                                 constraint_policy_loss_term_avg_per_game.values.squeeze(axis=1)))
    ax1.plot(episodes, m_avg_per_episode)
    ax1.plot(episodes, u_avg_per_episode)
    ax1.plot(episodes, np.sqrt(np.array(u_avg_per_episode)))

    # Zoom in
    ax2 = plt.axes([.3, .225, .59, .59])
    ax2.plot(episodes[start_zoom_in_episode:],
             np.append(np.array([constraint_policy_loss_term_avg_per_game.values[0, 0]]),
                       constraint_policy_loss_term_avg_per_game.values.squeeze(axis=1))[start_zoom_in_episode:])
    ax2.plot(episodes[start_zoom_in_episode:], m_avg_per_episode[start_zoom_in_episode:])
    ax2.plot(episodes[start_zoom_in_episode:], u_avg_per_episode[start_zoom_in_episode:])
    ax2.plot(episodes[start_zoom_in_episode:], np.sqrt(np.array(u_avg_per_episode))[start_zoom_in_episode:])

    # Dashed lines for zoom
    ax2_pos1 = [ax2.get_position().x0, ax2.get_position().y0]
    ax2_pos2 = [ax2.get_position().x1, ax2.get_position().y1]
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    ax1.plot([start_zoom_in_episode+1, ax2_pos1[1] * episodes[-1]],
             [0.0, y_lim[0] + (ax2_pos1[0] * (y_lim[1] - y_lim[0]))
              - (0.13 * (y_lim[1] - y_lim[0]))], '--', color='black', linewidth=1.)
    ax1.plot([episodes[-1], x_lim[0] + (ax2_pos2[1] * (x_lim[1] - x_lim[0])) + (0.175 * (x_lim[1] - x_lim[0]))],
             [0.0, y_lim[0] + (ax2_pos1[1] * (y_lim[1] - y_lim[0])) - (0.08 * (y_lim[1] - y_lim[0]))],
             '--', color='black', linewidth=1.)

    # Plot the zoom points
    ax1.plot(start_zoom_in_episode+1, 0.0, 'o', color='black')
    ax1.plot(episodes[-1], 0.0, 'o', color='black')

    # Get current tick locations and append specific ticks
    x_ticks = np.append(ax1.get_xticks(), [start_zoom_in_episode, episodes[-1]-1])
    # Set xtick locations to the values of the array `x_ticks`
    ax1.set_xticks(x_ticks)

    # Adjust limits
    ax1.set_ylim(y_lim)
    ax1.set_xlim(x_lim)

    plt.gca().legend(('constraint policy loss term', 'm', 'u', 'sqrt u'), loc='upper left')
    plt.show()
    ######################################

    ######################################
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    plt.plot(episodes, np.array(m_avg_per_episode)/(np.sqrt(np.array(u_avg_per_episode)) + eps), zorder=1)
    plt.gca().legend(('m / (sqrt(u)+e)',), loc='upper left')
    plt.show()
    ######################################

    ######################################
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    ax1_lines = ax1.plot(episodes, u_avg_per_episode)

    # Zoom in
    ax2 = plt.axes([.3, .225, .59, .59])
    ax2.plot(episodes[start_zoom_in_episode:], u_avg_per_episode[start_zoom_in_episode:])

    # Dashed lines for zoom
    ax2_pos1 = [ax2.get_position().x0, ax2.get_position().y0]
    ax2_pos2 = [ax2.get_position().x1, ax2.get_position().y1]
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    ax1.plot([start_zoom_in_episode+1, ax2_pos1[1] * episodes[-1]],
             [0.0, y_lim[0] + (ax2_pos1[0] * (y_lim[1] - y_lim[0]))
              - (0.13 * (y_lim[1] - y_lim[0]))], '--', color='black', linewidth=1.)
    ax1.plot([episodes[-1], x_lim[0] + (ax2_pos2[1] * (x_lim[1] - x_lim[0])) + (0.175 * (x_lim[1] - x_lim[0]))],
             [0.0, y_lim[0] + (ax2_pos1[1] * (y_lim[1] - y_lim[0])) - (0.08 * (y_lim[1] - y_lim[0]))],
             '--', color='black', linewidth=1.)

    # Plot the zoom points
    ax1.plot(start_zoom_in_episode+1, 0.0, 'o', color='black')
    ax1.plot(episodes[-1], 0.0, 'o', color='black')

    # Get current tick locations and append specific ticks
    x_ticks = np.append(ax1.get_xticks(), [start_zoom_in_episode, episodes[-1]-1])
    # Set xtick locations to the values of the array `x_ticks`
    ax1.set_xticks(x_ticks)

    # Adjust limits
    ax1.set_ylim(y_lim)
    ax1.set_xlim(x_lim)

    plt.gca().legend(('u',), loc='upper left')
    plt.show()
    ######################################

    ######################################
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    plt.plot(episodes[:-1], adaptive_ratio_avg_per_episode)
    plt.gca().legend(('adaptive ratio',), loc='upper left')
    plt.show()
    ######################################

    ######################################
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.xlabel('Episodes')
    plt.ylabel('Original Values')
    ax1_lines = ax1.plot(episodes[:-1], steps_ratio_avg_per_episode)
    plt.gca().legend(('b1 b2 ratio',), loc='lower right')

    # Zoom in
    ax2 = plt.axes([.24, .23, .25, .59])
    ax2.plot(episodes[:20], steps_ratio_avg_per_episode[:20])
    ax3 = plt.axes([.60, .23, .25, .59])
    ax3.plot(episodes[20:-1], steps_ratio_avg_per_episode[20:])

    # Get current tick locations and append specific ticks
    x_ticks = ax3.get_xticks()
    x_ticks = np.array([20] + x_ticks[2:].tolist())
    # Set xtick locations to the values of the array `x_ticks`
    ax3.set_xticks(x_ticks)

    plt.show()
    ######################################


if __name__ == '__main__':

    # Lambda
    var1_ = 5.

    # Gradients
    var2_ = 15.

    # Reduce gradients only in one step
    g_dif_ = 0.

    # For Adam and SGD
    lr_ = 0.0003

    # For Adam only
    eps_ = 1e-4

    # For Pytorch opts only
    n_steps_ = 5

    # For custom Adam only
    m_prev_ = 0.
    u_prev_ = 0.
    b1_ = 0.9
    b2_ = 0.999

    # For custom opts only
    t_start_ = 1
    t_stop_ = 6

    # Only for plot
    batch_size_ = 256
    start_zoom_in_episode_ = 500
    file_path_constraint_policy_loss_term_ = "/home/georgepap/PycharmProjects/HAI-MAZE_master/HAI-MAZE/results/tmp/sac_single_agent_exp_test1_2/X_Y_constraint_policy_loss_term_per_step.csv"
    file_path_constraint_policy_loss_term_avg_per_game_ = "/home/georgepap/PycharmProjects/HAI-MAZE_master/HAI-MAZE/results/tmp/sac_single_agent_exp_test1_2/X_Y_constraint_policy_loss_term_avg_per_game.csv"
    file_path_lambda_avg_per_game_ = "/home/georgepap/PycharmProjects/HAI-MAZE_master/HAI-MAZE/results/tmp/sac_single_agent_exp_test1_2/X_Y_constraint_lambda_avg_per_game.csv"
    file_path_episode_length_ = "/home/georgepap/PycharmProjects/HAI-MAZE_master/HAI-MAZE/results/tmp/sac_single_agent_exp_test1_2/length_list.csv"

    Adam_pytorch_opt(var1=var1_, var2=var2_, g_dif=g_dif_, lr=lr_, eps=eps_, n_steps=n_steps_)
    Adam_custom_opt(var1=var1_, var2=var2_, g_dif=g_dif_, lr=lr_, eps=eps_,
                    m_prev=m_prev_, u_prev=u_prev_, b1=b1_, b2=b2_, t_start=t_start_, t_stop=t_stop_)

    SGD_pytorch_opt(var1=var1_, var2=var2_, g_dif=g_dif_, lr=lr_, n_steps=n_steps_)
    SGD_custom_opt(var1=var1_, var2=var2_, g_dif=g_dif_, lr=lr_, t_start=t_start_, t_stop=t_stop_)

    plot_lambda_different_increases(var_initial=var1_, m_initial=m_prev_, u_initial=u_prev_,
                                    b1=b1_, b2=b2_, eps=eps_, lr=lr_, batch_size=batch_size_,
                                    start_zoom_in_episode=start_zoom_in_episode_,
                                    file_path_constraint_policy_loss_term=file_path_constraint_policy_loss_term_,
                                    file_path_constraint_policy_loss_term_avg_per_game=file_path_constraint_policy_loss_term_avg_per_game_,
                                    file_path_lambda_avg_per_game=file_path_lambda_avg_per_game_,
                                    file_path_episode_length=file_path_episode_length_)
