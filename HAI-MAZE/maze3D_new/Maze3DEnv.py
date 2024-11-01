import os
import numpy
import time

# Reward functions
from game import rewards
# Virtual environment
from maze3D_new.gameObjects import *
from maze3D_new.utils import checkTerminal, convert_actions, goals
from maze3D_new.layouts import *

# RL modules
from plot_utils.plot_utils import get_config

# the game layouts
layouts = \
    [(layout_10_up_right, "layout_10_up_right"),
     (layout_5_first_row_up, "layout_5_first_row_up"),
     (layout_6_first_row_up, "layout_6_first_row_up"),
     (layout_7_first_row_up, "layout_7_first_row_up"),
     (layout_8_first_row_up, "layout_8_first_row_up"),
     (layout_9_first_row_up, "layout_9_first_row_up"),
     (layout_6_second_row_up, "layout_6_second_row_up"),
     (layout_7_second_row_up, "layout_7_second_row_up"),
     (layout_8_second_row_up, "layout_8_second_row_up"),
     (layout_9_second_row_up, "layout_9_second_row_up"),
     (layout_10_second_row_up, "layout_10_second_row_up"),
     (layout_7_third_row_up, "layout_7_third_row_up"),
     (layout_8_third_row_up, "layout_8_third_row_up"),
     (layout_9_third_row_up, "layout_9_third_row_up"),
     (layout_10_third_row_up, "layout_10_third_row_up"),
     (layout_7_forth_row_up, "layout_7_forth_row_up"),
     (layout_8_forth_row_up, "layout_8_forth_row_up"),
     (layout_9_forth_row_up, "layout_9_forth_row_up"),
     (layout_10_forth_row_up, "layout_10_forth_row_up"),
     (layout_4_forth_row_up, "layout_4_forth_row_up"),
     (layout_3_fifth_row_bottom, "layout_3_fifth_row_bottom"),
     (layout_4_fifth_row_bottom, "layout_4_fifth_row_bottom"),
     (layout_5_fifth_row_bottom, "layout_5_fifth_row_bottom"),
     (layout_6_fifth_row_bottom, "layout_6_fifth_row_bottom"),
     (layout_9_fifth_row_up, "layout_9_fifth_row_up"),
     (layout_10_fifth_row_up, "layout_10_fifth_row_up"),
     (layout_3_sixth_row_bottom, "layout_3_sixth_row_bottom"),
     (layout_4_sixth_row_bottom, "layout_4_sixth_row_bottom"),
     (layout_5_sixth_row_bottom, "layout_5_sixth_row_bottom"),
     (layout_6_sixth_row_bottom, "layout_6_sixth_row_bottom"),
     (layout_10_sixth_row_up, "layout_10_sixth_row_up"),
     (layout_4_seventh_row_bottom, "layout_4_seventh_row_bottom"),
     (layout_5_seventh_row_bottom, "layout_5_seventh_row_bottom"),
     (layout_6_seventh_row_bottom, "layout_6_seventh_row_bottom"),
     (layout_5_eighth_row_bottom, "layout_5_eighth_row_bottom"),
     (layout_6_eighth_row_bottom, "layout_6_eighth_row_bottom"),
     (layout_7_eighth_row_bottom, "layout_7_eighth_row_bottom")]

class ActionSpace:
    def __init__(self):

        # Possible actions for each axis:
        # 0) No move
        # 1) Up/Left
        # 2) Down/Right
        self.actions = list(range(0, 3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    """The environment wrapper for the Maze3D game"""

    def __init__(
            self,
            config=None,
            config_file=None,
            seed=None,
            initialize_seed=False,
            replay_game=False,
            replay_game_layout_index=None
    ):

        assert not (initialize_seed and replay_game)
        assert not (replay_game and not isinstance(seed, int))

        # get the configuration dictionary
        self.config = get_config(config_file) if config_file is not None else config

        # Initialize randomness based on which layouts are selected
        if initialize_seed:
            if isinstance(seed, int) or seed is None:
                np.random.seed(seed)
            elif seed == 'Seq':
                self.layout_index = 0
            else:
                raise ValueError

        # choose one starting point for the ball
        if seed != 'Seq':
            # randomly
            self.layout_index = np.random.choice(range(len(layouts))) if not replay_game else replay_game_layout_index
        elif seed == 'Seq' and not initialize_seed:
            # sequentially
            self.layout_index += 1
            assert self.layout_index < len(layouts)

        current_layout = layouts[self.layout_index][0]
        current_layout_name = layouts[self.layout_index][1]

        # specify to visualize or not
        self.render = self.config['Experiment']['render']
        if self.render is True:
            print("\nThe current layout is: '{}'".format(current_layout_name))

        # specify to freeze the screen or not
        self.freeze_motion = False if 'freeze_motion' not in self.config['Experiment'].keys() \
                                   else self.config['Experiment']['freeze_motion']

        # specify constraint parameters
        self.constr_ball_only_at_the_right_side_wrt_hole = \
            self.config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole']
        self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef = None
        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
            self.value_satisfied_constr_ball_only_at_the_right_side_wrt_hole = \
                self.config['Experiment']['satisfied_constraint_value']
            self.value_not_satisfied_constr_ball_only_at_the_right_side_wrt_hole = \
                self.config['Experiment']['not_satisfied_constraint_value']
            self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef = \
                self.config['Experiment']['constraint_ball_only_at_the_right_side_wrt_hole_x_coef']
        self.constr_ball_only_at_the_up_side_wrt_hole = \
            self.config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole']
        self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef = None
        if self.constr_ball_only_at_the_up_side_wrt_hole is True:
            self.value_satisfied_constr_ball_only_at_the_up_side_wrt_hole = \
                self.config['Experiment']['satisfied_constraint_value']
            self.value_not_satisfied_constr_ball_only_at_the_up_side_wrt_hole = \
                self.config['Experiment']['not_satisfied_constraint_value']
            self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef = \
                self.config['Experiment']['constraint_ball_only_at_the_up_side_wrt_hole_y_coef']
        self.constraint_ball_not_in_circle = \
            self.config['Experiment']['constraint_ball_not_in_circle']
        self.constr_ball_not_in_circle_circle_position = None
        self.constr_ball_not_in_circle_circle_radius = None
        if self.constraint_ball_not_in_circle is True:
            self.value_satisfied_constr_ball_not_in_circle = \
                self.config['Experiment']['satisfied_constraint_value']
            self.value_not_satisfied_constr_ball_not_in_circle = \
                self.config['Experiment']['not_satisfied_constraint_value']
            self.constr_ball_not_in_circle_circle_position = \
                self.config['Experiment']['constraint_ball_not_in_circle_circle_position']
            self.constr_ball_not_in_circle_circle_radius = \
                self.config['Experiment']['constraint_ball_not_in_circle_circle_radius']

        self.goal = self.config['game']['goal']
        # create the game board
        self.board = GameBoard(current_layout,
                               render=self.render,
                               constr_ball_only_at_the_right_side_wrt_hole=self.constr_ball_only_at_the_right_side_wrt_hole,
                               constraint_ball_only_at_the_right_side_wrt_hole_x_coef=self.constraint_ball_only_at_the_right_side_wrt_hole_x_coef,
                               constr_ball_only_at_the_up_side_wrt_hole=self.constr_ball_only_at_the_up_side_wrt_hole,
                               constraint_ball_only_at_the_up_side_wrt_hole_y_coef=self.constraint_ball_only_at_the_up_side_wrt_hole_y_coef,
                               constr_ball_not_in_circle=self.constraint_ball_not_in_circle,
                               constr_ball_not_in_circle_circle_position=self.constr_ball_not_in_circle_circle_position,
                               constr_ball_not_in_circle_circle_radius=self.constr_ball_not_in_circle_circle_radius,
                               scaling_x=self.config['game']['scaling_x'],
                               scaling_y=self.config['game']['scaling_y'],
                               goal=self.goal)
        # boolean that check if game has finished
        self.done = False
        # get the initial state of the board
        self.observation = self.get_state()  # must init board first
        # get the action space
        self.action_space = ActionSpace()
        # get the shape of the observation space
        self.observation_shape = (len(self.observation),)
        # the fps to run the game in
        self.fps = 60
        # retrieve the reward
        rewards.main(self.config)

        if self.render:
            from maze3D_new.config import clock
            from maze3D_new.assets import glClearDepth, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glClear
            import pygame as pg

            self.clock = clock
            self.glClearDepth = glClearDepth
            self.GL_COLOR_BUFFER_BIT = GL_COLOR_BUFFER_BIT
            self.GL_DEPTH_BUFFER_BIT = GL_DEPTH_BUFFER_BIT
            self.glClear = glClear
            self.pg = pg

            # create the key dictionary
            self.keys = {pg.K_w: 1, pg.K_s: 2, pg.K_LEFT: 4, pg.K_RIGHT: 8}
            # create conversion key dictionary
            self.conversion_keys = {pg.K_w: 0, pg.K_s: 1, pg.K_LEFT: 2, pg.K_RIGHT: 3}

    def step(self, actions_agents, timed_out, goal, action_iterations):
        """
        Performs the action of the agent to the environment for action_duration time.
        Simultaneously, receives input from the user via the keyboard arrows
        :param actions_agents: list with the actions of the agents. make sure it is compatible.
                                if None, human is used both axes.
                                if actions_agents[0] is None, human action is used for X-axis.
                                if actions_agents[1] is None, human action is used for Y-axis
        :param timed_out: bool variable. true if game has been timed out
        :param goal: the goal of the game
        :param action_iterations: the number of environment updates during executing an agent's action on the game
        :return: a transition [observation, reward, done, train_fps, duration_pause, action_list]
        """

        assert goal == self.goal

        duration_pause, current_duration_pause, extra_time = 0, 0, 0
        actions = [0, 0, 0, 0]
        action_list = [] # to store all the agent-human action pairs performed to the game.

        iteration = 1

        # perform agent's action for 'action_iterations' environment updates
        while iteration <= action_iterations and not self.done:

            if self.freeze_motion:
                iters = 0
                while len(action_list) == 0:
                    if len(action_list) == 0:
                        # get keyboard action from user
                        current_duration_pause, _, human_actions = \
                            self.getKeyboard([None] * len(actions), sleep_mode=True)
                        duration_pause += current_duration_pause
                        if ((np.array(human_actions) != 0) & (np.array(human_actions) != None)).any() or (np.array(human_actions) == 0).all():
                            if ((np.array(human_actions) != 0) & (np.array(human_actions) != None)).any():
                                human_actions = [0 if elem is None else elem for elem in human_actions]
                            action_list.append(human_actions)
                            action = human_actions
                    iters += 1
            else:
                human_actions = [None, None]

                if len(action_list) == 0:
                    if self.render:
                        # get keyboard action from user
                        current_duration_pause, _, human_actions = self.getKeyboard(actions)
                    duration_pause += current_duration_pause
                    if actions_agents[0] is not None and actions_agents[1] is not None:
                        action = actions_agents
                    elif actions_agents[0] is None and actions_agents[1] is not None:
                        action = [human_actions[0], actions_agents[1]]
                    elif actions_agents[0] is not None and actions_agents[1] is None:
                        action = [actions_agents[0], human_actions[1]]
                    else:
                        action = human_actions
                    action_list.append(action)

            self.board.handleKeys(action)  # apply action to the environment
            self.board.update()  # update board's rotations

            fps = 0
            if self.render:
                self.glClearDepth(1000.0)
                self.glClear(self.GL_COLOR_BUFFER_BIT | self.GL_DEPTH_BUFFER_BIT)
                self.board.draw()  # render new graphics of the game
                self.pg.display.flip()

                self.clock.tick(self.fps)  # set the fps tick
                fps = self.clock.get_fps()  # get the actual fps performed
                self.pg.display.set_caption("Running at " + str(int(fps)) + " fps")

            # Get next state
            self.observation = self.get_state()
            if checkTerminal(self.board.ball, goal):
                self.done = True
                if self.render:
                    extra_time = self.display_terminating_screen()
            elif timed_out:
                if self.render:
                    extra_time = self.display_timed_out_screen()
                self.done = True
            duration_pause += extra_time
            actions = [0, 0, 0, 0]

            iteration += 1

        reward = rewards.reward_function_maze(self.done, timed_out, ball=self.board.ball, goal=goal)

        self.check_constraints_and_augment_observation()

        return self.observation, reward, self.done, fps, duration_pause, action_list

    def getKeyboard(self, actions, sleep_mode=False):
        """
        Retrieves human's action from keyboard arrows.
        -left/right
        -space: pause (press again to unpause)
        - w/s if applicable (in 2 humans set up or 1 human moving both axes)
        :param actions: an action vector used to convert actions
        :param sleep_mode: specify whether to wait for W key to be pressed to assign zero action or not.
        :return: duration_pause, action vector, human action
        """
        duration_pause = 0
        space_pressed = True
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return 1
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_SPACE and space_pressed:
                    space_pressed = False
                    start_pause = time.time()
                    pause()
                    end_pause = time.time()
                    duration_pause += end_pause - start_pause
                if event.key == self.pg.K_q:
                    exit(1)
                if event.key in self.keys:
                    actions[self.conversion_keys[event.key]] = 1
                if sleep_mode:
                    if event.key == self.pg.K_RETURN:
                        actions = [0] * len(actions)
            if event.type == self.pg.KEYUP:
                if event.key in self.keys:
                    actions[self.conversion_keys[event.key]] = 0
        human_actions = convert_actions(actions)
        return duration_pause, actions, human_actions

    def get_state(self):
        """
        ball pos x | ball pos y | ball vel x | ball vel y|  theta(x) | phi(y) |  theta_dot(x) | phi_dot(y)
        :return: the current state of the board
        """
        return np.asarray(
            [self.board.ball.x, self.board.ball.y, self.board.ball.velocity[0], self.board.ball.velocity[1],
             self.board.rot_x, self.board.rot_y, self.board.velocity[0], self.board.velocity[1]]
                         )

    def reset(self, seed=None, initialize_seed=False, replay_game=False):
        """
        Resets the game.
        param seed: Integer number to feed numpy random
        param initialize_seed: Boolean specifying to reseed numpy random or not.
        :return: the initial observation of the game, the set-up duration
        """

        self.__init__(
            config=self.config,
            seed=seed,
            initialize_seed=initialize_seed,
            replay_game=replay_game,
            replay_game_layout_index=None if not replay_game else self.layout_index
        )

        setting_up_duration = 0
        if self.render:
            setting_up_duration = self.display_starting_screen()

        self.check_constraints_and_augment_observation()

        return self.observation, setting_up_duration

    def check_constraints_and_augment_observation(self):

        violation_print_text = "\n"

        if self.constr_ball_only_at_the_right_side_wrt_hole is True:
            assert self.board.vertical_red_line.x == self.board.vertical_green_line.x, \
                "x coordinates of vertical red line and vertical green line do not match!"
            value_constr_ball_only_at_the_right_side_wrt_hole = \
                self.value_not_satisfied_constr_ball_only_at_the_right_side_wrt_hole \
                if self.board.ball.x <= self.board.vertical_red_line.x \
                else \
                self.value_satisfied_constr_ball_only_at_the_right_side_wrt_hole
            self.observation = np.concatenate([self.observation,
                                               np.array([value_constr_ball_only_at_the_right_side_wrt_hole])],
                                              axis=0)
            if (self.render is True and
                (value_constr_ball_only_at_the_right_side_wrt_hole ==
                 self.value_not_satisfied_constr_ball_only_at_the_right_side_wrt_hole)):
                violation_print_text += "Violation of vertical line constraint!\n"

        if self.constr_ball_only_at_the_up_side_wrt_hole is True:
            assert self.board.horizontal_red_line.x == self.board.horizontal_green_line.x, \
                "x coordinates of horizontal green line and horizontal red line do not match!"
            value_constr_ball_only_at_the_up_side_wrt_hole = \
                self.value_not_satisfied_constr_ball_only_at_the_up_side_wrt_hole \
                if self.board.ball.y <= self.board.horizontal_red_line.y \
                else \
                self.value_satisfied_constr_ball_only_at_the_up_side_wrt_hole
            self.observation = np.concatenate([self.observation,
                                               np.array([value_constr_ball_only_at_the_up_side_wrt_hole])],
                                              axis=0)
            if (self.render is True and
                (value_constr_ball_only_at_the_up_side_wrt_hole ==
                 self.value_not_satisfied_constr_ball_only_at_the_up_side_wrt_hole)):
                violation_print_text += "Violation of horizontal line constraint!\n"

        if self.constraint_ball_not_in_circle is True:
            assert len(self.board.green_torus) == len(self.board.red_torus), "The number of green torus is other than red torus!"
            value_constr_ball_not_in_circle = None
            for torus_idx in range(len(self.board.green_torus)):
                assert self.board.green_torus[torus_idx].x == self.board.red_torus[torus_idx].x, \
                    f"x coordinates of red and green torus {torus_idx} do not match!"
                assert self.board.green_torus[torus_idx].y == self.board.red_torus[torus_idx].y, \
                    f"y coordinates of red and green torus {torus_idx} do not match!"
                value_constr_ball_not_in_circle = \
                    self.value_not_satisfied_constr_ball_not_in_circle \
                    if (self.board.red_torus[torus_idx].x - self.board.red_torus[torus_idx].radius_outer_circle <= self.board.ball.x <=
                        self.board.red_torus[torus_idx].x + self.board.red_torus[torus_idx].radius_outer_circle) and \
                       (self.board.red_torus[torus_idx].y - self.board.red_torus[torus_idx].radius_outer_circle <= self.board.ball.y <=
                        self.board.red_torus[torus_idx].y + self.board.red_torus[torus_idx].radius_outer_circle) \
                    else \
                    self.value_satisfied_constr_ball_not_in_circle
                if value_constr_ball_not_in_circle == self.value_not_satisfied_constr_ball_not_in_circle:
                    if self.render is True:
                        violation_print_text += f"Violation of circle constraint {torus_idx}!\n"
                    # Only one constraint can be violated at each timestep. Thus, when one found, we can break the loop safely.
                    break
            self.observation = np.concatenate([self.observation,
                                               np.array([value_constr_ball_not_in_circle])],
                                              axis=0)
            if violation_print_text != "\n":
                print(violation_print_text)

    def display_terminating_screen(self):
        """
        Displays a message to the user when the goal has been reached
        :return: GUI display_duration
        """

        display_duration = self.config['GUI']['goal_screen_display_duration']
        timeStart = time.time()
        i = 0
        while time.time() - timeStart <= display_duration:
            self.glClearDepth(1000.0)
            self.glClear(self.GL_COLOR_BUFFER_BIT | self.GL_DEPTH_BUFFER_BIT)
            self.board.draw(mode=2, idx=i)  # mode: 2 for reaching goal
            self.pg.display.flip()
            time.sleep(1)
            i += 1
        self.done = True
        return display_duration

    def display_timed_out_screen(self):
        """
        Displays a timeout message to the user
        :return: GUI display_duration
        """

        display_duration = self.config['GUI']['timeout_screen_display_duration']
        timeStart = time.time()
        i = 0
        while time.time() - timeStart <= display_duration:
            self.glClearDepth(1000.0)
            self.glClear(self.GL_COLOR_BUFFER_BIT | self.GL_DEPTH_BUFFER_BIT)
            self.board.draw(mode=3, idx=i)  # mode: 3 for time out
            self.pg.display.flip()
            time.sleep(1)
            i += 1
        self.done = True
        return display_duration

    def display_starting_screen(self):
        """
        Displays a starting countdown message to the user before the game starts
        :return: GUI display_duration
        """

        display_duration = self.config['GUI']['start_up_screen_display_duration']
        self.board.update()

        # Count down
        for i in range(display_duration, -1, -1):
            self.glClearDepth(1000.0)
            self.glClear(self.GL_COLOR_BUFFER_BIT | self.GL_DEPTH_BUFFER_BIT)
            self.board.draw(mode=1, idx=i)
            self.pg.display.flip()
            time.sleep(1)
        return display_duration
