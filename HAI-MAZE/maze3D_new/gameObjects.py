from numpy.linalg import norm
import math
import numpy as np
from scipy.spatial import distance
import time
import pyrr
from maze3D_new.utils import goals

ball_diameter = 43.615993
damping_factor = 0.3
discrete_steps_from_center = 5
box_size = 43.615993

class GameBoard:
    def __init__(self, layout, render=True,
                 constr_ball_only_at_the_right_side_wrt_hole=False,
                 constraint_ball_only_at_the_right_side_wrt_hole_x_coef=2.,
                 constr_ball_only_at_the_up_side_wrt_hole=False,
                 constraint_ball_only_at_the_up_side_wrt_hole_y_coef=2.,
                 constr_ball_not_in_circle=False,
                 constr_ball_not_in_circle_circle_position=[(0.0, 0.0)],
                 constr_ball_not_in_circle_circle_radius=[1.0],
                 scaling_x='auto', scaling_y='auto', goal=None):

        self.box_size = box_size
        self.velocity = [0, 0]
        self.walls = []
        self.layout = layout
        self.render = render
        self.max_x_rotation = 0.5
        self.max_y_rotation = 0.5
        self.scaling_x = (self.max_x_rotation / discrete_steps_from_center) if scaling_x == 'auto' else scaling_x
        self.scaling_y = self.scaling_x if scaling_y == 'auto' else scaling_y
        self.constr_ball_only_at_the_right_side_wrt_hole = constr_ball_only_at_the_right_side_wrt_hole
        self.constr_ball_only_at_the_up_side_wrt_hole = constr_ball_only_at_the_up_side_wrt_hole
        self.constr_ball_not_in_circle = constr_ball_not_in_circle
        self.goal = goal

        self.num_of_boxes_x = len(layout)
        self.num_of_boxes_y = len(layout[0])

        for row in range(self.num_of_boxes_x):
            self.walls.append([])
            for col in range(self.num_of_boxes_y):
                self.walls[row].append(None)
                if layout[row][col] != 0:
                    if layout[row][col] == 2:
                        self.hole = Hole(self.box_size * col - self.num_of_boxes_y * self.box_size / 2,
                                         self.box_size * row - self.num_of_boxes_x * self.box_size / 2, self)
                        # hole_col = col
                        # hole_row = row
                    elif layout[row][col] == 3:
                        self.ball = Ball(self.box_size * col - self.num_of_boxes_y * self.box_size / 2,
                                         self.box_size * row - self.num_of_boxes_x * self.box_size / 2, self)
                    else:
                        self.walls[row][col] = Wall(self.box_size * col - self.num_of_boxes_y * self.box_size / 2,
                                                    self.box_size * row - self.num_of_boxes_x * self.box_size / 2,
                                                    layout[row][col], self)

        if self.constr_ball_only_at_the_right_side_wrt_hole:
            self.vertical_green_line = Line(self.box_size * constraint_ball_only_at_the_right_side_wrt_hole_x_coef - self.num_of_boxes_y * self.box_size / 2,
                                            self.box_size * self.num_of_boxes_x / 3 - self.num_of_boxes_x * self.box_size / 2,
                                            self,
                                            'green',
                                            'Y')
            self.vertical_red_line = Line(self.box_size * constraint_ball_only_at_the_right_side_wrt_hole_x_coef - self.num_of_boxes_y * self.box_size / 2,
                                          self.box_size * self.num_of_boxes_x / 3 - self.num_of_boxes_x * self.box_size / 2,
                                          self,
                                          'red',
                                          'Y')
        if self.constr_ball_only_at_the_up_side_wrt_hole:
            self.horizontal_green_line = Line(self.box_size * self.num_of_boxes_y / 3 - self.num_of_boxes_y * self.box_size / 2,
                                              self.box_size * constraint_ball_only_at_the_up_side_wrt_hole_y_coef - self.num_of_boxes_x * self.box_size / 2,
                                              self,
                                              'green',
                                              'X')
            self.horizontal_red_line = Line(self.box_size * self.num_of_boxes_y / 3 - self.num_of_boxes_y * self.box_size / 2,
                                            self.box_size * constraint_ball_only_at_the_up_side_wrt_hole_y_coef - self.num_of_boxes_x * self.box_size / 2,
                                            self,
                                            'red',
                                            'X')
        if self.constr_ball_not_in_circle:
            assert len(constr_ball_not_in_circle_circle_position) == len(constr_ball_not_in_circle_circle_radius), \
                'Position list and radius list of the circle constraint have different length!'
            assert len(constr_ball_not_in_circle_circle_position) > 0, 'Not specified parameters for the circle constraint!'
            self.green_torus = []
            self.red_torus = []
            for constr_ball_not_in_circle_idx in range(len(constr_ball_not_in_circle_circle_position)):
                self.green_torus.append(Torus(constr_ball_not_in_circle_circle_position[constr_ball_not_in_circle_idx][0],
                                              constr_ball_not_in_circle_circle_position[constr_ball_not_in_circle_idx][1],
                                              constr_ball_not_in_circle_circle_radius[constr_ball_not_in_circle_idx],
                                              self,
                                              'green'))
                self.red_torus.append(Torus(constr_ball_not_in_circle_circle_position[constr_ball_not_in_circle_idx][0],
                                            constr_ball_not_in_circle_circle_position[constr_ball_not_in_circle_idx][1],
                                            constr_ball_not_in_circle_circle_radius[constr_ball_not_in_circle_idx],
                                            self,
                                            'red'))

        self.rot_x = 0
        self.rot_y = 0
        self.count_slide = 0
        self.slide = False
        self.slide_velx, self.slide_vely = 0, 0
        self.rotationMatrix = None
        self.model = None

        self.keyMap = {1: (1, 0),
                       2: (-1, 0),
                       4: (0, 1),
                       5: (1, 1),
                       6: (-1, 1),
                       7: (0, 1),
                       8: (0, -1),
                       9: (1, -1),
                       10: (-1, -1),
                       11: (0, -1),
                       13: (1, 0),
                       14: (-1, 0)}

        if self.render:

            from maze3D_new.config import MODEL_LOC, BOARD_MODEL, BOARD, TEXT_MODEL, TEXT
            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                                  GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.MODEL_LOC = MODEL_LOC
            self.BOARD_MODEL = BOARD_MODEL
            self.BOARD = BOARD
            self.TEXT_MODEL = TEXT_MODEL
            self.TEXT = TEXT
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

    def getBallCoords(self):
        return self.ball.x, self.ball.y

    def collideSquare(self, x, y):
        # if the ball hits a square obstacle, it will return True
        # and the collideTriangle will not be called

        xGrid = math.floor((x + self.num_of_boxes_x * self.box_size / 2) / self.box_size)
        yGrid = math.floor((y + self.num_of_boxes_y * self.box_size / 2) / self.box_size)

        biggest = max(xGrid, yGrid)
        smallest = min(xGrid, yGrid)
        # check the perimeter walls of the tray
        if biggest > 13 or smallest < 1:
            return True, None
        # checks collisions with corner blocks
        if self.walls[yGrid][xGrid] is not None and self.layout[yGrid][xGrid] == 1:
            return True, self.layout[yGrid][xGrid]
        return False, None

    def update(self):
        # compute rotation matrix
        rot_x_m = pyrr.Matrix44.from_x_rotation(self.rot_x)
        rot_y_m = pyrr.Matrix44.from_y_rotation(self.rot_y)
        self.rotationMatrix = pyrr.matrix44.multiply(rot_x_m, rot_y_m)

        self.ball.update()
        self.hole.update()

        if self.constr_ball_only_at_the_right_side_wrt_hole:
            self.vertical_green_line.update()
            self.vertical_red_line.update()

        if self.constr_ball_only_at_the_up_side_wrt_hole:
            self.horizontal_green_line.update()
            self.horizontal_red_line.update()

        if self.constr_ball_not_in_circle:
            assert len(self.green_torus) == len(self.red_torus), "The number of green torus is other than red torus!"
            for torus_idx in range(len(self.green_torus)):
                self.green_torus[torus_idx].update()
                self.red_torus[torus_idx].update()

        for row in self.walls:
            for wall in row:
                if wall is not None:
                    wall.update()

    def handleKeys(self, angleIncrement):
        if angleIncrement[0] == 2:
            angleIncrement[0] = -1
        elif angleIncrement[0] == 1:
            angleIncrement[0] = 1

        if angleIncrement[1] == 2:
            angleIncrement[1] = -1
        elif angleIncrement[1] == 1:
            angleIncrement[1] = 1

        self.velocity[0] = self.scaling_y * angleIncrement[0]
        self.rot_x += self.velocity[0]
        if self.rot_x >= self.max_x_rotation:
            self.rot_x = self.max_x_rotation
            self.velocity[0] = 0
        elif self.rot_x <= -self.max_x_rotation:
            self.rot_x = -self.max_x_rotation
            self.velocity[0] = 0

        self.velocity[1] = self.scaling_x * angleIncrement[1]
        self.rot_y += self.velocity[1]
        if self.rot_y >= self.max_y_rotation:
            self.rot_y = self.max_y_rotation
            self.velocity[1] = 0
        elif self.rot_y <= -self.max_y_rotation:
            self.rot_y = -self.max_y_rotation
            self.velocity[1] = 0

    def draw(self, mode=0, idx=0):
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-80, -80, 0]))
        self.model = pyrr.matrix44.multiply(translation, self.rotationMatrix)
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.BOARD_MODEL.getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.BOARD.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.BOARD_MODEL.getVertexCount())

        self.ball.draw()
        self.hole.draw()

        # Draw constraint line/torus
        if self.constr_ball_only_at_the_right_side_wrt_hole:
            assert self.vertical_red_line.x == self.vertical_green_line.x, \
                "x coordinates of vertical green line and vertical red line do not match!"
            if self.ball.x <= self.vertical_red_line.x:
                self.vertical_red_line.draw()
            else:
                self.vertical_green_line.draw()
        if self.constr_ball_only_at_the_up_side_wrt_hole:
            assert self.horizontal_red_line.x == self.horizontal_green_line.x, \
                "x coordinates of horizontal green line and horizontal red line do not match!"
            if self.ball.y <= self.horizontal_red_line.y:
                self.horizontal_red_line.draw()
            else:
                self.horizontal_green_line.draw()
        if self.constr_ball_not_in_circle:
            assert len(self.green_torus) == len(self.red_torus), "The number of green torus is other than red torus!"
            for torus_idx in range(len(self.green_torus)):
                assert self.green_torus[torus_idx].x == self.red_torus[torus_idx].x, \
                    f"x coordinates of red and green torus {torus_idx} do not match!"
                assert self.green_torus[torus_idx].y == self.red_torus[torus_idx].y, \
                    f"y coordinates of red and green torus {torus_idx} do not match!"
                if (self.red_torus[torus_idx].x - self.red_torus[torus_idx].radius_outer_circle <= self.ball.x <=
                     self.red_torus[torus_idx].x + self.red_torus[torus_idx].radius_outer_circle) and \
                   (self.red_torus[torus_idx].y - self.red_torus[torus_idx].radius_outer_circle <= self.ball.y <=
                     self.red_torus[torus_idx].y + self.red_torus[torus_idx].radius_outer_circle):
                    self.red_torus[torus_idx].draw()
                else:
                    self.green_torus[torus_idx].draw()

        for row in self.walls:
            for wall in row:
                if wall is not None:
                    wall.draw()
        # Used for resetting the game. Logs above the board "Game starts in ..."
        if mode == 1:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE,
                                    pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            self.glBindVertexArray(self.TEXT_MODEL.getVAO())
            self.glBindTexture(self.GL_TEXTURE_2D, self.TEXT[idx].getTexture())
            self.glDrawArrays(self.GL_TRIANGLES, 0, self.TEXT_MODEL.getVertexCount())
        # Used when goal has been reached. Logs above the board "Goal reached"
        elif mode == 2:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE,
                                    pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            self.glBindVertexArray(self.TEXT_MODEL.getVAO())
            self.glBindTexture(self.GL_TEXTURE_2D, self.TEXT[-2].getTexture())
            self.glDrawArrays(self.GL_TRIANGLES, 0, self.TEXT_MODEL.getVertexCount())
        # Used for resetting the game. Logs above the board "Timeout"
        elif mode == 3:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE,
                                    pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            self.glBindVertexArray(self.TEXT_MODEL.getVAO())
            self.glBindTexture(self.GL_TEXTURE_2D, self.TEXT[-1].getTexture())
            self.glDrawArrays(self.GL_TRIANGLES, 0, self.TEXT_MODEL.getVertexCount())


class Wall:
    def __init__(self, x, y, type_, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.model = None
        if type_ in [6, 7]:
            type_ = 1
        self.type = type_ - 1

        if self.parent.render:
            from maze3D_new.config import WALL_MODELS, WALL, MODEL_LOC
            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.WALL_MODELS = WALL_MODELS
            self.WALL = WALL
            self.MODEL_LOC = MODEL_LOC
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.WALL_MODELS[self.type].getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.WALL.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.WALL_MODELS[self.type].getVertexCount())

def compute_angle(nextX, nextY):
    if nextX >= 0:
        return np.arctan(nextY / nextX) * 180 / np.pi
    else:
        return 180 + np.arctan(nextY / nextX) * 180 / np.pi

def distance_from_line(p2, p1, p0):
    return norm(np.cross(p2 - p1, p1 - p0)) / norm(p2 - p1)


class Ball:
    def __init__(self, x, y, parent):
        self.exception = True
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.model = None
        self.velocity = [0, 0]
        self.box_size = box_size

        if self.parent.render:
            from maze3D_new.config import MODEL_LOC, BALL_MODEL, BALL
            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.MODEL_LOC = MODEL_LOC
            self.BALL_MODEL = BALL_MODEL
            self.BALL = BALL
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

        acceleration = [-0.1 * self.parent.rot_y, 0.1 * self.parent.rot_x]
        self.velocity[0] += 1.5 * acceleration[0]
        self.velocity[1] += 1.5 * acceleration[1]

        nextX = self.x + self.velocity[0]
        nextY = self.y + self.velocity[1]

        test_nextX = nextX + ball_diameter / 2 * np.sign(self.velocity[0])
        test_nextY = nextY + ball_diameter / 2 * np.sign(self.velocity[1])

        # check x direction
        checkXCol, gridX = self.parent.collideSquare(test_nextX, self.y)
        checkYCol, gridY = self.parent.collideSquare(self.x, test_nextY)

        if checkXCol:
            if abs(self.velocity[0]) < 0.1:
                self.velocity[0] = 0
            else:
                self.velocity[0] *= -damping_factor

        # check y direction
        if checkYCol:
            if abs(self.velocity[1]) < 0.1:
                self.velocity[1] = 0
            else:
                self.velocity[1] *= -damping_factor

        angle_from_center = compute_angle(nextX, nextY)

        # check if in the upper diagonal barrier
        if -45 <= angle_from_center <= 135:
            # if ball is in the upper triangle of the tray
            self.slide_on_upper_triangle(nextX, nextY, angle_from_center)
        elif 135 < angle_from_center <= 180 or angle_from_center <= -45:
            self.slide_on_lower_triangle(nextX, nextY, angle_from_center)

        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def draw(self):
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.BALL_MODEL.getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.BALL.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.BALL_MODEL.getVertexCount())

    def slide_on_upper_triangle(self, nextX, nextY, theta):
        # distance of a point (ball's edge towards the move direction) from a line
        p1 = np.asarray([0, self.box_size])
        p2 = np.asarray([self.box_size, 0])
        d = norm(np.cross(p2 - p1, p1 - [nextX, nextY])) / norm(p2 - p1)

        if d <= (ball_diameter / 2):
            # check if there is an opening
            if self.velocity[0] > 0 > self.velocity[1] and theta >= 90:
                if (nextX - ball_diameter/3) > -self.box_size/2:
                    self.velocity *= 1
                    return
                elif (nextX - ball_diameter/2) < -self.box_size/2 and \
                        nextY - ball_diameter/2 <= self.box_size*1.5:
                    self.velocity *= 1                                                   
                    return
            if -self.box_size/2 <= nextX - ball_diameter/2 and -self.box_size/2 <= nextY - ball_diameter/2:
                pass
            # block 2
            elif nextX - ball_diameter/2 <= self.box_size*1.5 and (nextY - ball_diameter/2) < -self.box_size/2:
                if self.velocity[1] < 0:
                    # keep going on the y-axis
                    self.velocity[1] *= - damping_factor
                    # bounce on the x-axis
                    self.velocity[0] = self.velocity[0] + self.velocity[1] * np.sin(theta * np.pi / 180)
            # block 1
            elif (nextX - ball_diameter/2) < -ball_diameter/2 and nextY - ball_diameter/2 <= 48:
                if self.velocity[0] < 0 <= self.velocity[1]:
                    # bounce on the x-axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y-axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(theta * np.pi / 180)
                elif self.velocity[0] < 0 and self.velocity[1] < 0:
                    # bounce on the x-axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y-axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(theta * np.pi / 180)
            # elif self.velocity[0] < 0 and self.velocity[1] > 0 and theta >= 90 and (nextX - ball_diameter/2) < -16 and nextY < 48:
            #         self.velocity[0] = 0.4 * self.velocity[0] + self.velocity[1] * np.cos((-theta) * np.pi / 180)
            #         self.velocity[1] *= np.cos(theta * np.pi / 180) * np.sin((90 - theta) * np.pi / 180)
            else:
                if self.velocity[0] > 0 > self.velocity[1]:
                    # bounce on the x-axis
                    if theta > 90:
                        self.velocity[0] = 0.4 * self.velocity[0] + \
                                           self.velocity[1] * np.cos((-theta) * np.pi / 180)
                        self.velocity[1] *= np.cos(theta * np.pi / 180) * np.sin((90-theta) * np.pi / 180)

                    else:
                        self.velocity[1] *= np.cos(-theta * np.pi / 180) * np.sin((-theta) * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + \
                                           self.velocity[1] * np.sin((-90+theta) * np.pi / 180)
                        # keep going on the y-axis
                # go to down right
                elif self.velocity[0] <= 0 and self.velocity[1] <= 0:
                    if theta > 90 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos(theta * np.pi / 180)
                        self.velocity[1] * np.sin(theta * np.pi / 180)
                    elif theta < 0 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos((180-theta) * np.pi / 180)
                        self.velocity[1] *= np.sin((-theta) * np.pi / 180)
                    else:
                        # keep going on the x-axis
                        self.velocity[0] *= -damping_factor
                        # bounce on the y-axis
                        self.velocity[1] *= -damping_factor
                # go up
                elif self.velocity[0] <= 0 <= self.velocity[1]:
                    if theta >= 0:
                        # keep going on the x-axis
                        self.velocity[0] *= np.sin((90-theta) * np.pi / 180) * np.cos(theta * np.pi / 180)
                        # bounce on the y-axis
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(theta * np.pi / 180)
                    else:
                        # keep going on the x-axis
                        self.velocity[0] *= np.sin(theta * np.pi / 180) * np.cos((90-theta) * np.pi / 180)
                        # bounce on the y-axis
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(90-theta * np.pi / 180)

    def slide_on_lower_triangle(self, nextX, nextY, theta):
        # define the line that the ball must not pass to insert in the frontier
        p1, p2 = np.asarray([0, -self.box_size]), np.asarray([-self.box_size, 0])
        # distance of a point (ball's edge towards the move direction) from a line
        d = distance_from_line(p2, p1, [nextX, nextY])

        # check if the ball's next center position closer than the ball's radius to the frontier line
        if d <= ball_diameter / 2:
            # check if there is an opening
            if self.velocity[0] > 0 > self.velocity[1] and theta >= 180:
                if (nextX + ball_diameter/3) > self.box_size/2:
                    self.velocity *= 1
                    return
                elif (nextX + ball_diameter/2) > self.box_size/2 and nextY - ball_diameter/2 <= 0:
                    self.velocity *= 1
                    return
            if nextX + ball_diameter/2 <= self.box_size/2 and nextY + ball_diameter/2 <= self.box_size/2:
                pass
            # block 2
            elif self.box_size/2 < nextX + ball_diameter/2 and -self.box_size*1.5 <= nextY + ball_diameter/2:
                if self.velocity[0] > 0:
                    # # bounce on the x axis
                    # self.velocity[0] *= -1 * damping_factor
                    # # keep going on the y axis
                    # self.velocity[1] *= damping_factor
                    # bounce on the x-axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y-axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(theta * np.pi / 180)
            # block 1
            elif -self.box_size*1.5 <= nextX + ball_diameter/2 and self.box_size/2 < nextY + ball_diameter/2:
                if self.velocity[1] > 0:
                    # # bounce on the x axis
                    # self.velocity[0] *= damping_factor
                    # # keep going on the y axis
                    # self.velocity[1] *= -damping_factor
                    # bounce on the x-axis
                    self.velocity[1] *= -1 * damping_factor
                    self.velocity[0] = self.velocity[0] + self.velocity[1] * np.cos((180-theta) * np.pi / 180)
                    # keep going on the y-axis

            else:
                if self.velocity[0] < 0 < self.velocity[1]:
                    if theta < -45:
                        # bounce on the y-axis
                        self.velocity[1] *= np.sin(-theta * np.pi / 180) * np.cos(theta * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + self.velocity[1] * np.sin(-90-theta * np.pi / 180)
                    else:
                        # bounce on the y-axis
                        self.velocity[1] *= np.sin(theta * np.pi / 180) * np.cos((180-theta) * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + self.velocity[1] * np.cos(-theta * np.pi / 180)

                # go to down right
                elif self.velocity[0] >= 0 and self.velocity[1] >= 0:
                    # keep going on the x axis
                    # self.velocity[0] *= -damping_factor
                    # # bounce on the y axis
                    # self.velocity[1] *= damping_factor
                    # bounce on the y-axis
                    # keep going on the x-axis
                    if theta <= -45 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos(theta * np.pi / 180)
                        self.velocity[1] *= np.sin((-theta) * np.pi / 180)
                    if theta <= 180 and norm(self.velocity) <= 1.5:
                        self.velocity[1] = self.velocity[0] * np.cos(theta * np.pi / 180)
                        self.velocity[0] *= np.sin(theta * np.pi / 180)
                    else:
                        self.velocity[0] *= - damping_factor
                        # bounce on the y-axis
                        self.velocity[1] *= damping_factor
                # go up
                elif self.velocity[0] >= 0 >= self.velocity[1]:
                    if theta < -45:
                        self.velocity[0] *= np.sin((90 - theta) * np.pi / 180) * np.cos(theta * np.pi / 180)
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(theta * np.pi / 180)
                    else:
                        # bounce on the y-axis
                        self.velocity[1] = self.velocity[0] * np.cos(theta * np.pi / 180)
                        # keep going on the x-axis
                        self.velocity[0] *= np.sin(theta * np.pi / 180)


class Hole:
    def __init__(self, x, y, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.model = None

        if self.parent.render:
            from maze3D_new.config import MODEL_LOC, HOLE_MODEL, HOLE
            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.MODEL_LOC = MODEL_LOC
            self.HOLE_MODEL = HOLE_MODEL
            self.HOLE = HOLE
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.HOLE_MODEL.getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.HOLE.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.HOLE_MODEL.getVertexCount())


class Line:

    def __init__(self, x, y, parent, color, axis):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.color = color
        self.axis = axis
        self.model = None

        if self.parent.render:
            from maze3D_new.config import MODEL_LOC

            if self.axis == 'X':
                from maze3D_new.config import HORIZONTAL_LINE_MODEL as LINE_MODEL
            elif self.axis == 'Y':
                from maze3D_new.config import VERTICAL_LINE_MODEL as LINE_MODEL
            else:
                raise NotImplementedError

            if self.color == 'green':
                from maze3D_new.config import GREEN_LINE as LINE
            elif self.color == 'red':
                from maze3D_new.config import RED_LINE as LINE
            else:
                raise NotImplementedError

            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.MODEL_LOC = MODEL_LOC
            self.LINE_MODEL = LINE_MODEL
            self.LINE = LINE
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.LINE_MODEL.getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.LINE.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.LINE_MODEL.getVertexCount())

class Torus:

    def __init__(self, x, y, radius, parent, color):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.radius_outer_circle = radius
        self.color = color
        self.model = None

        if self.parent.render is True:
            from maze3D_new.config import MODEL_LOC
            from maze3D_new.config import TORUS_MODEL

            if self.color == 'green':
                from maze3D_new.config import GREEN_TORUS as TORUS
            elif self.color == 'red':
                from maze3D_new.config import RED_TORUS as TORUS
            else:
                raise NotImplementedError

            from OpenGL.GL import glUniformMatrix4fv, glBindVertexArray, glBindTexture, glDrawArrays, \
                GL_FALSE, GL_TEXTURE_2D, GL_TRIANGLES

            self.MODEL_LOC = MODEL_LOC
            self.TORUS_MODEL = TORUS_MODEL
            self.TORUS = TORUS
            self.glUniformMatrix4fv = glUniformMatrix4fv
            self.glBindVertexArray = glBindVertexArray
            self.glBindTexture = glBindTexture
            self.glDrawArrays = glDrawArrays
            self.GL_FALSE = GL_FALSE
            self.GL_TEXTURE_2D = GL_TEXTURE_2D
            self.GL_TRIANGLES = GL_TRIANGLES

            assert self.radius_outer_circle == self.TORUS_MODEL.radius_outer_circle

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        self.glUniformMatrix4fv(self.MODEL_LOC, 1, self.GL_FALSE, self.model)
        self.glBindVertexArray(self.TORUS_MODEL.getVAO())
        self.glBindTexture(self.GL_TEXTURE_2D, self.TORUS.getTexture())
        self.glDrawArrays(self.GL_TRIANGLES, 0, self.TORUS_MODEL.getVertexCount())
