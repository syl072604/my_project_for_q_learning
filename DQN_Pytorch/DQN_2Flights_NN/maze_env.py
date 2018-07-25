"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from __future__ import division
import sys
import time
import random
import numpy as np
from itertools import permutations
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

ovals = globals()
rects = globals()

origin_position = [[0,0],
                   [1,0],
                   [2,0],
                   [3,0]]
target_position = [[1,3],
                   [1,1],
                   [3,1],
                   [3,3]]

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.crash = False
        self.n_flights = 4
        self.n_features = 2 *self.n_flights
        self.action_type = ['s', 'u', 'd', 'r', 'l']   #['s, 'u', 'd', 'r', 'l']
        self.action_type_extend = []
        self.n_action_type = len(self.action_type)
        self.maze_space = [MAZE_W-1, MAZE_H-1]
        self.n_actions = self.n_action_type**self.n_flights - 1
        self.action_space = []
        for i in range(0, self.n_actions+1):
            action_name = []
            n = i
            for j in range(1, self.n_flights+1):
                a = n // self.n_action_type**(self.n_flights-j)
                action_name.extend(self.action_type[a])
                n = n % self.n_action_type**(self.n_flights-j)
            self.action_space.append(action_name)
        # stay = []
        # for i in range(0, self.n_flights):
        #     stay.extend('0')
        # self.action_space.remove(stay)
        # ['uuu', 'uud', 'uul', 'uur', 'uus',
        #  'udu', 'udd', 'udl', 'udr', 'uds',
        #  'ulu', 'uld', 'ull', 'ulr', 'uls',
        #  'uru', 'urd', 'url', 'urr', 'urs',
        #  'usu', 'usd', 'usl', 'usr', 'uss',
        #  'duu', 'dud', 'dul', 'dur', 'dus',
        #  'ddu', 'ddd', 'ddl', 'ddr', 'dds',
        #  'dlu', 'dld', 'dll', 'dlr', 'dls',
        #  'dru', 'drd', 'drl', 'drr', 'drs',
        #  'dsu', 'dsd', 'dsl', 'dsr', 'dss',
        #  ...]

        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):


        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        for i in range(0,self.n_flights):
            oval_center = origin + np.array([UNIT * target_position[i][0], UNIT *  target_position[i][1]])
            ovals['oval'+str(i)] = self.canvas.create_oval(
                oval_center[0] - 15, oval_center[1] - 15,
                oval_center[0] + 15, oval_center[1] + 15,
                fill='yellow')

        for i in range(0,self.n_flights):
            rect_center = origin + np.array([UNIT * origin_position[i][0], UNIT *  origin_position[i][1]])
            rects['rect'+str(i)] = self.canvas.create_rectangle(
                rect_center[0] - 15, rect_center[1] - 15,
                rect_center[0] + 15, rect_center[1] + 15,
                fill='red')

        self.flag = self.canvas.create_rectangle(
            origin[0] + (MAZE_H-1) * UNIT - 15, origin[1] - 15,
            origin[0] + (MAZE_H-1) * UNIT + 15, origin[1] + 15,
            fill='blue')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        origin = np.array([20, 20])
        for i in range(0, self.n_flights):
            self.canvas.delete(rects['rect'+str(i)])
            rect_center = origin + np.array([UNIT * origin_position[i][0], UNIT * origin_position[i][1]])
            rects['rect'+str(i)] = self.canvas.create_rectangle(
                rect_center[0] - 15, rect_center[1] - 15,
                rect_center[0] + 15, rect_center[1] + 15,
                fill='red')

        self.canvas.delete(self.flag)
        s_ = []
        ss_ = []
        ss_ovals = []
        suggest_action = []
        for i in range(0, self.n_flights):
            rect_index = (np.array(self.canvas.coords(rects['rect'+str(i)])[:2],dtype=int) - np.array([5, 5]))//UNIT
            oval_index = (np.array(self.canvas.coords(rects['oval'+str(i)])[:2],dtype=int) - np.array([5, 5]))//UNIT
            ss_.append(self.canvas.coords(rects['rect' + str(i)]))
            ss_ovals.append(self.canvas.coords(ovals['oval'+str(i)]))
            distance = oval_index - rect_index
            s_.extend(distance)
            diff_ss = np.array(ss_ovals[i])[:2] - np.array(ss_[i])[:2]

            if diff_ss[0] > 0:                   # right
                if diff_ss[1] > 0:                  # down
                    action_list = ['r', 'd']
                elif diff_ss[1] < 0:                # up
                    action_list = ['r', 'u']
                else:
                    action_list = ['r', 's']        # stay

            elif diff_ss[0] < 0:                 # left
                if diff_ss[1] > 0:                  # down
                    action_list = ['l', 'd']
                elif diff_ss[1] < 0:                # up
                    action_list = ['l', 'u']
                else:
                    action_list = ['l', 's']        # stay

            elif diff_ss[1] > 0:                # down
                action_list = ['d', 's']            # stay

            elif diff_ss[1] < 0:                # up
                action_list = ['u', 's']            # stay

            else:
                action_list = ['s']  # stay

            a = random.sample(action_list, 1)
            suggest_action.extend(a)
        suggest_action_num = self.action_space.index(suggest_action)

        # return observation
        return s_, suggest_action_num

    def step(self, action, ignore_crash):

        if self.crash:
            can_be_stored = False
            self.crash = False
        else:
            can_be_stored = True

        states = locals()
        action_name = self.action_space[action]
        out_of_bond = False
        for i in range(0, self.n_flights):
            states['s'+str(i)] = self.canvas.coords(rects['rect'+str(i)])
            base_action = np.array([0, 0])
            if action_name[i] == 'u':   # up
                if states['s'+str(i)][1] > UNIT:
                    base_action[1] -= UNIT
                else:
                    out_of_bond = True
                    break
            elif action_name[i] == 'd':   # down
                if states['s'+str(i)][1] < (MAZE_H - 1) * UNIT:
                    base_action[1] += UNIT
                else:
                    out_of_bond = True
                    break
            elif action_name[i] == 'r':   # right
                if states['s'+str(i)][0] < (MAZE_W - 1) * UNIT:
                    base_action[0] += UNIT
                else:
                    out_of_bond = True
                    break
            elif action_name[i] == 'l':   # left
                if states['s'+str(i)][0] > UNIT:
                    base_action[0] -= UNIT
                else:
                    out_of_bond = True
                    break
            self.canvas.move(rects['rect'+str(i)], base_action[0], base_action[1])  # move agent

        s_ = []
        ss_ = []
        ss_ovals = []
        for i in range(0, self.n_flights):
            rect_index = (np.array(self.canvas.coords(rects['rect'+str(i)])[:2],dtype=int) - np.array([5, 5]))//UNIT
            oval_index = (np.array(self.canvas.coords(rects['oval'+str(i)])[:2],dtype=int) - np.array([5, 5]))//UNIT
            distance = oval_index - rect_index
            s_.extend(distance)
            ss_.append(self.canvas.coords(rects['rect' + str(i)]))
            ss_ovals.append(self.canvas.coords(ovals['oval'+str(i)]))

        # reward function
        reward = 0
        done = False
        achieved = False
        suggest_action = []
        stay_index = []
        suggest_action_num = self.n_actions

        if out_of_bond:
            reward = -1
            done = True
        else:
            for i in range(0, self.n_flights):
                if ss_.count(ss_[i]) > 1:
                    reward = -1
                    if ignore_crash:
                        done = False
                        self.crash = True
                    else:
                        down = True
                    break

        if not done:
            reached_flag = True
            for i in range(0, self.n_flights):
                if ss_[i] != ss_ovals[i]:
                    reached_flag = False

            if reached_flag:
                reward = 1
                done = True
                achieved = True
                origin = np.array([20, 20])
                self.flag = self.canvas.create_rectangle(
                    origin[0] + UNIT * 3 - 15, origin[1] - 15,
                    origin[0] + UNIT * 3 + 15, origin[1] + 15,
                    fill='blue')
            else:
                free_ss_count = 0
                for i in range(0, self.n_flights):
                    if i not in stay_index:
                        diff_ss = np.array(ss_ovals[i])[:2] - np.array(ss_[i])[:2]
                        if diff_ss[0] > 0:  # right
                            if diff_ss[1] > 0:  # down
                                action_list = ['r', 'd']
                            elif diff_ss[1] < 0:  # up
                                action_list = ['r', 'u']
                            else:
                                action_list = ['r', 's']  # stay

                        elif diff_ss[0] < 0:  # left
                            if diff_ss[1] > 0:  # down
                                action_list = ['l', 'd']
                            elif diff_ss[1] < 0:  # up
                                action_list = ['l', 'u']
                            else:
                                action_list = ['l', 's']  # stay

                        elif diff_ss[1] > 0:  # down
                            action_list = ['d', 's']  # stay

                        elif diff_ss[1] < 0:  # up
                            action_list = ['u', 's']  # stay
                        else:
                            action_list = ['s']  # stay

                        a = random.sample(action_list, 1)
                        suggest_action.extend(a)
                        free_ss_count = free_ss_count + 1
                    else:
                        suggest_action.append('s')                                 # stay
                suggest_action_num = self.action_space.index(suggest_action)
        return s_, reward, done, achieved, suggest_action_num, can_be_stored

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 12
            s, r, done, achieved, suggest_action_num = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()