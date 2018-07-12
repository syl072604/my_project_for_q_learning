"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import sys
import time

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

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.n_flights = 2
        self.n_features = 2 *self.n_flights
        self.action_type = ['u', 'd', 'l', 'r', 's']
        self.action_type_extend = []
        self.n_action_type = len(self.action_type)
        self.maze_space = [MAZE_W-1, MAZE_H-1]
        self.n_actions = self.n_action_type**self.n_flights - 1

        for i in range(0, self.n_flights):
            self.action_type_extend.extend(self.action_type)

        self.action_space = list(set(list(permutations(self.action_type_extend, self.n_flights))))
        self.action_space.sort()
        stay = []
        for i in range(0, self.n_flights):
            stay.extend('s')
        self.action_space.remove(tuple(stay))
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
            oval_center = origin + np.array([UNIT * (MAZE_H - 1 - i), UNIT * (MAZE_H-1)])
            ovals['oval'+str(i)] = self.canvas.create_oval(
                oval_center[0] - 15, oval_center[1] - 15,
                oval_center[0] + 15, oval_center[1] + 15,
                fill='yellow')

        for i in range(0,self.n_flights):
            rects['rect'+str(i)] = self.canvas.create_rectangle(
                origin[0] + i * UNIT - 15, origin[1] - 15,
                origin[0] + i * UNIT + 15, origin[1] + 15,
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
            rects['rect'+str(i)] = self.canvas.create_rectangle(
                origin[0] + i * UNIT - 15, origin[1] - 15,
                origin[0] + i * UNIT + 15, origin[1] + 15,
                fill='red')

        self.canvas.delete(self.flag)
        r = []
        for i in range(0, self.n_flights):
            r.extend((np.array(self.canvas.coords(rects['rect'+str(i)])[:2]) - np.array([5, 5]))/UNIT)
        # return observation
        return np.array(r)

    def step(self, action):
        states = locals()
        action_name = self.action_space[action]
        for i in range(0, self.n_flights):
            states['s'+str(i)] = self.canvas.coords(rects['rect'+str(i)])
            base_action = np.array([0, 0])
            if action_name[i] == 'u':   # up
                if states['s'+str(i)][1] > UNIT:
                    base_action[1] -= UNIT
            elif action_name[i] == 'd':   # down
                if states['s'+str(i)][1] < (MAZE_H - 1) * UNIT:
                    base_action[1] += UNIT
            elif action_name[i] == 'r':   # right
                if states['s'+str(i)][0] < (MAZE_W - 1) * UNIT:
                    base_action[0] += UNIT
            elif action_name[i] == 'l':   # left
                if states['s'+str(i)][0] > UNIT:
                    base_action[0] -= UNIT
            self.canvas.move(rects['rect'+str(i)], base_action[0], base_action[1])  # move agent

        s_ = []
        ss_ = []
        ss_ovals = []
        for i in range(0, self.n_flights):
            s_.extend((np.array(self.canvas.coords(rects['rect'+str(i)])[:2]) - np.array([5, 5]))/UNIT)
            ss_.append(self.canvas.coords(rects['rect' + str(i)]))
            ss_ovals.append(self.canvas.coords(ovals['oval'+str(i)]))
        # reward function

        reward = 0
        done = False
        achieved = False
        for i in range(0, self.n_flights):
            if ss_.count(ss_[i]) > 1:
                reward = -1
                done = True
                break

        if not done:
            reached_flag = True
            for i in range(0, self.n_flights):
                if ss_[i] not in ss_ovals:
                    reached_flag = False
                    break

            if reached_flag:
                reward = 1
                done = True
                achieved = True
                origin = np.array([20, 20])
                self.flag= self.canvas.create_rectangle(
                    origin[0] + UNIT * 3 - 15, origin[1] - 15,
                    origin[0] + UNIT * 3 + 15, origin[1] + 15,
                    fill='blue')
        # if not done:
        #     for i in range(0, self.n_flights):
        #         if ss_[i] in ss_ovals:
        #              s_[2 * i] = 999
        #              s_[2 * i + 1] = 999

        return np.array(s_), reward, done, achieved

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 23
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()