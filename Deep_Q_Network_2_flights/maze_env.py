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


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.n_flights = 2
        self.action_type = ['u', 'd', 'l', 'r', 's']
        self.action_type_extend = []
        self.n_action_type = len(self.action_space)

        self.n_actions = self.n_action_type**self.n_flights -1

        for i in range(0, self.n_flights):
            self.action_type_extend.extend(self.action_type)

        self.action_space = list(set(list(permutations(self.action_type_extend, self.n_flights))))
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

        # create oval 1
        oval1_center = origin + UNIT * 2
        self.oval1 = self.canvas.create_oval(
            oval1_center[0] - 15, oval1_center[1] - 15,
            oval1_center[0] + 15, oval1_center[1] + 15,
            fill='yellow')

        # create oval 2
        oval2_center = origin + UNIT * 3
        self.oval2 = self.canvas.create_oval(
            oval2_center[0] - 15, oval2_center[1] - 15,
            oval2_center[0] + 15, oval2_center[1] + 15,
            fill='yellow')

        # create red rect 1
        self.rect1 = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # create grey rect 2
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + UNIT - 15, origin[1] - 15,
            origin[0] + UNIT + 15, origin[1] + 15,
            fill='grey')

        # create blue rect 3
        self.rect3 = self.canvas.create_rectangle(
            origin[0] + UNIT * 3 - 15, origin[1] - 15,
            origin[0] + UNIT * 3 + 15, origin[1] + 15,
            fill='blue')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)
        self.canvas.delete(self.rect3)
        origin = np.array([20, 20])
        self.rect1 = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + UNIT - 15, origin[1] - 15,
            origin[0] + UNIT + 15, origin[1] + 15,
            fill='grey')
        # return observation
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2),

    def step(self, action):
        s1 = self.canvas.coords(self.rect1)
        base_action1 = np.array([0, 0])
        for i in range(1, self.n_flights+1):
            if action//(self.n_action_type**self.(n_flights-i)) == 'u':   # up
                if s1[1] > UNIT:
                    base_action1[1] -= UNIT
            elif action[0] == 'd':   # down
                if s1[1] < (MAZE_H - 1) * UNIT:
                    base_action1[1] += UNIT
            elif action[0] == 'r':   # right
                if s1[0] < (MAZE_W - 1) * UNIT:
                    base_action1[0] += UNIT
            elif action[0] == 'l':   # left
                if s1[0] > UNIT:
                    base_action1[0] -= UNIT

        s2 = self.canvas.coords(self.rect2)
        base_action2 = np.array([0, 0])
        if action[1] == 'u':  # up
            if s2[1] > UNIT:
                base_action2[1] -= UNIT
        elif action[1] == 'd':  # down
            if s2[1] < (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action[1] == 'r':  # right
            if s2[0] < (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action[1] == 'l':  # left
            if s2[0] > UNIT:
                base_action2[0] -= UNIT

        self.canvas.move(self.rect1, base_action1[0], base_action1[1])  # move agent
        self.canvas.move(self.rect2, base_action2[0], base_action2[1])  # move agent

        s_ = [self.canvas.coords(self.rect1), self.canvas.coords(self.rect2)] # next state

        # reward function
        if s_[0] == self.canvas.coords(self.oval1) and  s_[1] == self.canvas.coords(self.oval2):
            reward = 1
            done = True
            s_ = 'terminal'
            origin = np.array([20, 20])
            self.rect3= self.canvas.create_rectangle(
                origin[0] + UNIT * 3 - 15, origin[1] - 15,
                origin[0] + UNIT * 3 + 15, origin[1] + 15,
                fill='blue')
        elif s_[0] == s_[1]:
            reward = -1
            done = True
            s_ = 'terminal'
        elif s_[0] == self.canvas.coords(self.oval1):
            reward = 0
            s_[0] = 'reached0'
            done = False
        elif s_[1] == self.canvas.coords(self.oval2):
            reward = 0
            s_[1] = 'reached1'
            done = False
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 'dr'
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()