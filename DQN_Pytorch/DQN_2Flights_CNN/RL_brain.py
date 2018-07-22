"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FEATURES = 4
N_ACTIONS = 24

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
EPSILON_INCREMENT = 0.005
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

np.random.seed(1)
tf.set_random_seed(1)

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 4, 4)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=4,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (4, 4, 4)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (4, 2, 2)
            nn.Conv2d(4, 8, 5, 1, 2),     # output shape (8, 2, 2)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (8, 1, 1)
        )
        self.out = nn.Linear(8 * 1 * 1, N_ACTIONS)   # fully connected layer, output 24 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = CNN(), CNN()

        self.learn_step_counter = 0                                     # for target updating
        self.epsilon = 0
        self.epsilon_max = EPSILON
        self.epsilon_increment = EPSILON_INCREMENT

        self.memory = ReplayMemory(2000)     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)
        a = torch.LongTensor([[a]])
        r = torch.FloatTensor([[r]])
        self.memory.push(s, a, r, s_)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        b_s = torch.cat(batch.state)
        b_a = torch.cat(batch.action)
        b_r = torch.cat(batch.reward)
        b_s_ = torch.cat(batch.next_state)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        test_a = GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = b_r + test_a   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




