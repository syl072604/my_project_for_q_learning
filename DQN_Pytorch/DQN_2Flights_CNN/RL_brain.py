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
                        ('state', 'action', 'reward', 'next_state', 'td_error'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, mem):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self.memory = sorted(self.memory, key=lambda s: s[4], reverse=True)
        m_index = list(range(0,batch_size//2))
        length = len(self.memory)
        m_index.extend(random.sample(range(batch_size//2,length), batch_size//2))
        mem_sampled = []
        for i in m_index:
            mem_sampled.append(self.memory[i])
        return mem_sampled, m_index

    def update(self, m_index, abs_errors):
        for i,td_e in zip(m_index, abs_errors):
            self.memory[i][4] =td_e
            # print(self.memory[i][4])

    def __len__(self):
        return len(self.memory)

class CNN(nn.Module):
    def __init__(self, n_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 8, 8)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (8, 8, 8)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (8, 4, 4)
            nn.Conv2d(8, 16, 5, 1, 2),     # output shape (16, 4, 4)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 2, 2)
        )
        self.fc1 = nn.Linear(16 * 2 * 2, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, n_actions)   # fully connected layer, output over 600 classes
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.out(x)
        return output    # return x for visualization


class DQN(object):
    def __init__(self, n_actions, n_features, n_flights, action_space):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_flights = n_flights
        self.action_space = action_space
        self.eval_net, self.target_net = CNN(self.n_actions), CNN(self.n_actions)
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
        self.memory.push([s, a, r, s_, 0.8])

    def choose_action(self, x, suggest_action_num):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            max_value = torch.max(actions_value, 1)
            action = action[0]
        elif np.random.uniform() < 0.5+0.5*self.epsilon and suggest_action_num < self.n_actions:
            action = suggest_action_num
        else:   # random
            action = np.random.randint(0, self.n_actions)
            # action = self.action_space.index(tuple(action_name))
        action_name = self.action_space[action]
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target net replaced')

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        if len(self.memory) < BATCH_SIZE:
            return
        transitions, m_index = self.memory.sample(BATCH_SIZE)
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
        abs_errors = torch.abs(q_target - q_eval).view(BATCH_SIZE).data.numpy()
        loss = self.loss_func(q_eval, q_target)
        self.memory.update(m_index, abs_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




