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

BATCH_SIZE = 1024
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
EPSILON_INCREMENT = 0.002
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 5000

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
        else:
            self.memory[self.position + 4] = mem
            self.position = (self.position + 1) % (self.capacity - 4)

    def sample(self, batch_size):
        self.memory = sorted(self.memory, key=lambda s: s[4], reverse=True)
        m_index = list(range(0, batch_size//16))
        length = len(self.memory)
        m_index.extend(random.sample(range(batch_size//16, length), batch_size*15//16))
        mem_sampled = []
        for i in m_index:
            mem_sampled.append(self.memory[i])
        return mem_sampled, m_index

    def update(self, m_index, abs_errors):
        for i,td_e in zip(m_index, abs_errors):
            self.memory[i][4] = td_e
            # print(self.memory[i][4])

    def __len__(self):
        return len(self.memory)

class NET(nn.Module):
    def __init__(self, n_features, n_actions):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(1000, n_actions)   # fully connected layer, output over 600 classes
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.out(x)
        return output    # return x for visualization


class DQN(object):
    def __init__(self, n_actions, n_features, n_flights, action_space):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_flights = n_flights
        self.action_space = action_space
        self.eval_net, self.target_net = NET(self.n_features, self.n_actions), NET(self.n_features,self.n_actions)
        self.learn_step_counter = 0                                     # for target updating
        self.epsilon = 0
        self.epsilon_max = EPSILON
        self.epsilon_increment = EPSILON_INCREMENT

        self.memory = ReplayMemory(2000)     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)
        abs_error_int = torch.max(torch.abs(torch.FloatTensor([r, 1.0]))).data.numpy()
        a = torch.LongTensor([[a]])
        r = torch.FloatTensor([[r]])
        self.memory.push([s, a, r, s_, abs_error_int])

    def choose_action(self, x, suggest_action_num, force_suggest):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon and not force_suggest:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            max_value = torch.max(actions_value, 1)
            action = action[0]
        elif (np.random.uniform() < 0.5+0.5*self.epsilon or force_suggest) and suggest_action_num < self.n_actions:
            action = suggest_action_num
            # if force_suggest:
                # print('force suggest works')
        else:   # random
            action = np.random.randint(0, self.n_actions)
            # action = self.action_space.index(tuple(action_name))
        action_name = self.action_space[action]
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('target net replaced')

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

        # q_eval4next = self.eval_net(b_s_).detach()  # detach from graph, don't backpropagate
        # b_a_ = q_eval4next.max(1)[1].view(BATCH_SIZE, 1)
        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # selected_q_next = q_next.gather(1, b_a_)
        # q_target = b_r + GAMMA*selected_q_next   # shape (batch, 1)

        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        test_a = GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = b_r + test_a   # shape (batch, 1)

        abs_errors = torch.abs(q_target - q_eval).view(BATCH_SIZE).data.numpy()
        loss = self.loss_func(q_eval, q_target)
        self.memory.update(m_index, abs_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




