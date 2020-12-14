# -*- encoding: utf-8 -*-

from collections import namedtuple, OrderedDict

import numpy as np
import torch
import torch.nn as nn

from util import TEMP_PATH, LOGGER
from . import DEVICE, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY

TRANSITIONS = namedtuple('Transitions', ['piece', 'state', 'action', 'next_state', 'reward'])
PIECES = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'r1', 'k1', 'b1', 'king', 'queen', 'b2', 'k2', 'r1']


class ReplayMemory(object):

    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = TRANSITIONS(*args)
        self.position = (self.position + 1) % self.capacity


class DeepQNetwork(nn.Module):

    def __init__(self, name, alpha=0.001, nc=1, dim=(512, 512)):
        """
        Initialize the 'DeepQNetwork' model instance with custom parameters.

        :param name: model instance name hence there can multiple instances for different purposes.
        :param alpha: learning rate for the optimizer.
        :param nc: number of input channel of the initial image stack.
        :param dim: dimensions of the initial input image stack.
        """
        super(DeepQNetwork, self).__init__()

        self.name = name
        self.checkpoint_file = f'{TEMP_PATH}/{self.name}.pt'
        self.main = nn.Sequential(OrderedDict([
            (
                'conv1', nn.Sequential(
                    nn.Conv2d(in_channels=nc, out_channels=32, kernel_size=5, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                )
            ), (
                'conv2', nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                )
            ), (
                'conv3', nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                )
            ), (
                'conv4', nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                )
            ), (
                'conv5', nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
            ), (
                'conv6', nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
            )
        ]))

        h, w = dim
        for _ in range(len(self.main)):
            h = self.conv2d_size_out(h)
            w = self.conv2d_size_out(w)

        self.head = nn.Sequential(
            nn.Linear(in_features=512 * h * w, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.LogSoftmax(dim=1)
        )

        self.to(device=DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha, betas=(0.9, 0.999))

    def forward(self, x):
        """
        Feed forward the input throughout the network by this method.

        :param x: input image stack for the network.
        :return: the out of the model network after feed forward is done.
        """
        x = self.main(x.to(DEVICE))
        return self.head(x.view(x.size(0), -1))

    def load_checkpoint_(self):
        LOGGER.info(f'... loading checkpoint of {self.name} ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def save_checkpoint_(self):
        LOGGER.info(f'... saving checkpoint of {self.name} ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        """
        Calculate the final dimensions of the images after going through
        the convolutional networks.

        :param size: current size of the frame (height, width).
        :param kernel_size: kernel size of each of the convolution layer.
        :param stride: strides through the each convolution layer.
        :return: the final size of the image after convolution.
        """
        return (size - (kernel_size - 1) - 1) // stride + 1


class Agent(object):

    def __init__(self, alpha=0.001, target=1_000, mem_size=10_000, action_space=tuple(range(8))):
        super().__init__()

        self.GAMMA = GAMMA
        self.EPSILON_START = EPS_START
        self.EPSILON_END = EPS_END
        self.EPSILON_DECAY = EPS_DECAY

        self.action_space = action_space
        self.steps = 0                  # Number of state transitions.
        self.learn_step_counter = 0     # Number of time learn function called to replace the 'target' network.

        self.memory = ReplayMemory(capacity=mem_size)

        self.replace_target_cnt = target

        self.q_eval = DeepQNetwork("eval", alpha)  # Estimate of the current set of states.
        self.q_next = DeepQNetwork("next", alpha)  # Estimate of the successor set of states.

    def choose_action(self, observations):
        random = np.random.random()
        actions = self.q_eval(observations)

        if random < 1 + self.EPSILON_START:
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        self.q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        sample = self.memory.sample(BATCH_SIZE)
