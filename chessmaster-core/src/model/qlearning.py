# -*- encoding: utf-8 -*-

from collections import namedtuple, OrderedDict

import numpy as np
import torch
import torch.nn as nn

from util import TEMP_PATH
from util.utility import LOGGER

TRANSITIONS = namedtuple('Transitions', ['state', 'action', 'next_state', 'reward'])
PIECES = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'r1', 'k1', 'b1', 'king', 'queen', 'b2', 'k2', 'r1']


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size: int):
        return np.random.choice(self.memory, batch_size)

    def push(self, *args):
        # do-nothing for now.
        pass


class DeepQNetwork(nn.Module):

    def __init__(self, name, nc=3, height=512, width=512):
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

        h, w = height, width
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

    def forward(self, x):
        x = self.main(x)
        return self.head(x.view(x.size(0), -1))

    def load_checkpoint_(self):
        LOGGER.info(f'... loading checkpoint of {self.name} ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def save_checkpoint_(self):
        LOGGER.info(f'... saving checkpoint of {self.name} ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1
