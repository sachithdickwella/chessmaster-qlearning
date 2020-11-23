# -*- encoding: utf-8 -*-

import random
from collections import namedtuple, OrderedDict

import torch.nn as nn

TRANSITIONS = namedtuple('Transitions', ['state', 'action', 'next_state', 'reward'])


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def push(self, *args):
        # do-nothing for now.
        pass


class DQE(nn.Module):

    def __init__(self, nc=3, height=508, width=512):
        super(DQE, self).__init__()
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
            nn.Linear(in_features=512 * h * w, out_features=20),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x)
        return self.head(x.view(x.size(0), -1))

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1
