#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import random
from collections import namedtuple

import torch.nn as nn

TRANSITIONS = namedtuple('Transitions', [])


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)


class DQE(nn.Module):

    def __init__(self):
        super(DQE, self).__init__()

    def forward(self, x):
        # do-nothing for now.
        pass
