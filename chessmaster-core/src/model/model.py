#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from collections import namedtuple

TRANSITIONS = namedtuple('Transitions', [])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class DQE(nn.Module):

    def __init__(self):
        super(DQE, self).__init__()

    def forward(self, x):
        # do-nothing for now.
        pass
