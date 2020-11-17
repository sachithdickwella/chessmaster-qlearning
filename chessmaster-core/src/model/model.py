#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class ReplayMemory(object):

    def __init__(self):
        super().__init__()


class DQE(nn.Module):

    def __init__(self):
        super(DQE, self).__init__()

    def forward(self, x):
        # do-nothing for now.
        pass
