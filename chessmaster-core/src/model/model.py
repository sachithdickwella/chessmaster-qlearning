#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQE(nn.Module):

    def __init__(self):
        super(DQE, self).__init__()

    def forward(self, x):
        # do-nothing for now.
        pass
