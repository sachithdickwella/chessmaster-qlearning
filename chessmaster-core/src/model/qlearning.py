# -*- coding: utf-8 -*-

from collections import namedtuple, OrderedDict

import chess
import numpy as np
import torch
import torch.nn as nn

from utils import TEMP_PATH, LOGGER
from . import DEVICE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE

BATCH_SIZE = 64
TRANSITIONS = namedtuple('Transitions', ['piece', 'state', 'action', 'next_state', 'reward'])

PIECES = namedtuple('Pieces', ('PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING'))(*'pnbrqk')
PIECES_ENCODING = list(dict(enumerate(PIECES)).keys())

BOARD_SQUARES = [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8,
                 chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7,
                 chess.A6, chess.B6, chess.C6, chess.D6, chess.E6, chess.F6, chess.G6, chess.H6,
                 chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5, chess.G5, chess.H5,
                 chess.A4, chess.B4, chess.C4, chess.D4, chess.E4, chess.F4, chess.G4, chess.H4,
                 chess.A3, chess.B3, chess.C3, chess.D3, chess.E3, chess.F3, chess.G3, chess.H3,
                 chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2,
                 chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1]

BOARD_SQUARES_ENCODING = list(dict(enumerate(BOARD_SQUARES)).keys())


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

    def __init__(self, alpha=0.001, mem_size=10_000, action_space=tuple(range(8))):
        super().__init__()

        self.GAMMA = GAMMA
        self.EPSILON_START = EPS_START
        self.EPSILON_END = EPS_END
        self.EPSILON_DECAY = EPS_DECAY

        self.action_space = action_space
        self.episodes = 0  # Number of state transitions.
        self.n_episodes = 0  # Number of time learn function called to replace the 'target' network.

        self.memory = ReplayMemory(capacity=mem_size)

        self.policy_net = DeepQNetwork("policy", alpha)  # Estimate of the current set of states.
        self.target_net = DeepQNetwork("target", alpha)  # Estimate of the successor set of states.

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, observations):
        rand = np.random.random()

        if rand < 1 + self.EPSILON_START:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(observations).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.choice(self.action_space)]], device=DEVICE)

    def learn(self):
        self.policy_net.optimizer.zero_grad()

        if self.n_episodes % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        sample = self.memory.sample(BATCH_SIZE)
        batch = TRANSITIONS(*zip(*sample))

        # torch.ga
