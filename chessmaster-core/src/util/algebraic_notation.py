# -*- encoding: utf-8 -*-

import re

import numpy as np


class Board(object):

    def __init__(self):
        super(Board, self).__init__()

        self.ranks = dict(zip(range(1, 9), range(8)))
        self.f_letters = dict(zip('abcdefgh', range(8)))  # File letters of the board.

        self.pieces = dict(zip('pnbrqk', range(1, 7)))
        self._board = self.init_board()

    def __getitem__(self, item):
        if type(item) is not str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^[a-h][1-8]$', item):
            raise KeyError('Item index doesn\'t match the pattern \'^[a-h][1-8]$\'')
        else:
            return self._board[self.ranks[int(item[1])], self.f_letters[item[0]]]

    def init_board(self):
        board = np.zeros((8, 8), dtype=np.uint8)
        board[self.ranks[2], :] = board[self.ranks[7], :] = self.pieces['p']
        board[self.ranks[1], self.f_letters['a']::7] = board[self.ranks[8], self.f_letters['a']::7] = self.pieces['r']
        board[self.ranks[1], self.f_letters['b']::5] = board[self.ranks[8], self.f_letters['b']::5] = self.pieces['n']
        board[self.ranks[1], self.f_letters['c']::3] = board[self.ranks[8], self.f_letters['c']::3] = self.pieces['b']
        board[self.ranks[1], self.f_letters['d']] = board[self.ranks[8], self.f_letters['d']] = self.pieces['q']
        board[self.ranks[1], self.f_letters['e']] = board[self.ranks[8], self.f_letters['e']] = self.pieces['k']

        return board

    def move(self, move):
        if type(move) is str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^([a-h][1-8]-?)+$', move):
            raise KeyError('Item index doesn\'t match the pattern \'([a-h][1-8]-?)+$\'')
        else:
            _from, _to = move.split('-')
            if self[_from] == 0:
                return None

