# -*- encoding: utf-8 -*-

import re

import numpy as np

FLAGS = {
    'n': 'a non capture',
    'b': 'a pawn push of two squares',
    'e': 'an en passant capture',
    'c': 'a standard capture',
    'p': 'a promotion',
    'k': 'king-side castling',
    'q': 'queen-side castling'
}


class Board(object):

    def __init__(self):
        super(Board, self).__init__()

        self.ranks = dict(zip(range(1, 9), range(8)))
        self.f_letters = dict(zip('abcdefgh', range(8)))  # File letters of the board.

        self.pieces = dict(zip('pnbrqk', range(1, 7)))
        self._board = self.init_board()

    def __getitem__(self, item):
        """
        Get the square value from the algebraic chess notation. If a chess piece is available
        on that square, return the piece value. Otherwise 0 returns.

        :param item: algebraic notation of the square (ex: e4, a8, b6).
        :return: the square value for input algebraic notation.
        """
        if type(item) is not str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^[a-hA-H][1-8]$', item):
            raise KeyError('Item index doesn\'t match the pattern \'^[a-h][1-8]$\'')
        else:
            item = item.lower()
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
        """
        Get the move details by passing the algebraic notation of the move and return
        'None' of the move is an illegal.

        :param move: algebraic notation of 2 squares that the piece should moved from
        and the destination (ex: b2-b4).
        :return: an instance of :func:`~Move` class with the details of source, target,
        color, flag, target SAN and piece if it's legal move. Otherwise 'None' returns.
        """
        if type(move) is str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^([a-hA-H][1-8]-?)+$', move):
            raise KeyError('Item index doesn\'t match the pattern \'([a-h][1-8]-?)+$\'')
        else:
            _from, _to = move.lower().split('-')
            if self[_from] == 0:
                return None


class Move(object):

    def __init__(self, color, flag, _from, _to, piece, san, captured=None):
        super(Move, self).__init__()

        self.color = color
        self.flag = flag
        self._from = _from
        self._to = _to
        self.piece = piece
        self.san = san
        self.captured = captured

    def dict(self):
        details: dict = {
            'color': self.color,
            'flag': self.flag,
            'from': self._from,
            'to': self._to,
            'piece': self.piece,
            'san': self.san
        }

        if self.captured is not None:
            details['captured'] = self.captured
        return details
