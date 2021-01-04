# Copyright 2020 Sachith Prasanna Dickwella (sdickwella@outlook.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -*- encoding: utf-8 -*-

import re
from collections import namedtuple

import numpy as np

# Players of the game.
PLAYERS = namedtuple('Players', ('WHITE', 'BLACK'))(*'wb')
PLAYERS_BITS = namedtuple('PlayersBits', ('WHITE', 'BLACK'))(1, 2)
# Flags for the movement.
FLAGS = namedtuple('Flags', ('NORMAL',
                             'CAPTURE',
                             'BIG_PAWN',
                             'EP_CAPTURE',
                             'PROMOTION',
                             'KING_SIDE_CASTLE',
                             'QUEEN_SIDE_CASTLE'))(*'ncbepkq')
# Chess pieces of the board.
PIECES = namedtuple('Pieces', ('PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING'))(*'pnbrqk')


class Board(object):

    def __init__(self):
        super(Board, self).__init__()

        self.ranks = dict(zip(range(1, 9), range(8)))
        self.f_letters = dict(zip('abcdefgh', range(8)))  # File letters of the board.

        self.pieces = dict(zip('pnbrqk', range(1, 7)))
        self._board, self.c_board = self.init_board()

        self._turn = PLAYERS_BITS.WHITE

    def get_square(self, item, color=False):
        """
        Get the square value from the algebraic chess notation. If a chess piece is available
        on that square, return the piece value. Otherwise 0 returns.

        :param item: algebraic notation of the square (ex: e4, a8, b6).
        :param color: boolean expression to get color value or piece value from the square.
        :return: the square value for input algebraic notation.
        """
        if type(item) is not str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^[a-hA-H][1-8]$', item):
            raise KeyError('Item index does not match the pattern \'^[a-h][1-8]$\'')
        else:
            item = item.lower()
            if color:
                return self._board[self.ranks[int(item[1])], self.f_letters[item[0]]]
            else:
                return self.c_board[self.ranks[int(item[1])], self.f_letters[item[0]]]

    def init_board(self):
        # Board with pieces location despite color of the pieces.
        board = np.zeros((8, 8), dtype=np.uint8)
        board[self.ranks[2], :] = board[self.ranks[7], :] = self.pieces[PIECES.PAWN]
        board[self.ranks[1], self.f_letters['a']::7] = board[self.ranks[8], self.f_letters['a']::7] = self.pieces[
            PIECES.ROOK]
        board[self.ranks[1], self.f_letters['b']::5] = board[self.ranks[8], self.f_letters['b']::5] = self.pieces[
            PIECES.KNIGHT]
        board[self.ranks[1], self.f_letters['c']::3] = board[self.ranks[8], self.f_letters['c']::3] = self.pieces[
            PIECES.BISHOP]
        board[self.ranks[1], self.f_letters['d']] = board[self.ranks[8], self.f_letters['d']] = self.pieces[
            PIECES.QUEEN]
        board[self.ranks[1], self.f_letters['e']] = board[self.ranks[8], self.f_letters['e']] = self.pieces[PIECES.KING]

        # Board with the colors of pieces despite the type of the pieces.
        _board = np.zeros((8, 8), dtype=np.uint8)
        _board[:2] = PLAYERS_BITS.BLACK
        _board[6:] = PLAYERS_BITS.WHITE

        return board, _board

    def toggle_player(self):
        """
        Toggle the :func:`~self._turn` value depending on the current value. Nothing returns.
        """
        self._turn = PLAYERS_BITS.WHITE if self._turn == PLAYERS_BITS.BLACK else PLAYERS_BITS.BLACK

    def move(self, move):
        """
        Get the move details by passing the algebraic notation of the move and return
        'None' if the move is an illegal.

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

            piece, p_color = self.get_square(_from), self.get_square(_from, True)
            d_piece, dp_color = self.get_square(_to), self.get_square(_to, True)

            if (not piece or self._turn != p_color) \
                    or (d_piece and self._turn == dp_color):
                return None

            moves = self.generate_moves(piece, p_color)

            if _to in moves:
                # TODO - define a valid 'flag' and 'captured' value on return value.
                return Move(self._turn, _from, _to, piece, f'{piece}{_to}', *moves[_to])
            else:
                return None

    @staticmethod
    def generate_moves(piece, color):
        # TODO - work on decision of valid move destinations.
        return {'to': ('flag', 'captured')}


class Move(object):

    def __init__(self, color, _from, _to, piece, san, flag, captured=None):
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
