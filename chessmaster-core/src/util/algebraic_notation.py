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

    def __str__(self):
        return str(self.dict())

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


class Board(object):

    def __init__(self):
        super(Board, self).__init__()
        self.last_move = {}

        self.ranks = dict(zip(range(1, 9), range(8)))
        self.f_letters = dict(zip('abcdefgh', range(8)))  # File letters of the board.

        self.pieces = dict(zip('pnbrqk', range(1, 7)))
        self._board, self.c_board = self.setup_board()

        self._turn = PLAYERS_BITS.WHITE

    def __str__(self):
        return "Pieces Location:\n" \
               "  {}\n" \
               "Pieces Location in Color:\n" \
               "  {}".format(np.array2string(self._board, separator=', ', prefix='\t'),
                             np.array2string(self.c_board, separator=', ', prefix='\t'))

    def setup_board(self):
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
        board[self.ranks[1], self.f_letters['e']] = board[self.ranks[8], self.f_letters['e']] = self.pieces[
            PIECES.KING]

        # Board with the colors of pieces despite the type of the pieces.
        _board = np.zeros((8, 8), dtype=np.uint8)
        _board[:2] = PLAYERS_BITS.BLACK
        _board[6:] = PLAYERS_BITS.WHITE

        return board, _board

    def update_board(self, _from, _to, flag):
        f_piece, _, f_loc = self.square(_from)
        _, _, t_loc = self.square(_to)

        if PIECES[f_piece - 1] == PIECES.PAWN and _from not in self.last_move:  # Yet strict for PAWNs.
            self.last_move[_from] = (_to, flag)

        self._board[t_loc] = self._board[f_loc]
        self.c_board[t_loc] = self.c_board[f_loc]

        self._board[f_loc] = self.c_board[f_loc] = 0

    def square(self, san):
        """
        Get the square value from the algebraic chess notation. If a chess piece is available
        on that square, return the piece value. Otherwise 0 returns.

        :param san: algebraic notation of the square (ex: e4, a8, b6).
        :return: the square value for input algebraic notation.
        """

        if type(san) is not str and type(san) is not tuple:
            raise TypeError('Item index should be string Algebraic Notation or Tuple index location')
        elif type(san) is str and not re.match('^[a-hA-H][1-8]$', san):
            raise KeyError('Item index does not match the pattern \'^[a-h][1-8]$\'')
        elif type(san) is tuple and len(san) != 2:
            raise KeyError(f'Item index should be 2-dimensional not {len(san)}-dimensional')
        else:
            if type(san) is str:
                san = san.lower()
                index = (self.ranks[(len(self.ranks) + 1) - int(san[1])], self.f_letters[san[0]])

                return namedtuple('Square', ('piece', 'color', 'location'))(
                    self._board[index], self.c_board[index], index
                )
            else:
                try:
                    file_letter = next(k for k, v in self.f_letters.items() if v == san[1])
                    rank = next(k for k, v in self.ranks.items() if v == ((len(self.ranks) - 1) - san[0]))

                    san = file_letter + str(rank)
                    square = self.square(san)

                    return namedtuple('Square', ('piece', 'color', 'location'))(square.piece, square.color, san)
                except StopIteration:
                    raise IndexError(f'Indexes should be between 0 and 7: not {san}')

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
        if type(move) is not str:
            raise TypeError('Item index should be Algebraic Notation')
        elif not re.match('^([a-hA-H][1-8]-?)+$', move):
            raise KeyError('Item index does not match the pattern \'([a-h][1-8]-?)+$\'')
        else:
            _from, _to = move.lower().split('-')

            piece, p_color, _ = self.square(_from)
            d_piece, dp_color, _ = self.square(_to)

            if (not piece or self._turn != p_color) \
                    or (d_piece and self._turn == dp_color):
                return None

            moves = self.generate_moves(_from)
            print(moves)

            if _to in moves:
                self.update_board(_from, _to, moves[_to][0])
                self.toggle_player()

                return Move(p_color, _from, _to, piece, f'{piece}{_to}', *moves[_to])
            else:
                return None

    def generate_moves(self, _from):  # NOSONAR
        piece, color, loc = self.square(_from)

        def pawn():
            min_r, max_r = (loc[0] - 1 if loc[0] > 0 else 0, loc[0] + 1 if loc[0] < 7 else 7)
            min_c, max_c = (loc[1] - 1 if loc[1] > 0 else 0, loc[1] + 1 if loc[1] < 7 else 7)

            out = {}  # Structure -> {'to': ('flag', 'captured=n|c|b|e|p|k|q')}
            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if loc != (i, j):
                        _to = self.square((i, j))
                        if _to.piece and _to.color != color \
                                and ((i > loc[0] and self._turn == PLAYERS_BITS.BLACK)
                                     or (i < loc[0] and self._turn == PLAYERS_BITS.WHITE)) \
                                and (j == loc[1] + 1 or j == loc[1] - 1):
                            out[_to.location] = (FLAGS.CAPTURE, PIECES[_to.piece - 1])

                        elif not _to.piece and j == loc[1] \
                                and ((i == loc[0] + 1 and self._turn == PLAYERS_BITS.BLACK)
                                     or (i == loc[0] - 1 and self._turn == PLAYERS_BITS.WHITE)):
                            out[_to.location] = (FLAGS.NORMAL,)

                        # TODO - Revalidate the below logic for confirmation.
                        elif not _to.piece and j != loc[1] \
                                and ((self._turn == PLAYERS_BITS.BLACK and loc[0] == 4 and i == loc[0] + 1)
                                     or (self._turn == PLAYERS_BITS.WHITE and loc[0] == 3) and i == loc[0] - 1) \
                                and loc[0] == i and (loc[1] > j or loc[1] < j):

                            en_passant = self.square((i, j))
                            if en_passant.color != color \
                                    and next(fl for k, (to, fl) in self.last_move.items()
                                             if to == en_passant.location) == FLAGS.BIG_PAWN:
                                out[_to.location] = (FLAGS.EP_CAPTURE, PIECES[_to.piece - 1])

            if self._turn == color and color == PLAYERS_BITS.BLACK and loc[0] == 1:
                _to = self.square((loc[0] + 2, loc[1]))
                if not _to.piece and _from not in self.last_move:
                    out[_to.location] = (FLAGS.BIG_PAWN,)

            elif self._turn == color and color == PLAYERS_BITS.WHITE and loc[0] == 6:
                _to = self.square((loc[0] - 2, loc[1]))
                if not _to.piece and _from not in self.last_move:
                    out[_to.location] = (FLAGS.BIG_PAWN,)

            return out

        def rook():  # NOSONAR
            pass

        if PIECES[piece - 1] == PIECES.PAWN:
            return pawn()
        elif PIECES[piece - 1] == PIECES.ROOK:
            return rook()
        else:
            return {}
