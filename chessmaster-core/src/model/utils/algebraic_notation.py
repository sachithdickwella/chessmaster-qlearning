# Copyright 2020 Sachith Prasanna Dickwella (sachith_prasanna@live.com)
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
# -*- coding: utf-8 -*-

import re
from collections import namedtuple

import numpy as np

from utils.commons import PromotionInvalidException

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
PROMOTABLE = 'nbrq'


class Move(object):
    """
    Instances of this class hold the information of successful move of any pieces despite
    the color and location. These information would be required to determine move history,
    movement pattern of each user for whatever the downstream program.

    One :class:`Move` instance includes the details if;

    * color - which color has been moved.
    * _from - source square location of the move initiated from.
    * _to - destination square location of the move lasts.
    * piece - chess piece name which is moved.
    * san - standard algebraic notation for the move. May not use by the downstream program.
    * flag - if the something special happens besides a simple move, flag character for that
             move.
    * captured - flag if the movement captured a opponent's piece and that piece's name.

    This class instances generate by the :class:`Board` during the successful movement from a
    player despite the color.
    """

    def __init__(self, color, _from, _to, piece, san, flag, captured=None, promotion=None):
        """
        Initialize :class:`Move` instances with the default member values generated from the
        movement by :class:`Board`.

        :param color: which color this movement originated from.
        :param _from: source square location of the move initiated from.
        :param _to: destination/target square location which the piece successfully moved.
        :param piece: chess piece name which this move originated.
        :param san: standard algebraic notation for the moves.
        :param flag: if something significant happens during this movement.
        :param captured: flag if the movement captured a opponent's piece and that piece's name.
        :param promotion: flag if the pawn movement is a promotion and to what it is promoted to.
        """
        super(Move, self).__init__()

        self.color = color
        self.flag = flag
        self._from = _from
        self._to = _to
        self.piece = piece
        self.san = san
        self.captured = captured
        self.promotion = promotion

    def __str__(self):
        """
        Get the :class:`string` representation of the :class:`Move` class from :func:`self.dict`
        method return value.

        :return: the :class:`string` representation of the :class:`Move` as a dictionary format.
        """
        return str(self.dict())

    def dict(self):
        """
        Get the entire class structure including its members as :class:`dict` object.

        :return: the :class:`dict` instance with class members and their values.
        """
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
        if self.promotion is not None:
            details['promotion'] = self.promotion
        return details


class M(object):

    def __init__(self):
        super(M, self).__init__()


class Board(object):

    def __init__(self):
        super(Board, self).__init__()
        self.pawns_history = {}

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

    def update_board(self, _from, _to, flag, promotion):
        f_piece, _, f_loc = self.square(_from)
        _, _, t_loc = self.square(_to)

        if PIECES[f_piece - 1] == PIECES.PAWN:  # Strictly for PAWNs only.
            self.pawns_history[_from] = (_to, flag)

            if flag == FLAGS.EP_CAPTURE:
                idx = 1 if self._turn == PLAYERS_BITS.WHITE else -1
                self._board[t_loc[0] + idx, t_loc[1]] = self.c_board[t_loc[0] + idx, t_loc[1]] = 0
            elif flag == FLAGS.PROMOTION or flag == FLAGS.CAPTURE + FLAGS.PROMOTION:
                if not promotion:
                    raise PromotionInvalidException(f'Promotion value is not provided: \'{promotion}\'')
                elif promotion not in PROMOTABLE:
                    raise PromotionInvalidException(f'Provided promotion value is not valid: \'{promotion}\'')
                else:
                    self._board[f_loc] = self.pieces[promotion]

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
        Toggle the attribute::self._turn value depending on the current value. Nothing returns.
        """
        self._turn = PLAYERS_BITS.WHITE if self._turn == PLAYERS_BITS.BLACK else PLAYERS_BITS.BLACK

    def move(self, move, promotion=None):  # NOSONAR
        """
        Get the move details by passing the algebraic notation of the move and return
        'None' if the move is an illegal.

        :param move: algebraic notation of 2 squares that the piece should moved from
        and the destination (ex: b2-b4).
        :param promotion: pawn promoted piece value on pawn promotions.
        :return: an instance of :class:`Move` class with the details of source, target,
        color, flag, target SAN and piece if it's legal move. Otherwise 'None' returns.
        """

        def notation(_piece, _to, _flag):
            _piece = _piece.upper()

            if _flag == FLAGS.CAPTURE:
                return _piece + 'x' + _to
            elif _flag == FLAGS.EP_CAPTURE:
                return _piece + 'x' + _to + 'e.p'
            elif _flag == FLAGS.PROMOTION:
                return _to + '=' + promotion.upper()
            elif _flag == FLAGS.CAPTURE + FLAGS.PROMOTION:
                return _piece + 'x' + _to + '=' + promotion.upper()
            else:
                return _piece + _to

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
                self.update_board(_from, _to, moves[_to][0], promotion)
                self.toggle_player()

                piece = PIECES[piece - 1]
                p_color = PLAYERS[p_color - 1]

                moves = moves[_to]
                return Move(p_color, _from, _to, piece, notation(piece, _to, moves[0]), *moves, promotion)
            else:
                return None

    def generate_moves(self, _from):  # NOSONAR
        piece, color, loc = self.square(_from)
        if not piece:
            return None

        out = {}  # Structure -> {'to': ('flag', 'captured=n|c|b|e|p|k|q')}

        def common(fun):
            """
            Common method for 'rook', 'bishop' and 'queen' pieces' core algorithm hence
            there are no restriction to their movement across the board despite the squares
            of each pieces.

            :param fun: to be invoked for each of the piece with the parameters defined
                         below in this very same method.
            :return: the `out` dictionary from parent method of this method.
            """

            def pick(i, j):
                _to = self.square((i, j))

                if not _to.piece:
                    out[_to.location] = (FLAGS.NORMAL,)
                    return False
                elif _to.piece and color != _to.color:
                    out[_to.location] = (FLAGS.CAPTURE, PIECES[_to.piece - 1])
                    return True
                elif _to.piece and color == _to.color:
                    return True

            position, _max = [], max(np.concatenate([[8, 8] - np.array(loc), np.array(loc)]))
            return fun(pick, position, _max)

        def margins():
            """
            Common method to extract the margin of the PAWN and KING square movements hence
            they can be moved around the 8 squares around them basically.

            :return: the value of :attr:`min_r`, :attr:`max_r`, :attr:`min_c` and :attr:`max_c`.
            """
            min_r, max_r = (loc[0] - 1 if loc[0] > 0 else 0, loc[0] + 1 if loc[0] < 7 else 7)
            min_c, max_c = (loc[1] - 1 if loc[1] > 0 else 0, loc[1] + 1 if loc[1] < 7 else 7)
            return min_r, max_r, min_c, max_c

        def pawn():
            min_r, max_r, min_c, max_c = margins()

            def big_pawn():
                for rank in range(2):
                    if self._turn == color and color == PLAYERS_BITS.BLACK and loc[0] == 1:
                        __to = self.square((loc[0] + rank + 1, loc[1]))
                        if not rank and __to.piece:
                            break
                        elif rank and not __to.piece and _from not in self.pawns_history:
                            out[__to.location] = (FLAGS.BIG_PAWN,)

                    elif self._turn == color and color == PLAYERS_BITS.WHITE and loc[0] == 6:
                        __to = self.square((loc[0] - rank - 1, loc[1]))
                        if not rank and __to.piece:
                            break
                        elif rank and not __to.piece and _from not in self.pawns_history:
                            out[__to.location] = (FLAGS.BIG_PAWN,)

            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if loc != (i, j):
                        _to = self.square((i, j))
                        if _to.piece and _to.color != color \
                                and ((i > loc[0] and color == PLAYERS_BITS.BLACK)
                                     or (i < loc[0] and color == PLAYERS_BITS.WHITE)) \
                                and (j == loc[1] + 1 or j == loc[1] - 1):
                            if (color == PLAYERS_BITS.BLACK and i == 7) or (color == PLAYERS_BITS.WHITE and i == 0):
                                out[_to.location] = (FLAGS.CAPTURE + FLAGS.PROMOTION, PIECES[_to.piece - 1])
                            else:
                                out[_to.location] = (FLAGS.CAPTURE, PIECES[_to.piece - 1])

                        elif not _to.piece and j == loc[1] \
                                and ((i == loc[0] + 1 and color == PLAYERS_BITS.BLACK)
                                     or (i == loc[0] - 1 and color == PLAYERS_BITS.WHITE)):
                            if (color == PLAYERS_BITS.BLACK and i == 7) or (color == PLAYERS_BITS.WHITE and i == 0):
                                out[_to.location] = (FLAGS.PROMOTION,)
                            else:
                                out[_to.location] = (FLAGS.NORMAL,)

                        elif not _to.piece and j != loc[1] \
                                and ((color == PLAYERS_BITS.BLACK and loc[0] == 4 and i == loc[0] + 1)
                                     or (color == PLAYERS_BITS.WHITE and loc[0] == 3) and i == loc[0] - 1):

                            enp = self.square((i - 1, j)) if color == PLAYERS_BITS.BLACK \
                                else self.square((i + 1, j))

                            if enp.piece and enp.color != color \
                                    and next(f for _, (t, f) in self.pawns_history.items()
                                             if t == enp.location) == FLAGS.BIG_PAWN \
                                    and len(self.pawns_history) == [lo for lo, _ in self.pawns_history.values()] \
                                    .index(enp.location) + 1:
                                out[_to.location] = (FLAGS.EP_CAPTURE, PIECES[enp.piece - 1])
            big_pawn()
            return out

        def knight():
            def pick(_to):
                if not _to.piece:
                    out[_to.location] = (FLAGS.NORMAL,)
                elif _to.piece and _to.color != color:
                    out[_to.location] = (FLAGS.CAPTURE, PIECES[_to.piece - 1])

            for i in [2, -2]:
                for j in [1, -1]:
                    if (0 <= loc[0] + i <= 7) and (0 <= loc[1] + j <= 7):
                        pick(self.square((loc[0] + i, loc[1] + j)))

                    if (0 <= loc[0] + j <= 7) and (0 <= loc[1] + i <= 7):
                        pick(self.square((loc[0] + j, loc[1] + i)))
            return out

        def rook(pick, pos, _max):
            for x in range(_max):
                if 0 <= loc[0] + x + 1 < 8 and 'd' not in pos \
                        and pick(loc[0] + x + 1, loc[1]):
                    pos.append('d')
                if 0 <= loc[0] - x - 1 < 8 and 'u' not in pos \
                        and pick(loc[0] - x - 1, loc[1]):
                    pos.append('u')
                if 0 <= loc[1] + x + 1 < 8 and 'r' not in pos \
                        and pick(loc[0], loc[1] + x + 1):
                    pos.append('r')
                if 0 <= loc[1] - x - 1 < 8 and 'l' not in pos \
                        and pick(loc[0], loc[1] - x - 1):
                    pos.append('l')
            return out

        def bishop(pick, pos, _max):
            for x, y in zip(range(_max), range(_max)):
                if (0 <= loc[0] + x + 1 < 8 and 0 <= loc[1] + y + 1 < 8) \
                        and 'dr' not in pos and pick(loc[0] + x + 1, loc[1] + y + 1):
                    pos.append('dr')
                if (0 <= loc[0] - x - 1 < 8 and 0 <= loc[1] + y + 1 < 8) \
                        and 'ur' not in pos and pick(loc[0] - x - 1, loc[1] + y + 1):
                    pos.append('ur')
                if (0 <= loc[0] + x + 1 < 8 and 0 <= loc[1] - y - 1 < 8) \
                        and 'dl' not in pos and pick(loc[0] + x + 1, loc[1] - y - 1):
                    pos.append('dl')
                if (0 <= loc[0] - x - 1 < 8 and 0 <= loc[1] - y - 1 < 8) \
                        and 'ul' not in pos and pick(loc[0] - x - 1, loc[1] - y - 1):
                    pos.append('ul')
            return out

        def queen(pick, pos, _max):
            rook(pick, pos, _max)
            bishop(pick, pos, _max)
            return out

        def king():
            min_r, max_r, min_c, max_c = margins()

            for i in range(min_r, max_r + 1):
                for j in range(min_c, max_c + 1):
                    if loc != (i, j):
                        _to = self.square((i, j))
                        if not _to.piece:
                            out[_to.location] = (FLAGS.NORMAL,)
                        if _to.piece and color != _to.color:
                            out[_to.location] = (FLAGS.CAPTURE, PIECES[_to.piece - 1])
            return out

        def castling():  # NOSONAR
            pass

        def check():  # NOSONAR
            pass

        def checkmate():  # NOSONAR
            pass

        switch = {
            PIECES.PAWN: pawn,
            PIECES.KNIGHT: knight,
            PIECES.ROOK: rook,
            PIECES.BISHOP: bishop,
            PIECES.QUEEN: queen,
            PIECES.KING: king
        }

        piece -= 1
        if PIECES[piece] == PIECES.ROOK \
                or PIECES[piece] == PIECES.BISHOP \
                or PIECES[piece] == PIECES.QUEEN:
            return common(switch.get(PIECES[piece]))
        else:
            return switch.get(PIECES[piece], None)()
