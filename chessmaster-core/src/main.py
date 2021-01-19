#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

from utils import server_access, IMAGE_PATH, LOG_PATH, TEMP_PATH

if __name__ == '__main__':
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)

    server_access.init()

    # from utils.algebraic_notation import Board
    #
    # san = Board()
    # print(san._board)
    # print(san['e8'])
