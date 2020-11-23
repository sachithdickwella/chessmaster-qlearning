#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

from util import server_acces, IMAGE_PATH, LOG_PATH

if __name__ == '__main__':
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    server_acces.init()
