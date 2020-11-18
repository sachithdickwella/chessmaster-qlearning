#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

from model import controller
from util import socket_acces

if __name__ == '__main__':
    os.makedirs(controller.IMAGE_PATH, exist_ok=True)
    os.makedirs(controller.DUMP_PATH, exist_ok=True)

    socket_acces.init()
