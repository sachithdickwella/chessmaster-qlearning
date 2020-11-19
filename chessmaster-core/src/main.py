#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

from model import controller
from util import server_acces

if __name__ == '__main__':
    os.makedirs(controller.IMAGE_PATH, exist_ok=True)
    os.makedirs(controller.LOG_PATH, exist_ok=True)

    server_acces.init()
