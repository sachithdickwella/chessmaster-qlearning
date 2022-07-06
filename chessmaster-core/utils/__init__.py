# -*- coding: utf-8 -*-

import os
import logging.config
import sys
from logging.handlers import RotatingFileHandler

# Global variable to setup environment directories.
BUILD_PATH = '../build'
IMAGE_PATH = f'{BUILD_PATH}/out'
LOG_PATH = f'{BUILD_PATH}/log'
LOG_FILENAME = 'chessmaster-process.log'
TEMP_PATH = f'{BUILD_PATH}/tmp'

IMAGE_SIZE = (512, 512)

if not os.path.exists(BUILD_PATH):
    os.makedirs(BUILD_PATH)

if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

# Logging configuration of the program.
FORMAT = '%(asctime)s %(levelname)-8s  %(thread)-5d --- [%(threadName)s] %(filename)s : %(message)s'
LEVEL = logging.DEBUG
FORMATTER = logging.Formatter(FORMAT)

logging.basicConfig(format=FORMAT, level=LEVEL, stream=sys.stdout)

file = RotatingFileHandler(filename=f'{LOG_PATH}/{LOG_FILENAME}', maxBytes=10_485_760, backupCount=3, encoding='utf-8')
file.setLevel(LEVEL)
file.setFormatter(FORMATTER)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(file)
