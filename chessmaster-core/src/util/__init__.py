# -*- encoding: utf-8 -*-

import logging.config
import sys
from logging.handlers import RotatingFileHandler

BUILD_PATH = '../build'
IMAGE_PATH = f'{BUILD_PATH}/out'
LOG_PATH = f'{BUILD_PATH}/log'
LOG_FILENAME = 'chessmaster-process.log'
TEMP_PATH = f'{BUILD_PATH}/tmp'

FORMAT = '%(asctime)s %(levelname)-8s  %(thread)d --- [%(threadName)s] %(filename)s : %(message)s'
LEVEL = logging.DEBUG
FORMATTER = logging.Formatter(FORMAT)

logging.basicConfig(format=FORMAT, level=LEVEL, stream=sys.stdout)

file = RotatingFileHandler(filename=f'{LOG_PATH}/{LOG_FILENAME}', maxBytes=10240, backupCount=3, encoding='utf-8')
file.setLevel(LEVEL)
file.setFormatter(FORMATTER)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(file)
