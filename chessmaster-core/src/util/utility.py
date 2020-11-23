# -*- encoding: utf-8 -*-

import logging.handlers
import logging.config
import sys

from . import LOG_PATH, LOG_FILENAME

FORMAT = '%(asctime)s %(levelname)-8s  %(thread)d --- [%(threadName)s] %(filename)s : %(message)s'
LEVEL = logging.DEBUG
FORMATTER = logging.Formatter(FORMAT)

logging.basicConfig(format=FORMAT, level=LEVEL, stream=sys.stdout)

file = logging.handlers.RotatingFileHandler(filename=f'{LOG_PATH}/{LOG_FILENAME}',
                                            maxBytes=10240,
                                            backupCount=3,
                                            encoding='utf-8')
file.setLevel(LEVEL)
file.setFormatter(FORMATTER)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(file)
