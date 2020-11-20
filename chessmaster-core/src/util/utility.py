# -*- encoding: utf-8 -*-

import os
import yaml
import logging
import logging.config

from . import LOG_PATH, LOG_FILENAME


class Logger(logging.Logger):

    def __init__(self, name='default', level=logging.DEBUG):
        super().__init__(name, level)

        config = yaml.load('../../resources/logging_config.yml')
        config.setdefault()
        logging.config.dictConfig(config)
