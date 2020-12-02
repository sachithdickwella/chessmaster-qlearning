# -*- encoding: utf-8 -*-

import logging.config
import sys
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

# Global variable to setup environment directories.
BUILD_PATH = '../build'
IMAGE_PATH = f'{BUILD_PATH}/out'
LOG_PATH = f'{BUILD_PATH}/log'
LOG_FILENAME = 'chessmaster-process.log'
TEMP_PATH = f'{BUILD_PATH}/tmp'

IMAGE_SIZE = (512, 512)

# Logging configuration of the program.
FORMAT = '%(asctime)s %(levelname)-8s  %(thread)d --- [%(threadName)s] %(filename)s : %(message)s'
LEVEL = logging.DEBUG
FORMATTER = logging.Formatter(FORMAT)

logging.basicConfig(format=FORMAT, level=LEVEL, stream=sys.stdout)

file = RotatingFileHandler(filename=f'{LOG_PATH}/{LOG_FILENAME}', maxBytes=10_485_760, backupCount=3, encoding='utf-8')
file.setLevel(LEVEL)
file.setFormatter(FORMATTER)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(file)


# Transformation class for the PIL images.
class ToColor(object):

    def __init__(self, code):
        """
        Constructor method for the 'ToColor' class to convert PIL.Image color image to different
        color channel image and return PIL.Image. Color channels should comply the OpenCV color
        codes.

        :param code: of the color need to change into. This only accepts codes support by OpenCV.
        """
        self.code = code

    def __call__(self, image):
        """
        Instance caller function to convert the color of the provided PIL.Images.

        :param image: PIL.Image to change the color channels.
        :return: the image as a PIL.Image.
        """
        if not type(image) is PngImageFile:
            raise TypeError(f"'image' should be a PIL.Image. Got {type(image)}")

        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, code=cv2.COLOR_RGBA2GRAY)

        return Image.fromarray(image)
