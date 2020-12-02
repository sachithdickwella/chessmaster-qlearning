# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch.utils.data import Dataset

IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if IS_CUDA else 'cpu')

NUM_WORKER = 2
BATCH_SIZE = 2
# Discount factor when offering awards.
GAMMA = 0.999
# Epsilon - Determine how the model should take random action. This is the starting value for the
# epsilon (Rate of random action exploration).
EPS_START = 0.9
# Epsilon decay stop at this number.
EPS_END = 0.5
# Epsilon decay rate. Smaller this number, epsilon takes time to reach plateau and explore lots of
# random actions over time. Could you linear decay, squared or any type of decaying method.
EPS_DECAY = 200
# Parameter to tell how often should replace the target network.
TARGET_UPDATE = 10


# Custom dataset class to load image data to the model.
class ChessBoardDataset(Dataset):

    def __init__(self, images, transform):
        super().__init__()
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])


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
