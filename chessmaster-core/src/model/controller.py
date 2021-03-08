# -*- coding: utf-8 -*-

from threading import Thread

import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from utils import IMAGE_PATH, LOGGER, IMAGE_SIZE
from . import BATCH_SIZE, NUM_WORKER, IS_CUDA, ToColor, ChessBoardDataset

TEST_ENABLED = True


class MovementHandler(object):

    def __init__(self, _id):
        self._id = _id
        self._wsid, self.result = None, None
        self.transform = T.Compose([
            ToColor(code=cv2.COLOR_RGBA2GRAY),
            T.Resize(IMAGE_SIZE),
            T.ToTensor()
        ])

    def accept(self, _wsid, images, train=False):
        # Save the second image of the incoming array which has the movement update.
        path = f'{IMAGE_PATH}/{self._id}.png'
        images[1].save(path)

        LOGGER.info(f'Incoming frame saved on {path} for later reference')

        self._wsid = _wsid

        dataset = ChessBoardDataset(images, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=IS_CUDA)

        if train:
            self.result = self.train(dataloader)
        else:
            self.result = self.evaluate(dataloader)

    def evaluate(self, dataloader):  # NOSONAR
        LOGGER.info(f'Evaluation -> data loader {dataloader}')
        self.print_(dataloader)

        return "nextMove"

    def train(self, dataloader):  # NOSONAR
        LOGGER.info(f'Training -> data loader {dataloader}')
        self.print_(dataloader)

        return "nextMove"  # Return the next move inside the loop.

    def response(self):
        return f"""{{ "_id": "{self._id}", "_wsid": "{self._wsid}", "move": "{self.result}" }}"""

    def print_(self, dataloader):
        if TEST_ENABLED:
            batch = next(iter(dataloader))
            Thread(target=self.test_, args=(batch,)).start()

    @staticmethod
    def test_(frame):
        cv2.namedWindow('FirstFrame', flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('SecondFrame', flags=cv2.WINDOW_AUTOSIZE)

        frame1 = frame[0].view(512, -1) * 255
        frame2 = frame[1].view(512, -1) * 255

        frame1 = frame1.numpy().astype(np.uint8)
        frame2 = frame2.numpy().astype(np.uint8)

        LOGGER.info(f'Incoming frame demonstration on UI -> FRAME1 - {frame1.shape}, FRAME2 - {frame2.shape}')

        while True:

            cv2.imshow('FirstFrame', frame1)
            cv2.imshow('SecondFrame', frame2)

            if cv2.waitKey(0) & 0xff == 27:
                break

        cv2.destroyAllWindows()
