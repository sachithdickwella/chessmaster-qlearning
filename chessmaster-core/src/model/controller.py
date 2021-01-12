# -*- encoding: utf-8 -*-

import cv2
from torch.utils.data import DataLoader
from torchvision import transforms as T

from util import IMAGE_PATH, LOGGER, IMAGE_SIZE
from . import BATCH_SIZE, NUM_WORKER, IS_CUDA, ToColor, ChessBoardDataset


class MovementHandler(object):

    def __init__(self, _id):
        self._id = _id
        self._wsid, self.result = None, None
        self.transform = T.Compose([
            ToColor(code=cv2.COLOR_RGBA2GRAY),
            T.Resize(IMAGE_SIZE),
            T.ToTensor()
        ])

    def accept(self, _wsid, images):
        # Save the second image of the incoming array which has the movement update.
        path = f'{IMAGE_PATH}/{self._id}.png'
        images[1].save(path)

        LOGGER.info(f'Incoming frame saved on {path} for later reference')

        self._wsid = _wsid
        self.result = self.model_invoke(images)

    def model_invoke(self, frames):
        dataset = ChessBoardDataset(frames, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=IS_CUDA)

        frame = next(iter(dataloader))  # NOSONAR TODO

        # frame1 = frame[0].view(512, -1) * 255  # NOSONAR
        # Image.fromarray(frame1.numpy().astype(np.uint8)).save('0.png')
        #
        # frame2 = frame[1].view(512, -1) * 255  # NOSONAR
        # Image.fromarray(frame2.numpy().astype(np.uint8)).save('1.png')

        return "nextMove"

    def response(self):
        return f"""{{ "_id": "{self._id}", "_wsid": "{self._wsid}", "move": "{self.result}" }}"""
