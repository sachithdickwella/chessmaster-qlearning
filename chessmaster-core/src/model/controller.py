# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from torchvision import transforms as T

from util import IMAGE_PATH, LOGGER, IMAGE_SIZE


class MovementHandler(object):

    def __init__(self, _id):
        self._id = _id
        self._wsid, self.result = None, None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor()
        ])

    def accept(self, _wsid, image):
        path = f'{IMAGE_PATH}/{self._id}.png'
        image.save(path)

        LOGGER.info(f'Incoming frame saved on {path} for later reference')

        self._wsid = _wsid
        self.result = self.model_invoke(image)

    def model_invoke(self, frame):
        frame = np.array(frame, dtype=np.uint8)
        frame = cv2.cvtColor(frame, code=cv2.COLOR_RGBA2GRAY)

        frame = self.transform(frame)  # TODO

        # frame = frame.view(512, -1) * 255 # NOSONAR
        # Image.fromarray(frame.numpy().astype(np.uint8)).save('test.png')

        return "nextMove"

    def response(self):
        return f"""
        {{
            "_id": "{self._id}",
            "_wsid": "{self._wsid}",
            "move": "{self.result}"
        }}
        """
