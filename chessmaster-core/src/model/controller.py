# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from torchvision import transforms as T

from util import IMAGE_PATH
from util.utility import LOGGER


class MovementHandler(object):

    def __init__(self, _id):
        self._id = _id
        self._wsid, self.result = None, None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def accept(self, _wsid, image):
        path = f'{IMAGE_PATH}/{self._id}.png'
        image.save(path)

        LOGGER.info(f'Incoming frame saved on {path} for later reference')

        self._wsid = _wsid
        self.result = self.model_invoke(image)

    def model_invoke(self, frame):
        frame = np.array(frame, dtype=np.uint8)
        frame = cv2.cvtColor(frame, code=cv2.COLOR_RGBA2RGB)

        frame = self.transform(frame)  # TODO

        return "nextMove"

    def response(self):
        return """
        {{
            "_id": "{0}",
            "_wsid": "{1}",
            "move": "{2}"
        }}
        """.format(self._id, self._wsid, self.result)
