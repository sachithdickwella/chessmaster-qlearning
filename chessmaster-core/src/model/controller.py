# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from torchvision import transforms as T

BUILD_PATH = '../build'
IMAGE_PATH = f'{BUILD_PATH}/out'
LOG_PATH = f'{BUILD_PATH}/log'


class MovementHandler(object):

    def __init__(self, _id):
        self._id = _id
        self._wsid, self.result = None, None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def accept(self, _wsid, image):
        image.save(f'{IMAGE_PATH}/{self._id}.png')

        self._wsid = _wsid
        self.result = self.model_invoke(image)

    def model_invoke(self, frame):
        frame = np.array(frame, dtype=np.uint8)
        frame = cv2.cvtColor(frame, code=cv2.COLOR_RGBA2RGB)

        frame = self.transform(frame)

        return "nextMove"

    def response(self):
        return """
        {{
            "_id": "{0}",
            "_wsid": "{1}",
            "move": "{2}"
        }}
        """.format(self._id, self._wsid, self.result)
