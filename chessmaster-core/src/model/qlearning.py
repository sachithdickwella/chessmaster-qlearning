#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np


class MovementHandler(object):

    def __init__(self, _id, image):
        self._id = _id
        self.frame = np.array(image)

    def response(self):
        print(f'{self._id}:\n {self.frame}')
        return """
        {{
            "_id": "{0}",
            "move": "Response to the move is this from model."
        }}
        """.format(self._id)
