#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np


class MovementHandler(object):

    def __init__(self, uuid, image):
        self.uuid = uuid
        self.frame = np.array(image)

    def response(self):
        print(f'{self.uuid}:\n {self.frame}')
        return """
        {{
            "uuid": "{0}",
            "move": "Response to the move is this from model."
        }}
        """.format(self.uuid)
