#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np


class MovementHandler(object):

    def __init__(self, _id, _wsid, image):
        self._id = _id
        self._wsid = _wsid
        self.frame = np.array(image)

    def response(self):
        print(f'{self._id} - {self._wsid}:\n {self.frame}')

        return """
        {{
            "_id": "{0}",
            "_wsid": "{1}",
            "move": "Response to the move is this from model."
        }}
        """.format(self._id, self._wsid)
