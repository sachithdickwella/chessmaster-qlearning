#!/usr/bin/env
# -*- encoding: utf-8 -*-

import numpy as np


class MovementHandler(object):

    def __init__(self, uuid, image):
        self.uuid = uuid
        self.frame = np.array(image)

    def response(self):
        print(self.frame)
        return "Response to the move!".encode('utf-8')  # To bytearray()
