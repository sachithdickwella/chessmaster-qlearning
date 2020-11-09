#!/usr/bin/env
# -*- encoding: utf-8 -*-

import numpy as np


class MovementHandler(object):

    def __init__(self, uuid, image):
        self.uuid = uuid
        self.frame = np.array(image)

    def response(self):
        print(f'{self.uuid}:\n {self.frame}')
        return "{}: Response to the move!".format(self.uuid).encode('utf-8')  # To bytearray()
