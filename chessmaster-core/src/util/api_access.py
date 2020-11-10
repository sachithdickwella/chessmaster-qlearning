#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import requests


class NextMove(object):
    """
    This :func: NextMove class uses to invoke UI managed REST endpoint call
    to send the model response values to the UI, so the Java program can update
    the UI accordingly.
    """
    def __init__(self, movement):
        """
        :param movement is a function object that actually enforce the model
        to give a response for frame.
        """
        super().__init__()
        self.host = 'localhost'
        self.port = 8081
        self.url = f"http://{self.host}:{self.port}/movement/next"
        self.movement = movement

    def move(self):
        req = requests.post(self.url, data=self.movement(), headers={'Content-Type': 'application/json'})
        if req.status_code == 200:
            print("Response success with HTTP 200")
        else:
            print(f"Response failed with HTTP {req.status_code}")

