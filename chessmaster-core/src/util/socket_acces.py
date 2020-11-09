#!/usr/bin/env
# -*- encoding: utf-8 -*-

import io
import socketserver

from PIL import Image

from model.qlearning import MovementHandler

HOST, PORT = ('localhost', 16375)


class TCPRequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        """
        The request handler class for the server.

        It is instantiated once per connection to the server, and must
        override the handle() method to implement communication to the
        client.
        """
        try:
            """
            Capture the UUID from the client program to distinguish which UI
            instance send the request to the model. 

            Capture the image into a :func:`~bytearray()` hence, file receives
            as a stream of bytes. 
            
            Maximum file size would be 1MB and that validates at the client end
            as well as the here. If the below statement receives a byte stream 
            larger that 1MB, exceeding parts still would right to the file without 
            data corruption.
            """
            data = self.rfile.readlines()

            iteration, uuid, buffer = 0, None, bytearray()
            for line in data:
                if iteration == 0:
                    uuid = line[:36].decode('utf-8')
                    line = line[36:]

                buffer.extend(line)
                iteration += 1

            image = Image.open(io.BytesIO(buffer))
            """
            Keep the :func:`~MovementHandler` object in the thread, in order 
            to receive the model response to move the chess pieces. Then write
            to the same socket as a response to this move and client take care 
            of the rest.
            """
            movement = MovementHandler(uuid, image)
            """
            Respond to the movement came from the UI. With this response, chess
            board will be updated.
            """
            self.wfile.write(movement.response())

        except (RuntimeError, ConnectionResetError) as ex:
            print(f'Runtime error: {ex}')


def init():
    """
    Initialize the 'server socket' to communicate with the Java client application.
    This program act as the server program hence, here resides the Q-Learning model.

    The server would continue to run until explicitly interrupted with Ctrl+C.

    For now, server host and port would be 'localhost' and ''16375' respectively and
    should be changed during the deployment.
    """
    with socketserver.TCPServer((HOST, PORT), TCPRequestHandler) as server:
        server.serve_forever()
