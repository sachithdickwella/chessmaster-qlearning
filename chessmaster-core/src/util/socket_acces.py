#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import io
import socketserver

from PIL import Image

from model.qlearning import MovementHandler
from .api_access import NextMove

HOST, PORT, SESSIONS = ('localhost', 16375, dict())


class TCPRequestHandler(socketserver.StreamRequestHandler):
    _id_length, _wsid_length = (32, 8)

    def handle(self):
        """
        The request handler class for the server.

        It is instantiated once per connection to the server, and must
        override the handle() method to implement communication to the
        client.
        """
        try:
            """
            Capture the ID from the client program to distinguish which UI
            instance send the request to the model. 

            Capture the image into a :func:`~bytearray()` hence, file receives
            as a stream of bytes. 
            
            Maximum file size would be 1MB and that validates at the client end
            as well as the here. If the below statement receives a byte stream 
            larger that 1MB, exceeding parts still would right to the file without 
            data corruption.
            """
            data = self.rfile.readlines()

            if len(data) > 0:
                if len(data) == 1:
                    data = data[0]
                    if len(data) > self._id_length:
                        _id = data[:self._id_length].decode('utf-8')
                        command = data[self._id_length:].decode('utf-8')

                        if command == 'create':
                            self.create_session(_id)
                        else:
                            self.invalidate_session(_id)
                    else:
                        self.clean_sessions()
                else:
                    self.receive_frames(data)

        except (RuntimeError, ConnectionResetError) as ex:
            print(f'Runtime error: {ex}')

        print(SESSIONS)

    def receive_frames(self, data):
        """
        Receives the images/frames from this method. Just pass the entire list
        if the size is greater than 1. Usually around 194 lines.

        This incoming binary data includes the session id, WebSocket id and image
        data from the upstream Java program in binary format.

        :param data: item just received via the TCP socket as a list.
        """
        iteration, _id, _wsid, buffer = (0, None, 0, bytearray())
        for line in data:
            if iteration == 0:
                _id = line[:self._id_length].decode('utf-8')
                _wsid = line[self._id_length: self._id_length + self._wsid_length].decode('utf-8')

                line = line[self._id_length + self._wsid_length:]

            buffer.extend(line)
            iteration += 1

        image = Image.open(io.BytesIO(buffer))
        """
        Keep the :func:`~MovementHandler` object in the thread, in order 
        to receive the model response to move the chess pieces. Then write
        to the same socket as a response to this move and client take care 
        of the rest.
        """
        if SESSIONS.get(_id) is None:
            SESSIONS[_id] = MovementHandler(_id)
        """
        Call the :func: 'accept()' to add the '_wsid' and 'image' to the model 
        on each invocation due to different frames.

        Note: '_wsid' doesn't involve on the model process, it just use to find 
        websocket destination where this frame originally came from. 
        """
        SESSIONS[_id].accept(_wsid, image)
        """
        Respond to the movement came from the UI. With this response, chess
        board will be updated.
        """
        NextMove(SESSIONS[_id].response).send()

    @staticmethod
    def create_session(_id):
        """
        Add new element to the static SESSIONS variable with '_id' as the key and value
        with None until a request received to create a new 'MovementHandler' instance and
        replace that 'None' value.

        :param _id: Session id from the upstream Java Web program.
        """
        SESSIONS[_id] = None

    @staticmethod
    def invalidate_session(_id):
        """
        Delete the entire entry from SESSIONS dictionary variable from the '_id' and release
        the memory allocated.

        :param _id: Session id from the upstream Java Web program.
        """
        try:
            del SESSIONS[_id]
        except KeyError as ex:
            print(f'Runtime error: {ex}')

    @staticmethod
    def clean_sessions():
        """
        Clear out the SESSIONS instance from previously created session ids and 'MovementHandler'
        objects for save memory.
        """
        SESSIONS.clear()


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
