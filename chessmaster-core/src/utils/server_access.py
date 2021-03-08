# -*- coding: utf-8 -*-

import io
import socketserver
from multiprocessing import Manager, Process

from PIL import Image
from requests.exceptions import ConnectionError

from model.controller import MovementHandler
from . import LOGGER
from .api_access import RefreshSessions, NextMove

HOST, PORT, SESSIONS = ('localhost', 16375, dict())


class TCPRequestHandler(socketserver.StreamRequestHandler):
    _id_length, _wsid_length = (32, 8)
    splitter = 'splitter'

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
                if len(data) == 1:  # Check if the line count of the file is 1 or else pass.
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

        except (RuntimeError, ConnectionResetError, KeyError) as ex:
            LOGGER.error(f'Runtime error: {ex}')

    def receive_frames(self, data):
        """
        Receives the images/frames from this method. Just pass the entire list
        if the size is greater than 1. Usually around 194 lines.

        This incoming binary data includes the session id, WebSocket id and image
        data from the upstream Java program in binary format.

        :param data: item just received via the TCP socket as a list.
        """
        _id, _wsid, _status = (None, None, 0)
        iteration = 0
        buffer = bytearray()
        images, is_train = [], False

        for line in data:
            if iteration == 0:
                _id = line[:self._id_length].decode('utf-8')
                _wsid = line[self._id_length: self._id_length + self._wsid_length].decode('utf-8')
                _status = int(line[self._id_length + self._wsid_length: self._id_length + self._wsid_length + 1]
                              .decode('utf-8'))

                line = line[self._id_length + self._wsid_length + 1:]

            if self.splitter in str(line):
                b_splitter = bytearray(self.splitter.encode())
                idx = line.find(b_splitter)

                first = line[idx + len(b_splitter):]
                last = line[:idx]

                buffer.extend(last)
                images.append(Image.open(io.BytesIO(buffer)))

                buffer.clear()
                buffer.extend(first)

                iteration += 1
                continue

            buffer.extend(line)
            iteration += 1

        images.append(Image.open(io.BytesIO(buffer)))
        LOGGER.info(
            f'New frame received for the session id {_id} with the size of {len(buffer)} bytes, and return '
            f'websocket is "{_wsid}"')
        """
        Keep the :func:`~MovementHandler` object in the thread, in order 
        to receive the model response to move the chess pieces. Then write
        to the same socket as a response to this move and client take care 
        of the rest.
        
        Call the :func: 'accept()' to add the '_wsid' and 'image' to the model 
        on each invocation due to different frames.

        Note: '_wsid' doesn't involve on the model process, it just use to find 
        websocket destination where this frame originally came from. 
        """
        SESSIONS[_id].accept(_wsid, images, _status)
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
        SESSIONS[_id] = MovementHandler(_id)
        LOGGER.info(f'New session created with id {_id}')

    @staticmethod
    def invalidate_session(_id):
        """
        Delete the entire entry from SESSIONS dictionary variable from the '_id' and release
        the memory allocated.

        :param _id: Session id from the upstream Java Web program.
        """
        del SESSIONS[_id]
        LOGGER.info(f'Session invalidated for the id {_id}')

    @staticmethod
    def clean_sessions():
        """
        Clear out the SESSIONS instance from previously created session ids and 'MovementHandler'
        objects for save memory.
        """
        SESSIONS.clear()
        LOGGER.info(f'Cleared out all the sessions in the context')


def add_session(D, _id):  # NOSONAR
    """
    Add the parameter '_id' to the global variable 'SESSION' with
    new 'MovementHandler' instances.

    :param D: Dictionary object to insert the element of _id and new
    instance of 'MovementHandler'.
    :param _id: session id to insert into the 'SESSION' dictionary.
    """
    D[_id] = MovementHandler(_id)


def init():
    """
    Retrieve session ids from the front-end Java application and store on global 'SESSIONS'
    object with new 'MovementHandler' instance for each.
    """
    try:
        session_ids = RefreshSessions().retrieve()
        """
        Add 'session ids' from the 'session_ids' list to the global 'SESSIONS' instance as 
        a parallel instances.  
        """
        with Manager() as manager:
            D = manager.dict()  # NOSONAR

            ps = []
            for _id in session_ids:
                p = Process(target=add_session, args=(D, _id))
                p.start()

                ps.append(p)
            for p in ps:
                p.join()

            SESSIONS.update(D)
            del D  # Retain the memory by deleting old dictionary from 'manager'.
    except ConnectionError:
        LOGGER.warning("REST endpoints for SESSION info are not accessible. Proceed with initial SESSION values.")
    finally:
        LOGGER.info(
            f"Initial SESSION information are: {list(SESSIONS.keys()) if len(SESSIONS.items()) > 0 else 'EMPTY'}")
    """
    Initialize the 'server socket' to communicate with the Java client application.
    This program act as the server program hence, here resides the Q-Learning model.

    The server would continue to run until explicitly interrupted with Ctrl+C.

    For now, server host and port would be 'localhost' and ''16375' respectively and
    should be changed during the deployment.
    """
    with socketserver.TCPServer((HOST, PORT), TCPRequestHandler) as server:
        LOGGER.info(f'Server socket listener started at {HOST} on port {PORT}')
        server.serve_forever()
