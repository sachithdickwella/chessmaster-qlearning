# -*- encoding: utf-8 -*-

import requests

HOST, PORT = 'localhost', 8081


class RefreshSessions(object):
    """
    Objective of the class is to fetch the active session information from the
    UI program and update the application global 'SESSIONS' dictionary with the
    details. So, the program would be able to keep the track with Java program
    with the session details.
    """
    def __init__(self):
        """
        Initialize the class instance with the endpoint URL, so terminal operation
        could execute successfully.
        """
        super().__init__()
        self.url = f"http://{HOST}:{PORT}/session/ids"

    def retrieve(self):
        """
        Invoke the REST endpoint of the UI end and fetch all the active session
        ids and return as a list.

        :return: the JSON response as a list() object of string session id.
        """
        res = requests.get(self.url, headers={'Accept': 'application/json'})
        return res.json()


class NextMove(object):
    """
    This NextMove class uses to invoke UI managed REST endpoint call to send
    the model response values to the UI, so the Java program can update the UI
    accordingly. This is a stateless class.
    """
    def __init__(self, movement):
        """
        Initialize the class instance with the endpoint URL, so terminal operation
        could execute successfully.

        :param movement is a function object that actually enforce the model
        to give a response for the frame from the model.
        """
        super().__init__()
        self.url = f"http://{HOST}:{PORT}/movement/next"
        self.movement = movement

    def send(self):
        """
        Invoke the REST endpoint to send the details of next chess piece move which
        recognized by the the
        """
        res = requests.post(self.url, data=self.movement(), headers={'Content-Type': 'application/json'})
        print(f'HTTP Response Code {res.status_code}')
