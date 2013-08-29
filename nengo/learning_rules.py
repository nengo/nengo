import nengo
from nengo.config import Default, Parameter


class LearningRule(object):
    learning_rate = Parameter(default=1e-5)

    def __init__(self, connection, learning_rate=Default):
        self.connection = connection
        self.learning_rate = learning_rate

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        if self._connection is not None:
            raise ValueError("Connection is already set and cannot be changed")
        self._connection = connection

    def __repr__(self):
        return self.__class__.__name__


class PES(LearningRule):
    def __init__(self, error, learning_rate=1.0):
        self.error = error
        self.learning_rate = learning_rate

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        if self._connection is not None:
            raise ValueError("Connection is already set and cannot be changed")
        self.error_connection = nengo.Connection(
            self.error, connection.post, modulatory=True)
        self._connection = connection
