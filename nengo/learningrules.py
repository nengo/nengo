from .objects import Connection


class LearningRule(object):
    _learning_rate = 1e-5
    _connection = None

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        if self._connection not in (None, connection):
            raise ValueError(
                "Connection is already set and cannot be changed.")
        self._connection = connection

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


class PES_Rule(LearningRule):
    def __init__(self, connection, error, learning_rate=1.0):
        # -- N.B. some of these are properties
        self.connection = connection
        self.error = error
        self.learning_rate = learning_rate

        # -- little reverse-lookup hacking for builder
        connection.learning_rule = self

        self.error_connection = Connection(
            self.error,
            self.connection.post,
            modulatory=True)

# -- make flake-8 happy
