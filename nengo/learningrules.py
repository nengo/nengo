import nengo
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
    def __init__(self, connection, error, base_learning_rate=1.0):
        # -- N.B. some of these are properties
        self.connection = connection
        self.error = error
        self.base_learning_rate = base_learning_rate
        self.learning_rate = nengo.builder.IVector(dimensions=1,
                                                   label='lr')

        self.error_connection = Connection(
            self.error,
            self.connection.post,
            modulatory=True)

        nengo.context.add_to_current(self)

    def add_to_model(self, model):
        model.objs.append(self)
        model.rules.append(self)
        return self

# -- make flake-8 happy
