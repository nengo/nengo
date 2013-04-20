import copy
import numpy as np  # Should we do this without numpy?

class LearningRule(object):
    """A learning rule changes a connection as the model runs.

    A learning rule can operate on decoders, weights, or both.

    Instance variables:

    learning_rate: The learning rate of the rule.
    connection: The connection that this learning rule changes.

    """

    def __init__(self, learning_rate):
        """Creates a learning rule.

        :param float learning_rate: How fast learning should occur.
        """
        self.learning_rate = learning_rate
        self._connection = None

    @property
    def connection(self):
        """The connection that this learning rule changes."""
        return self._connection

    @connection.setter
    def connection(self, connection):
        """Set the connection that this learning rule changes."""
        self._connection = connection
        self.original_decoders = copy.deepcopy(connection.pre.decoders)
        self.original_weights = copy.deepcopy(connection.weights)

    def update_decoders(self, dt):
        """A function that updates the decoders associated with a connection.

        Learning rules that implement this method affect vector-level
        connections. These are the connections that are created with a
        ``model.connect(...)`` call.

        """
        raise NotImplementedError

    def update_weights(self, dt):
        """A function that updates the weights associated with a connection.

        Learning rules that implement this method affect neuron-level
        connections. These are the connections that are created with a
        ``model.connect_neurons(...)`` call.

        """
        raise NotImplementedError

    def reset(self):
        """Resets the decoders and weights to their original values
        (before any learning occurred).

        """
        self.connection.pre.decoders = self.original_decoders
        self.connection.weights = self.original_weights

        
class HPESLearningRule(LearningRule):
    SCALING_FACTOR = 10

    def __init__(self, error, supervision_ratio=0.8, learning_rate=5e-7):
        self.error = error
        self.supervision_ratio = supervision_ratio
        self.pre_filter = PSCFilter(tau=0.02)
        self.post_filter = PSCFilter(tau=0.02)
        self.error_filter = PSCFilter(tau=0.02)
        self.theta_filter = PSCFilter(tau=200.0)
        LearningRule.__init__(self, learning_rate)

    @connection.setter
    def connection(self, connection):
        LearningRule.connection(self, connection)
        self.pre = self.connection.pre.neurons.output
        self.post = self.connection.post.neurons.output
        self.encoders = self.connection.post.encoders
        self.gains = self.connection.post.gains
        self.theta = np.random.uniform(low=5e-5, high=15e-5)
        self.initial_theta = copy.deepcopy(self.theta)
        # self.scaled_encoders = self.encoders * self.gains * self.learning_rate

    def update_decoders(self, dt):
        pass

    def update_weights(self, dt):
        # Should these be self.x.filtered_output?
        filtered_pre = self.pre_filter(self.pre.neuron_output, dt)
        filtered_post = self.post_filter(self.post.neuron_output, dt)
        self.theta = self.theta_filter(filtered_post)
        filtered_error = self.error_filter(self.error.decoded_output, dt)
        encoded_error = np.sum(self.encoders * np.reshape(
                filtered_error, self.post.dimensions))
        delta_supervised = filtered_pre * encoded_error * self.learning_rate
        delta_unsupervised = (self.learning_rate * self.SCALING_FACTOR
                              * self.filtered_post
                              * (self.filtered_post - self.theta)
                              * self.gains)
        self.connection.weights += self.supervision_ratio * delta_supervised
        self.connection.weights += ((1. - self.supervision_ratio)
                                    * delta_unsupervised)

