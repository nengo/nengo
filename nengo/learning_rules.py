from nengo.config import Default, Parameter


class LearningRule(object):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a learning_rule keyword argument to
    the Connection that you want to do learning.

    Example:
        nengo.Connection(a, b, learning_rule=nengo.PES(error))

    Parameters
    ----------
    label : string, optional
        A name for the learning rule.

    Attributes
    ----------
    label : string
        Given label, or None.
    """
    learning_rate = Parameter(default=1e-5)
    label = Parameter(default=None)

    def __init__(self, learning_rate=Default, label=Default):
        self.learning_rate = learning_rate
        self.label = label

    def __str__(self):
        return "%s: %s" % (self.__class__.__name__, self.label)


class PES(LearningRule):
    """Prescribed Error Sensitivity Learning Rule

    Modifies a connection's decoders to minimize an error signal.

    Parameters
    ----------
    error : NengoObject
        The Node, Ensemble, or Neurons providing the error signal. Must be
        connectable to the post-synaptic object that is being used for this
        learning rule.
    synapse : float, optional
        Post-synaptic time constant (PSTC) to use for filtering on the
        modulatory error connection. Defaults to 0.005.
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    label : string, optional
        A name for the learning rule. Defaults to None.

    Attributes
    ----------
    label : string
        The given label.
    error : NengoObject
        The given error Node, Ensemble, or Neurons.
    learning_rate : float
        The given learning rate.
    error_connection : Connection
        The modulatory connection created to project the error signal.
    """

    def __init__(self, error_connection, learning_rate=1e-5, label=None):
        self.error_connection = error_connection
        super(PES, self).__init__(learning_rate, label)
