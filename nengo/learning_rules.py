from nengo.synapses import Lowpass, SynapseParam


class LearningRule(object):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule`` keyword argument to
    the Connection on which you want to do learning.
    """

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def __str__(self):
        return self.__class__.__name__


class PES(LearningRule):
    """Prescribed Error Sensitivity Learning Rule

    Modifies a connection's decoders to minimize an error signal.

    Parameters
    ----------
    error : NengoObject
        The Node, Ensemble, or Neurons providing the error signal. Must be
        connectable to the post-synaptic object that is being used for this
        learning rule.
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.

    Attributes
    ----------
    learning_rate : float
        The given learning rate.
    error_connection : Connection
        The modulatory connection created to project the error signal.
    """

    modifies = ['Ensemble', 'Neurons']

    def __init__(self, error_connection, learning_rate=1.0):
        self.error_connection = error_connection
        super(PES, self).__init__(learning_rate)


class BCM(LearningRule):
    """Bienenstock-Cooper-Munroe learning rule

    Modifies connection weights.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    theta_synapse : float, optional
        A scalar indicating the time constant for theta integration.
    pre_synapse : float, optional
        Filter constant on activities of neurons in pre population.
    post_synapse : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    TODO
    """

    modifies = ['Neurons']
    pre_synapse = SynapseParam(default=Lowpass(0.005))
    post_synapse = SynapseParam(default=Lowpass(0.005))
    theta_synapse = SynapseParam(default=Lowpass(100))

    def __init__(self, pre_synapse=0.005, post_synapse=0.005,
                 theta_synapse=100, learning_rate=1.0):
        self.theta_synapse = theta_synapse
        self.pre_synapse = pre_synapse
        self.post_synapse = post_synapse
        super(BCM, self).__init__(learning_rate)


class Oja(LearningRule):
    """Oja's learning rule

    Modifies connection weights.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    beta : float, optional
        A scalar governing the amount of forgetting. Larger => more forgetting.
    pre_synapse : float, optional
        Filter constant on activities of neurons in pre population.
    post_synapse : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    TODO
    """

    modifies = ['Neurons']
    pre_synapse = SynapseParam(default=Lowpass(0.005))
    post_synapse = SynapseParam(default=Lowpass(0.005))

    def __init__(self, pre_synapse=0.005, post_synapse=0.005,
                 beta=1.0, learning_rate=1.0):
        self.pre_synapse = pre_synapse
        self.post_synapse = post_synapse
        self.beta = beta
        super(Oja, self).__init__(learning_rate)
