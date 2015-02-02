import warnings

from nengo.base import NengoObjectParam
from nengo.params import Parameter, NumberParam
from nengo.utils.compat import is_iterable, itervalues


class ConnectionParam(NengoObjectParam):
    def validate(self, instance, conn):
        from nengo.connection import Connection
        if not isinstance(conn, Connection):
            raise ValueError("'%s' is not a Connection" % conn)
        super(ConnectionParam, self).validate(instance, conn)


class LearningRuleType(object):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule`` keyword argument to
    the Connection on which you want to do learning.
    """

    learning_rate = NumberParam(low=0, low_open=True)
    error_type = 'none'
    probeable = []

    def __init__(self, learning_rate=1e-6):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.learning_rate = learning_rate

    def __repr__(self):
        return '<%s>' % self.__class__.__name__


class PES(LearningRuleType):
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

    error_type = 'decoder'
    modifies = ['Ensemble', 'Neurons']
    probeable = ['error', 'scaled_error', 'activities']

    def __init__(self, learning_rate=1e-6):
        super(PES, self).__init__(learning_rate)


class BCM(LearningRuleType):
    """Bienenstock-Cooper-Munroe learning rule

    Modifies connection weights.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    theta_tau : float, optional
        A scalar indicating the time constant for theta integration.
    pre_tau : float, optional
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    learning_rate : float
        The given learning rate.
    theta_tau : float
        A scalar indicating the time constant for theta integration.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    pre_tau = NumberParam(low=0, low_open=True)
    post_tau = NumberParam(low=0, low_open=True)
    theta_tau = NumberParam(low=0, low_open=True)
    error_type = 'none'
    modifies = ['Neurons']
    probeable = ['theta', 'pre_filtered', 'post_filtered']

    def __init__(self, pre_tau=0.005, post_tau=None, theta_tau=1.0,
                 learning_rate=1e-9):
        self.theta_tau = theta_tau
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        super(BCM, self).__init__(learning_rate)


class Oja(LearningRuleType):
    """Oja's learning rule

    Modifies connection weights.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    beta : float, optional
        A scalar governing the amount of forgetting. Larger => more forgetting.
    pre_tau : float, optional
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    learning_rate : float
        The given learning rate.
    beta : float
        A scalar governing the amount of forgetting. Larger => more forgetting.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    pre_tau = NumberParam(low=0, low_open=True)
    post_tau = NumberParam(low=0, low_open=True)
    beta = NumberParam(low=0)
    error_type = 'none'
    modifies = ['Neurons']
    probeable = ['pre_filtered', 'post_filtered']

    def __init__(self, pre_tau=0.005, post_tau=None, beta=1.0,
                 learning_rate=1e-6):
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        self.beta = beta
        super(Oja, self).__init__(learning_rate)


class LearningRuleTypeParam(Parameter):
    def validate(self, instance, rule):
        if is_iterable(rule):
            for r in (itervalues(rule) if isinstance(rule, dict) else rule):
                self.validate_rule(instance, r)
        elif rule is not None:
            self.validate_rule(instance, rule)
        super(LearningRuleTypeParam, self).validate(instance, rule)

    def validate_rule(self, instance, rule):
        if not isinstance(rule, LearningRuleType):
            raise ValueError("'%s' must be a learning rule type or a dict or "
                             "list of such types." % rule)
