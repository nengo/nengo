import warnings

from nengo.base import NengoObjectParam
from nengo.exceptions import ValidationError
from nengo.params import FrozenObject, NumberParam, Parameter
from nengo.utils.compat import is_iterable, itervalues


class ConnectionParam(NengoObjectParam):
    def validate(self, instance, conn):
        from nengo.connection import Connection
        if not isinstance(conn, Connection):
            raise ValidationError("'%s' is not a Connection" % conn,
                                  attr=self.name, obj=instance)
        super(ConnectionParam, self).validate(instance, conn)


class LearningRuleType(FrozenObject):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule`` keyword argument to
    the Connection on which you want to do learning.

    Attributes
    ----------
    error_type : str
        The type (which determines the dimensionality) of the incoming error
        signal. Options are 'none': no error signal; 'scalar': scalar error
        signal; 'decoded': vector error signal in decoded space;
        'neuron': vector error signal in neuron space.
    modifies : str
        The signal targeted by the learning rule. Options are 'encoders',
        'decoders' (will also be adapted to modify a full weight matrix by
        multiplying by the post population encoders), or 'weights' (only works
        on full weight matrices).
    """

    learning_rate = NumberParam('learning_rate', low=0, low_open=True)

    error_type = 'none'
    modifies = None
    probeable = ()

    def __init__(self, learning_rate=1e-6):
        super(LearningRuleType, self).__init__()
        self.learning_rate = learning_rate

    @property
    def _argreprs(self):
        return (["learning_rate=%g" % self.learning_rate]
                if self.learning_rate != 1e-6 else [])

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ", ".join(self._argreprs))


class PES(LearningRuleType):
    """Prescribed Error Sensitivity Learning Rule

    Modifies a connection's decoders to minimize an error signal.

    Parameters
    ----------
    pre_tau : float, optional
        Filter constant on activities of neurons in pre population.
        Defaults to 0.005.
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.

    Attributes
    ----------
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    learning_rate : float
        The given learning rate.
    error_connection : Connection
        The modulatory connection created to project the error signal.
    """

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)

    error_type = 'decoded'
    modifies = 'decoders'
    probeable = ('error', 'correction', 'activities', 'delta')

    def __init__(self, learning_rate=1e-4, pre_tau=0.005):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        super(PES, self).__init__(learning_rate)

    @property
    def _argreprs(self):
        args = []
        if self.learning_rate != 1e-4:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        return args


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

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    theta_tau = NumberParam('theta_tau', low=0, low_open=True)

    error_type = 'none'
    modifies = 'weights'
    probeable = ('theta', 'pre_filtered', 'post_filtered', 'delta')

    def __init__(self, pre_tau=0.005, post_tau=None, theta_tau=1.0,
                 learning_rate=1e-9):
        self.theta_tau = theta_tau
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        super(BCM, self).__init__(learning_rate)

    @property
    def _argreprs(self):
        args = []
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        if self.post_tau != self.pre_tau:
            args.append("post_tau=%f" % self.post_tau)
        if self.theta_tau != 1.0:
            args.append("theta_tau=%f" % self.theta_tau)
        if self.learning_rate != 1e-9:
            args.append("learning_rate=%g" % self.learning_rate)
        return args


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

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    beta = NumberParam('beta', low=0)

    error_type = 'none'
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    def __init__(self, pre_tau=0.005, post_tau=None, beta=1.0,
                 learning_rate=1e-6):
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        self.beta = beta
        super(Oja, self).__init__(learning_rate)

    @property
    def _argreprs(self):
        args = []
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        if self.post_tau != self.pre_tau:
            args.append("post_tau=%f" % self.post_tau)
        if self.beta != 1.0:
            args.append("beta=%f" % self.beta)
        if self.learning_rate != 1e-6:
            args.append("learning_rate=%g" % self.learning_rate)
        return args


class Voja(LearningRuleType):
    """Vector Oja's learning rule.

    Modifies an ensemble's encoders to be selective to its inputs.

    A connection to the learning rule will provide a scalar weight for the
    learning rate (minus 1). For instance, 0 is normal learning, -1 is no
    learning, and less than -1 causes anti-learning or "forgetting".

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which encoders will be adjusted.
        Defaults to 1e-2.
    post_tau : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    learning_rate : float
        The given learning rate.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    post_tau = NumberParam('post_tau', low=0, low_open=True, optional=True)

    error_type = 'scalar'
    modifies = 'encoders'
    probeable = ('post_filtered', 'scaled_encoders', 'delta')

    def __init__(self, post_tau=0.005, learning_rate=1e-2):
        self.post_tau = post_tau
        super(Voja, self).__init__(learning_rate)


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
            raise ValidationError(
                "'%s' must be a learning rule type or a dict or "
                "list of such types." % rule, attr=self.name, obj=instance)
        if rule.error_type not in ('none', 'scalar', 'decoded', 'neuron'):
            raise ValidationError(
                "Unrecognized error type %r" % rule.error_type,
                attr=self.name, obj=instance)
        if rule.modifies not in ('encoders', 'decoders', 'weights'):
            raise ValidationError("Unrecognized target %r" % rule.modifies,
                                  attr=self.name, obj=instance)
