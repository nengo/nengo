import warnings
import weakref

from nengo.base import ObjView
from nengo.ensemble import Neurons
from nengo.exceptions import ValidationError
from nengo.params import (FrozenObject, NumberParam, Parameter, IntParam,
                          BoolParam)
from nengo.utils.compat import is_iterable, itervalues


class LearningRule(object):
    """An interface for making connections to a learning rule.

    Connections to a learning rule are to allow elements of the network to
    affect the learning rule. For example, learning rules that use error
    information can obtain that information through a connection.

    Learning rule objects should only ever be accessed through the
    ``learning_rule`` attribute of a connection.
    """

    def __init__(self, connection, learning_rule_type):
        self._connection = weakref.ref(connection)
        self.learning_rule_type = learning_rule_type

        if learning_rule_type.size_in is not None:
            self.size_in = learning_rule_type.size_in
        else:
            # infer size_in from post
            self.size_in = (self.connection.post_obj.ensemble.size_in
                            if isinstance(self.connection.post_obj, Neurons)
                            else self.connection.size_out)

    def __getitem__(self, key):
        return ObjView(self, key)

    def __repr__(self):
        return "<LearningRule at 0x%x modifying %r with type %r>" % (
            id(self), self.connection, self.learning_rule_type)

    def __str__(self):
        return "<LearningRule modifying %s with type %s>" % (
            self.connection, self.learning_rule_type)

    @property
    def connection(self):
        """(Connection) The connection modified by the learning rule."""
        return self._connection()

    @property
    def modifies(self):
        """(str) The variable modified by the learning rule."""
        return self.learning_rule_type.modifies

    @property
    def probeable(self):
        """(tuple) Signals that can be probed in the learning rule."""
        return self.learning_rule_type.probeable

    @property
    def size_out(self):
        """(int) Cannot connect from learning rules, so always 0."""
        return 0  # since a learning rule can't connect to anything
        # TODO: allow probing individual learning rules


class LearningRuleType(FrozenObject):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule_type`` keyword
    argument to the `~nengo.Connection` on which you want to do learning.

    Each learning rule exposes two important pieces of metadata that the
    builder uses to determine what information should be stored.

    The ``error_type`` is the type of the incoming error signal. Options are:

    * ``'none'``: no error signal
    * ``'scalar'``: scalar error signal
    * ``'decoded'``: vector error signal in decoded space
    * ``'neuron'``: vector error signal in neuron space

    The ``modifies`` attribute denotes the signal targeted by the rule.
    Options are:

    * ``'encoders'``
    * ``'decoders'``
    * ``'weights'``

    Parameters
    ----------
    learning_rate : float, optional (Default: 1e-6)
        A scalar indicating the rate at which ``modifies`` will be adjusted.

    Attributes
    ----------
    error_type : str
        The type of the incoming error signal. This also determines
        the dimensionality of the error signal.
    learning_rate : float
        A scalar indicating the rate at which ``modifies`` will be adjusted.
    modifies : str
        The signal targeted by the learning rule.
    """

    modifies = None
    probeable = ()

    learning_rate = NumberParam('learning_rate', low=0, low_open=True)
    size_in = IntParam('size_in', low=0, optional=True)

    def __init__(self, learning_rate=1e-6, size_in=None):
        super(LearningRuleType, self).__init__()
        self.learning_rate = learning_rate
        self.size_in = size_in

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ", ".join(self._argreprs))

    @property
    def _argreprs(self):
        return (["learning_rate=%g" % self.learning_rate]
                if self.learning_rate != 1e-6 else [])


class PES(LearningRuleType):
    """Prescribed Error Sensitivity learning rule.

    Modifies a connection's decoders to minimize an error signal provided
    through a connection to the connection's learning rule.

    Parameters
    ----------
    learning_rate : float, optional (Default: 1e-4)
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    """

    modifies = 'decoders'
    probeable = ('error', 'correction', 'activities', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)

    def __init__(self, learning_rate=1e-4, pre_tau=0.005):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        super(PES, self).__init__(learning_rate, size_in=None)

    @property
    def _argreprs(self):
        args = []
        if self.learning_rate != 1e-4:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        return args


class BCM(LearningRuleType):
    """Bienenstock-Cooper-Munroe learning rule.

    Modifies connection weights as a function of the presynaptic activity
    and the difference between the postsynaptic activity and the average
    postsynaptic activity.

    Parameters
    ----------
    theta_tau : float, optional (Default: 1.0)
        A scalar indicating the time constant for theta integration.
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional (Default: None)
        Filter constant on activities of neurons in post population.
        If None, post_tau will be the same as pre_tau.
    learning_rate : float, optional (Default: 1e-9)
        A scalar indicating the rate at which weights will be adjusted.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    theta_tau : float
        A scalar indicating the time constant for theta integration.
    """

    modifies = 'weights'
    probeable = ('theta', 'pre_filtered', 'post_filtered', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    theta_tau = NumberParam('theta_tau', low=0, low_open=True)

    def __init__(self, pre_tau=0.005, post_tau=None, theta_tau=1.0,
                 learning_rate=1e-9):
        self.theta_tau = theta_tau
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        super(BCM, self).__init__(learning_rate, size_in=0)

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
    """Oja learning rule.

    Modifies connection weights according to the Hebbian Oja rule, which
    augments typicaly Hebbian coactivity with a "forgetting" term that is
    proportional to the weight of the connection and the square of the
    postsynaptic activity.

    Parameters
    ----------
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional (Default: None)
        Filter constant on activities of neurons in post population.
        If None, post_tau will be the same as pre_tau.
    beta : float, optional (Default: 1.0)
        A scalar weight on the forgetting term.
    learning_rate : float, optional (Default: 1e-6)
        A scalar indicating the rate at which weights will be adjusted.

    Attributes
    ----------
    beta : float
        A scalar weight on the forgetting term.
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    """

    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    beta = NumberParam('beta', low=0)

    def __init__(self, pre_tau=0.005, post_tau=None, beta=1.0,
                 learning_rate=1e-6):
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        self.beta = beta
        super(Oja, self).__init__(learning_rate, size_in=0)

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
    """Vector Oja learning rule.

    Modifies an ensemble's encoders to be selective to its inputs.

    A connection to the learning rule will provide a scalar weight for the
    learning rate, minus 1. For instance, 0 is normal learning, -1 is no
    learning, and less than -1 causes anti-learning or "forgetting".

    Parameters
    ----------
    post_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in post population.
    learning_rate : float, optional (Default: 1e-2)
        A scalar indicating the rate at which encoders will be adjusted.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which encoders will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    modifies = 'encoders'
    probeable = ('post_filtered', 'scaled_encoders', 'delta')

    post_tau = NumberParam('post_tau', low=0, low_open=True, optional=True)

    def __init__(self, post_tau=0.005, learning_rate=1e-2):
        self.post_tau = post_tau
        super(Voja, self).__init__(learning_rate, size_in=1)


class GenericRule(LearningRuleType):
    """Learning rule implemented by generic Python function.

    Parameters
    ----------
    function : callable
        Accepts previous values of learned array and a data vector
        as input, outputs a delta for the learned array.

        Optionally the function may also accept a `params` argument, in which
        case `model.params` will be passed to the function as well (can be used
        to look up things like encoder values).
    learning_rate : float
        A scalar used to weight the output of the function
    size_in : int
        The dimensionality of the data vector input to the function
    modifies : 'weights' or 'decoders' or 'encoders'
        The signal targeted by the learning rule
    """

    probeable = ('target', 'data', 'delta')

    def __init__(self, function, learning_rate=1.0, size_in=0,
                 modifies="decoders", filters={}):
        self.modifies = modifies
        self.function = function
        self.filters = filters

        super(GenericRule, self).__init__(learning_rate=learning_rate,
                                          size_in=size_in)


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
        if rule.modifies not in ('encoders', 'decoders', 'weights'):
            raise ValidationError("Unrecognized target %r" % rule.modifies,
                                  attr=self.name, obj=instance)
