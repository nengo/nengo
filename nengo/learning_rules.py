import warnings

from nengo.config import SupportDefaultsMixin
from nengo.exceptions import ValidationError
from nengo.params import (
    Default,
    IntParam,
    FrozenObject,
    NumberParam,
    Parameter,
)
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.numpy import is_iterable


class LearningRuleTypeSizeInParam(IntParam):
    valid_strings = ("pre", "post", "mid", "pre_state", "post_state")

    def coerce(self, instance, size_in):
        if isinstance(size_in, str):
            if size_in not in self.valid_strings:
                raise ValidationError(
                    "%r is not a valid string value (must be one of %s)"
                    % (size_in, self.strings),
                    attr=self.name,
                    obj=instance,
                )
            return size_in
        else:
            return super().coerce(instance, size_in)  # IntParam validation


class LearningRuleType(FrozenObject, SupportDefaultsMixin):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule_type`` keyword
    argument to the `~nengo.Connection` on which you want to do learning.

    Each learning rule exposes two important pieces of metadata that the
    builder uses to determine what information should be stored.

    The ``size_in`` is the dimensionality of the incoming error signal. It
    can either take an integer or one of the following string values:

    * ``'pre'``: vector error signal in pre-object space
    * ``'post'``: vector error signal in post-object space
    * ``'mid'``: vector error signal in the ``conn.size_mid`` space
    * ``'pre_state'``: vector error signal in pre-synaptic ensemble space
    * ``'post_state'``: vector error signal in pre-synaptic ensemble space

    The difference between ``'post_state'`` and ``'post'`` is that with the
    former, if a ``Neurons`` object is passed, it will use the dimensionality
    of the corresponding ``Ensemble``, whereas the latter simply uses the
    ``post`` object ``size_in``. Similarly with ``'pre_state'`` and ``'pre'``.

    The ``modifies`` attribute denotes the signal targeted by the rule.
    Options are:

    * ``'encoders'``
    * ``'decoders'``
    * ``'weights'``

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which ``modifies`` will be adjusted.
    size_in : int, str, optional
        Dimensionality of the error signal (see above).

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which ``modifies`` will be adjusted.
    size_in : int, str
        Dimensionality of the error signal.
    modifies : str
        The signal targeted by the learning rule.
    """

    modifies = None
    probeable = ()

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    size_in = LearningRuleTypeSizeInParam("size_in", low=0)

    def __init__(self, learning_rate=Default, size_in=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.size_in = size_in

    def __repr__(self):
        r = []
        for name, default in self._argdefaults:
            value = getattr(self, name)
            if value != default:
                r.append("%s=%r" % (name, value))
        return "%s(%s)" % (type(self).__name__, ", ".join(r))

    @property
    def _argdefaults(self):
        return (("learning_rate", LearningRuleType.learning_rate.default),)


class PES(LearningRuleType):
    """Prescribed Error Sensitivity learning rule.

    Modifies a connection's decoders to minimize an error signal provided
    through a connection to the connection's learning rule.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    """

    modifies = "decoders"
    probeable = ("error", "activities", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-4)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default):
        super().__init__(learning_rate, size_in="post_state")
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                "This learning rate is very high, and can result "
                "in floating point errors from too much current."
            )

        self.pre_synapse = pre_synapse

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", PES.learning_rate.default),
            ("pre_synapse", PES.pre_synapse.default),
        )


class BCM(LearningRuleType):
    """Bienenstock-Cooper-Munroe learning rule.

    Modifies connection weights as a function of the presynaptic activity
    and the difference between the postsynaptic activity and the average
    postsynaptic activity.

    Notes
    -----
    The BCM rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the BCM rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the BCM rule by ``1 / post.n_neurons``.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    theta_synapse : `.Synapse`, optional
        Synapse model used to filter the theta signal.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    theta_synapse : `.Synapse`
        Synapse model used to filter the theta signal.
    """

    modifies = "weights"
    probeable = ("theta", "pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-9)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    theta_synapse = SynapseParam(
        "theta_synapse", default=Lowpass(tau=1.0), readonly=True
    )

    def __init__(
        self,
        learning_rate=Default,
        pre_synapse=Default,
        post_synapse=Default,
        theta_synapse=Default,
    ):
        super().__init__(learning_rate, size_in=0)

        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.theta_synapse = theta_synapse

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", BCM.learning_rate.default),
            ("pre_synapse", BCM.pre_synapse.default),
            ("post_synapse", self.pre_synapse),
            ("theta_synapse", BCM.theta_synapse.default),
        )


class Oja(LearningRuleType):
    """Oja learning rule.

    Modifies connection weights according to the Hebbian Oja rule, which
    augments typically Hebbian coactivity with a "forgetting" term that is
    proportional to the weight of the connection and the square of the
    postsynaptic activity.

    Notes
    -----
    The Oja rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the Oja rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the Oja rule by ``1 / post.n_neurons``.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    beta : float, optional
        A scalar weight on the forgetting term.

    Attributes
    ----------
    beta : float
        A scalar weight on the forgetting term.
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    beta = NumberParam("beta", low=0, readonly=True, default=1.0)

    def __init__(
        self,
        learning_rate=Default,
        pre_synapse=Default,
        post_synapse=Default,
        beta=Default,
    ):
        super().__init__(learning_rate, size_in=0)

        self.beta = beta
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", Oja.learning_rate.default),
            ("pre_synapse", Oja.pre_synapse.default),
            ("post_synapse", self.pre_synapse),
            ("beta", Oja.beta.default),
        )


class Voja(LearningRuleType):
    """Vector Oja learning rule.

    Modifies an ensemble's encoders to be selective to its inputs.

    A connection to the learning rule will provide a scalar weight for the
    learning rate, minus 1. For instance, 0 is normal learning, -1 is no
    learning, and less than -1 causes anti-learning or "forgetting".

    Parameters
    ----------
    post_tau : float, optional
        Filter constant on activities of neurons in post population.
    learning_rate : float, optional
        A scalar indicating the rate at which encoders will be adjusted.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which encoders will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    """

    modifies = "encoders"
    probeable = ("post_filtered", "scaled_encoders", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-2)
    post_synapse = SynapseParam(
        "post_synapse", default=Lowpass(tau=0.005), readonly=True
    )

    def __init__(self, learning_rate=Default, post_synapse=Default):
        super().__init__(learning_rate, size_in=1)

        self.post_synapse = post_synapse

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", Voja.learning_rate.default),
            ("post_synapse", Voja.post_synapse.default),
        )


class RLS(LearningRuleType):
    r"""Recursive least-squares rule for online decoder optimization.

    This implements an online version of the standard least-squares solvers used to
    learn connection weights offline (e.g. `nengo.solvers.LstsqL2`). It can be applied
    in the same scenarios as `.PES`, to minimize an error signal.

    The cost of RLS is :math:`\mathcal{O}(n^2)` extra time and memory. If possible, it
    is more efficient to do the learning offline using e.g. `nengo.solvers.LstsqL2`.

    Parameters
    ----------
    learning_rate : ``float``, optional
        Effective learning rate. This is better understood as :math:`\frac{1}{\alpha}`,
        where :math:`\alpha` is an L2-regularization term. A large learning rate means
        little regularization, which implies quick over-fitting. A small learning rate
        means large regularization, which translates to slower learning. [#]_
    pre_synapse : :class:`nengo.synapses.Synapse`, optional
        Synapse model applied to the pre-synaptic neural activities.

    See Also
    --------
    :class:`nengo.PES`
    :class:`nengo.solvers.LstsqL2`

    Notes
    -----
    RLS works by maintaining the inverse neural correlation matrix,
    :math:`P = \Gamma^{-1}`, where :math:`\Gamma = A^T A + \alpha I` are the regularized
    correlations, :math:`A` is a matrix of (possibly filtered) neural activities, and
    :math:`\alpha` is an L2-regularization term controlled by the ``learning_rate``.
    :math:`P` is used to project the error signal and update the weights each time-step.

    References
    ----------
    .. [#] Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns
       of activity from chaotic neural networks. Neuron, 63(4), 544-557.

    Examples
    --------
    See :doc:`examples/learning/force-learning` for an example of how to
    use RLS to learn spiking FORCE [1]_ and "full-FORCE" networks in Nengo.

    Below, we compare `.PES` against `.RLS`, learning a feed-forward communication
    channel (identity function) online, starting with 100 spiking LIF neurons with
    decoders (weights) set to zero. A faster learning rate for `.PES` results in
    over-fitting to the most recent online example, while a slower learning rate does
    not learn quickly enough. This is a general problem with greedy optimization.
    `.RLS` performs better since it is L2-optimal.

    .. testcode::

        from nengo.learning_rules import PES, RLS

        tau = 0.005
        learning_rules = (
            PES(learning_rate=1e-3, pre_synapse=tau),
            RLS(learning_rate=1e-3, pre_synapse=tau),
        )

        with nengo.Network() as model:
            u = nengo.Node(output=lambda t: np.sin(2 * np.pi * t))
            probes = []
            for lr in learning_rules:
                e = nengo.Node(size_in=1, output=lambda t, e: e if t < 1 else 0)
                x = nengo.Ensemble(100, 1, seed=0)
                y = nengo.Node(size_in=1)

                nengo.Connection(u, e, synapse=None, transform=-1)
                nengo.Connection(u, x, synapse=None)
                conn = nengo.Connection(
                    x, y, synapse=None, learning_rule_type=lr, function=lambda x: 0
                )
                nengo.Connection(y, e, synapse=None)
                nengo.Connection(e, conn.learning_rule, synapse=tau)
                probes.append(nengo.Probe(y, synapse=tau))
            probes.append(nengo.Probe(u, synapse=tau))

        with nengo.Simulator(model) as sim:
            sim.run(2.0)

        plt.plot(sim.trange(), sim.data[probes[0]], label=str(learning_rules[0]))
        plt.plot(sim.trange(), sim.data[probes[1]], label=str(learning_rules[1]))
        plt.plot(sim.trange(), sim.data[probes[2]], label="Ideal", linestyle="--")
        plt.vlines([1], -1, 1, label="Training -> Testing")
        plt.ylim(-2, 2)
        plt.legend(loc="upper right")
        plt.xlabel("Time (s)")

    .. testoutput::
       :hide:

       ...
    """

    modifies = "decoders"
    probeable = ("pre_filtered", "error", "delta", "inv_gamma")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-3)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default):
        super().__init__(learning_rate=learning_rate, size_in="post_state")
        self.pre_synapse = pre_synapse

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", RLS.learning_rate.default),
            ("pre_synapse", RLS.pre_synapse.default),
        )


class LearningRuleTypeParam(Parameter):
    def check_rule(self, instance, rule):
        if not isinstance(rule, LearningRuleType):
            raise ValidationError(
                "'%s' must be a learning rule type or a dict or "
                "list of such types." % rule,
                attr=self.name,
                obj=instance,
            )
        if rule.modifies not in ("encoders", "decoders", "weights"):
            raise ValidationError(
                "Unrecognized target %r" % rule.modifies, attr=self.name, obj=instance
            )

    def coerce(self, instance, rule):
        if is_iterable(rule):
            for r in rule.values() if isinstance(rule, dict) else rule:
                self.check_rule(instance, r)
        elif rule is not None:
            self.check_rule(instance, rule)
        return super().coerce(instance, rule)
