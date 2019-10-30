import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError
from nengo.learning_rules import BCM, Oja, PES, Voja
from nengo.node import Node


class SimPES(Operator):
    r"""Calculate connection weight change according to the PES rule.

    Implements the PES learning rule of the form

    .. math:: \Delta \omega_{ij} = \frac{\kappa}{n} e_j a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`n` is the number of presynaptic neurons
    * :math:`e_j` is the error for the jth output dimension, and
    * :math:`a_i` is the activity of a presynaptic neuron.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    encoders : Signal, optional
        If not None, multiply the error signal by these post-synaptic
        encoders (in the case that we want to learn a neuron-to-neuron
        weight matrix instead of decoder weights).
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    error : Signal
        The error signal, :math:`e_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    encoders : Signal, optional
        If not None, multiply the error signal by these post-synaptic
        encoders (in the case that we want to learn a neuron-to-neuron
        weight matrix instead of decoder weights).
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, error, encoders]``
    4. updates ``[delta]``
    """

    def __init__(
        self, pre_filtered, error, delta, learning_rate, encoders=None, tag=None
    ):
        super(SimPES, self).__init__(tag=tag)

        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error] + ([] if encoders is None else [encoders])
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def encoders(self):
        return None if len(self.reads) < 3 else self.reads[2]

    @property
    def error(self):
        return self.reads[1]

    @property
    def pre_filtered(self):
        return self.reads[0]

    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.delta)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        error = signals[self.error]
        delta = signals[self.delta]
        n_neurons = pre_filtered.shape[0]
        alpha = -self.learning_rate * dt / n_neurons

        if self.encoders is None:

            def step_simpes():
                np.outer(alpha * error, pre_filtered, out=delta)

        else:
            encoders = signals[self.encoders]

            def step_simpes():
                np.outer(alpha * np.dot(encoders, error), pre_filtered, out=delta)

        return step_simpes


class SimBCM(Operator):
    r"""Calculate connection weight change according to the BCM rule.

    Implements the Bienenstock-Cooper-Munroe learning rule of the form

    .. math:: \Delta \omega_{ij} = \kappa a_j (a_j - \theta_j) a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    * :math:`\theta_j` is an estimate of the average :math:`a_j`, and
    * :math:`a_i` is the activity of a presynaptic neuron.

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    theta : Signal
        The modification threshold, :math:`\theta_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    theta : Signal
        The modification threshold, :math:`\theta_j`.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, theta]``
    4. updates ``[delta]``
    """

    def __init__(
        self, pre_filtered, post_filtered, theta, delta, learning_rate, tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def theta(self):
        return self.reads[2]

    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.delta,
        )

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simbcm():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta), pre_filtered
            )

        return step_simbcm


class SimOja(Operator):
    r"""Calculate connection weight change according to the Oja rule.

    Implements the Oja learning rule of the form

    .. math:: \Delta \omega_{ij} = \kappa (a_i a_j - \beta a_j^2 \omega_{ij})

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_i` is the activity of a presynaptic neuron,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    * :math:`\beta` is a scalar forgetting rate, and
    * :math:`\omega_{ij}` is the connection weight between the two neurons.

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    beta : float
        The scalar forgetting rate, :math:`\beta`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    beta : float
        The scalar forgetting rate, :math:`\beta`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(
        self, pre_filtered, post_filtered, weights, delta, learning_rate, beta, tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.beta = beta

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.reads[2]

    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.delta,
        )

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt
        beta = self.beta

        def step_simoja():
            # perform forgetting
            post_squared = alpha * post_filtered * post_filtered
            delta[...] = -beta * weights * post_squared[:, None]

            # perform update
            delta[...] += np.outer(alpha * post_filtered, pre_filtered)

        return step_simoja


class SimVoja(Operator):
    r"""Simulates a simplified version of Oja's rule in the vector space.

    See :doc:`examples/learning/learn_associations` for details.

    Parameters
    ----------
    pre_decoded : Signal
        Decoded activity from presynaptic ensemble, :math:`a_i`.
    post_filtered : Signal
        Filtered postsynaptic activity signal.
    scaled_encoders : Signal
        2d array of encoders, multiplied by ``scale``.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    scale : ndarray
        The length of each encoder.
    learning_signal : Signal
        Scalar signal to be multiplied by ``learning_rate``. Expected to be
        either 0 or 1 to turn learning off or on, respectively.
    learning_rate : float
        The scalar learning rate.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate.
    learning_signal : Signal
        Scalar signal to be multiplied by ``learning_rate``. Expected to be
        either 0 or 1 to turn learning off or on, respectively.
    post_filtered : Signal
        Filtered postsynaptic activity signal.
    pre_decoded : Signal
        Decoded activity from presynaptic ensemble, :math:`a_i`.
    scale : ndarray
        The length of each encoder.
    scaled_encoders : Signal
        2d array of encoders, multiplied by ``scale``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_decoded, post_filtered, scaled_encoders, learning_signal]``
    4. updates ``[delta]``
    """

    def __init__(
        self,
        pre_decoded,
        post_filtered,
        scaled_encoders,
        delta,
        scale,
        learning_signal,
        learning_rate,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.scale = scale
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_decoded, post_filtered, scaled_encoders, learning_signal]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def learning_signal(self):
        return self.reads[3]

    @property
    def pre_decoded(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def scaled_encoders(self):
        return self.reads[2]

    @property
    def weights(self):
        return self.reads[2]

    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_decoded,
            self.post_filtered,
            self.delta,
        )

    def make_step(self, signals, dt, rng):
        pre_decoded = signals[self.pre_decoded]
        post_filtered = signals[self.post_filtered]
        scaled_encoders = signals[self.scaled_encoders]
        delta = signals[self.delta]
        learning_signal = signals[self.learning_signal]
        alpha = self.learning_rate * dt
        scale = self.scale[:, np.newaxis]

        def step_simvoja():
            delta[...] = (
                alpha
                * learning_signal
                * (
                    scale * np.outer(post_filtered, pre_decoded)
                    - post_filtered[:, np.newaxis] * scaled_encoders
                )
            )

        return step_simvoja


def get_pre_ens(conn):
    return conn.pre_obj if isinstance(conn.pre_obj, Ensemble) else conn.pre_obj.ensemble


def get_post_ens(conn):
    return (
        conn.post_obj
        if isinstance(conn.post_obj, (Ensemble, Node))
        else conn.post_obj.ensemble
    )


def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)


@Builder.register(LearningRule)
def build_learning_rule(model, rule):
    """Builds a `.LearningRule` object into a model.

    A brief summary of what happens in the learning rule build process,
    in order:

    1. Create a delta signal for the weight change.
    2. Add an operator to increment the weights by delta.
    3. Call build function for the learning rule type.

    The learning rule system is designed to work with multiple learning rules
    on the same connection. If only one learning rule was to be applied to the
    connection, then we could directly modify the weights, rather than
    calculating the delta here and applying it in `.build_connection`.
    However, with multiple learning rules, we must isolate each delta signal
    in case calculating the delta depends on the weights themselves,
    making the calculation depend on the order of the learning rule
    evaluations.

    Parameters
    ----------
    model : Model
        The model to build into.
    rule : LearningRule
        The learning rule to build.

    Notes
    -----
    Sets ``model.params[rule]`` to ``None``.
    """

    conn = rule.connection

    # --- Set up delta signal
    if rule.modifies == "encoders":
        if not conn.is_decoded:
            ValueError(
                "The connection must be decoded in order to use encoder learning."
            )
        post = get_post_ens(conn)
        target = model.sig[post]["encoders"]
        tag = "encoders += delta"
    elif rule.modifies in ("decoders", "weights"):
        target = model.sig[conn]["weights"]
        tag = "weights += delta"
    else:
        raise BuildError("Unknown target %r" % rule.modifies)

    delta = Signal(shape=target.shape, name="Delta")

    model.add_op(Copy(delta, target, inc=True, tag=tag))
    model.sig[rule]["delta"] = delta

    model.params[rule] = None  # by default, no build-time info to return
    model.build(rule.learning_rule_type, rule)  # updates delta


@Builder.register(BCM)
def build_bcm(model, bcm, rule):
    """Builds a `.BCM` object into a model.

    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimBCM` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    bcm : BCM
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.BCM` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, bcm.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, bcm.post_synapse, post_activities)
    theta = build_or_passthrough(model, bcm.theta_synapse, post_activities)

    model.add_op(
        SimBCM(
            pre_filtered,
            post_filtered,
            theta,
            model.sig[rule]["delta"],
            learning_rate=bcm.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["theta"] = theta
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered


@Builder.register(Oja)
def build_oja(model, oja, rule):
    """Builds a `.BCM` object into a model.

    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimOja` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    oja : Oja
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Oja` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, oja.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, oja.post_synapse, post_activities)

    model.add_op(
        SimOja(
            pre_filtered,
            post_filtered,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=oja.learning_rate,
            beta=oja.beta,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered


@Builder.register(Voja)
def build_voja(model, voja, rule):
    """Builds a `.Voja` object into a model.

    Calls synapse build functions to filter the post activities,
    and adds a `.SimVoja` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    voja : Voja
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Voja` instance.
    """

    conn = rule.connection

    # Filtered post activity
    post = conn.post_obj
    post_filtered = build_or_passthrough(
        model, voja.post_synapse, model.sig[post]["out"]
    )

    # Learning signal, defaults to 1 in case no connection is made
    # and multiplied by the learning_rate * dt
    learning = Signal(shape=rule.size_in, name="Voja:learning")
    assert rule.size_in == 1
    model.add_op(Reset(learning, value=1.0))
    model.sig[rule]["in"] = learning  # optional connection will attach here

    scaled_encoders = model.sig[post]["encoders"]
    # The gain and radius are folded into the encoders during the ensemble
    # build process, so we need to make sure that the deltas are proportional
    # to this scaling factor
    encoder_scale = model.params[post].gain / post.radius
    assert post_filtered.shape == encoder_scale.shape

    model.add_op(
        SimVoja(
            pre_decoded=model.sig[conn]["out"],
            post_filtered=post_filtered,
            scaled_encoders=scaled_encoders,
            delta=model.sig[rule]["delta"],
            scale=encoder_scale,
            learning_signal=learning,
            learning_rate=voja.learning_rate,
        )
    )

    model.sig[rule]["scaled_encoders"] = scaled_encoders
    model.sig[rule]["post_filtered"] = post_filtered


@Builder.register(PES)
def build_pes(model, pes, rule):
    """Builds a `.PES` object into a model.

    Calls synapse build functions to filter the pre activities,
    and adds a `.SimPES` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    pes : PES
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.PES` instance.
    """

    conn = rule.connection

    # Create input error signal
    error = Signal(shape=rule.size_in, name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough(model, pes.pre_synapse, model.sig[conn.pre_obj]["out"])

    if conn.is_decoded:
        encoders = None
    else:
        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"][:, conn.post_slice]

    model.add_op(
        SimPES(
            acts, error, model.sig[rule]["delta"], pes.learning_rate, encoders=encoders
        )
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
