import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import ElementwiseInc, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError
from nengo.learning_rules import BCM, Oja, PES, Voja
from nengo.node import Node
from nengo.synapses import Lowpass


class IntermittentOperator(Operator):
    """Base class for operators that support intermittent execution.

    An intermittent operator allows for execution in less than each simulation
    time step. To implement an intermittent operator derive from this class
    and ensure that:

    * In the constructor of the derived class you append to the operator lists
      ``self.sets``, ``self.incs``, ``self.reads``, and ``self.updates`` or, if
      you decide to overwrite them, include `step` in ``self.reads``.
    * Set ``self.scaled`` to a list of signals that you want to scale by the
      period length (``apply_every``) each time step.
    * Instead of `make_step` implement `make_concrete_step` in the same way
      you would implement `make_step` normally. No special case code is needed.

    Parameters
    ----------
    apply_every : float or None
        A scalar indicating how often to apply the operator (in simulation
        time). A value of ``None`` will apply it every time step.
    step : Signal
        Signal providing the integer timestep of the simulator.
    tag : str or None, optional (Default: None)
        A label associated with the operator, for debugging purposes

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply the operator (in simulation
        time). A value of ``None`` will apply it every time step.
    step : Signal
        Signal providing the integer timestep of the simulator.
    tag : str or None, optional (Default: None)
        A label associated with the operator, for debugging purposes
    scaled : list of Signals
        Signals to be scaled by the period length (``apply_every``).

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[step]``
    4. updates ``[]``

    Deriving classes are expected to extend these lists as needed.
    """
    def __init__(self, apply_every, step, tag=None):
        super(IntermittentOperator, self).__init__(tag=tag)
        self.apply_every = apply_every
        self.step = step

        self.sets = []
        self.incs = []
        self.reads = [step]
        self.updates = []

    def make_concrete_step(self, signals, dt, rng):
        return lambda: None

    def make_step(self, signals, dt, rng):
        period = 1. if self.apply_every is None else self.apply_every / dt
        scaled = [signals[s] for s in self.scaled]
        concrete_step = self.make_concrete_step(signals, dt, rng)

        if period > 1.:
            step = signals[self.step]

            def step_intermittent():
                if step % period < 1.:
                    concrete_step()
                    for s in scaled:
                        s[...] *= period
                else:
                    for s in scaled:
                        s[...] = 0.
        else:
            step_intermittent = concrete_step

        return step_intermittent


class SimBCM(IntermittentOperator):
    """Calculate connection weight change according to the BCM rule.

    Implements the Bienenstock-Cooper-Munroe learning rule of the form

    .. math:: \Delta \omega_{ij} = \kappa a_j (a_j - \\theta_j) a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    * :math:`\\theta_j` is an estimate of the average :math:`a_j`, and
    * :math:`a_i` is the activity of a presynaptic neuron.
    * :math:`\omega_{ij}` is the connection weight between the two neurons.

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    theta : Signal
        The modification threshold, :math:`\\theta_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    step : Signal
        The integer timestep which the simulator is executing.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    step : Signal
        The integer timestep which the simulator is executing.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    theta : Signal
        The modification threshold, :math:`\\theta_j`.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, theta, step]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, theta, delta,
                 learning_rate, apply_every, step, tag=None):
        super(SimBCM, self).__init__(
            apply_every=apply_every, step=step, tag=tag)
        self.pre_filtered = pre_filtered
        self.post_filtered = post_filtered
        self.theta = theta
        self.delta = delta
        self.learning_rate = learning_rate

        self.sets += []
        self.incs += []
        self.reads += [pre_filtered, post_filtered, theta]
        self.updates += [delta]
        self.scaled = [delta]

    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (
            self.pre_filtered, self.post_filtered, self.delta)

    def make_concrete_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simbcm():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta),
                pre_filtered)
        return step_simbcm


class SimOja(IntermittentOperator):
    """Calculate connection weight change according to the Oja rule.

    Implements the Oja learning rule of the form

    .. math:: \Delta \omega_{ij} = \kappa (a_i a_j - \\beta a_j^2 \omega_{ij})

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_i` is the activity of a presynaptic neuron,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    * :math:`\\beta` is a scalar forgetting rate, and
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
        The scalar forgetting rate, :math:`\\beta`.
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    step : Signal
        The integer timestep which the simulator is executing.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    beta : float
        The scalar forgetting rate, :math:`\\beta`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    step : Signal
        The integer timestep which the simulator is executing.
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
    3. reads ``[pre_filtered, post_filtered, weights, step]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, weights, delta,
                 learning_rate, beta, apply_every, step, tag=None):
        super(SimOja, self).__init__(
            apply_every=apply_every, step=step, tag=tag)
        self.pre_filtered = pre_filtered
        self.post_filtered = post_filtered
        self.weights = weights
        self.delta = delta
        self.learning_rate = learning_rate
        self.beta = beta

        self.sets += []
        self.incs += []
        self.reads += [pre_filtered, post_filtered, weights]
        self.updates += [delta]
        self.scaled = [delta]

    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (
            self.pre_filtered, self.post_filtered, self.delta)

    def make_concrete_step(self, signals, dt, rng):
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


class SimVoja(IntermittentOperator):
    """Simulates a simplified version of Oja's rule in the vector space.

    See :doc:`examples/learn_associations` for details.

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
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    step : Signal
        The integer timestep which the simulator is executing.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate.
    learning_signal : Signal
        Scalar signal to be multiplied by ``learning_rate``. Expected to be
        either 0 or 1 to turn learning off or on, respectively.
    step : Signal
        The integer timestep which the simulator is executing.
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
    3. reads ``[pre_decoded, post_filtered, scaled_encoders,
                learning_signal, step]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_decoded, post_filtered, scaled_encoders, delta,
                 scale, learning_signal, learning_rate, apply_every,
                 step, tag=None):
        super(SimVoja, self).__init__(
            apply_every=apply_every, step=step, tag=tag)
        self.pre_decoded = pre_decoded
        self.post_filtered = post_filtered
        self.scaled_encoders = scaled_encoders
        self.delta = delta
        self.scale = scale
        self.learning_signal = learning_signal
        self.learning_rate = learning_rate

        self.sets += []
        self.incs += []
        self.reads += [pre_decoded, post_filtered, scaled_encoders,
                       learning_signal]
        self.updates += [delta]
        self.scaled = [delta]

    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (
            self.pre_decoded, self.post_filtered, self.delta)

    def make_concrete_step(self, signals, dt, rng):
        pre_decoded = signals[self.pre_decoded]
        post_filtered = signals[self.post_filtered]
        scaled_encoders = signals[self.scaled_encoders]
        delta = signals[self.delta]
        learning_signal = signals[self.learning_signal]
        alpha = self.learning_rate * dt
        scale = self.scale[:, np.newaxis]

        def step_simvoja():
            delta[...] = alpha * learning_signal * (
                scale * np.outer(post_filtered, pre_decoded) -
                post_filtered[:, np.newaxis] * scaled_encoders)
        return step_simvoja


class SimPESDecoders(IntermittentOperator):
    """Calculate decoder change according to the PES rule.

    Implements the PES learning rule of the form:

    .. math:: \Delta d_i = -\frac{kappa}{n} E a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`n` is the number of neurons
    * :math:`E` is the error signal
    * :math:`a_i` is the activity of a presynaptic neuron, and
    * :math:`d_i` is the value of a decoder being learned

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    delta : Signal
        The decoder value change to be applied, :math:`\Delta d_i`.
    correction : Signal
        The scaled negative error
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    error : Signal
        The error driving the learning, :math:`E`
    n_neurons : int
        The number of neurons in the ensemble, :math:`n`
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    step : Signal
        The integer timestep which the simulator is executing.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    correction : Signal
        The scaled negative error
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    error : Signal
        The error driving the learning, :math:`E`
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    n_neurons : int
        The number of neurons in the ensemble, :math:`n`
    step : Signal
        The integer timestep which the simulator is executing.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, error, step]``
    4. updates ``[delta, correction]``
    """

    def __init__(self, pre_filtered, delta, correction, learning_rate, error,
                 n_neurons, apply_every, step, tag=None):
        super(SimPESDecoders, self).__init__(
            apply_every=apply_every, step=step, tag=tag)
        self.pre_filtered = pre_filtered
        self.delta = delta
        self.correction = correction
        self.learning_rate = learning_rate
        self.error = error
        self.n_neurons = n_neurons

        self.sets += []
        self.incs += []
        self.reads += [pre_filtered, error]
        self.updates += [delta, correction]
        self.scaled = [delta]

    def _descstr(self):
        return 'pre=%s, err=%s -> %s' % (
            self.pre_filtered, self.error, self.delta)

    def make_concrete_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        delta = signals[self.delta]
        correction = signals[self.correction]
        alpha = self.learning_rate * dt
        n_neurons = self.n_neurons
        error = signals[self.error]

        def step_simpesdecoders():
                    correction[...] = -alpha / n_neurons * error
                    delta[...] = np.outer(correction, pre_filtered)
        return step_simpesdecoders


class SimPESWeights(IntermittentOperator):
    """Calculate connection weight change according to the PES rule.

    Implements the PES learning rule of the form:

    .. math:: \Delta \omega_{ij} = -\frac{kappa}{n} \alpha_j e_j \cdot E a_i

    where

    * :math:`\kappa` is a scalar learning rate,
    * :math:`n` is the number of neurons
    * :math:`alpha_j` is the gain of a postsynaptic neuron
    * :math:`e_j` is the encoder of a postsynaptic neuron
    * :math:`E` is the error signal
    * :math:`a_i` is the activity of a presynaptic neuron, and
    * :math:`\omega_{ij}` is the connection weight between the two neurons.

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    delta : Signal
        The decoder value change to be applied, :math:`\Delta d_i`.
    correction : Signal
        The scaled negative error
    encoders : Signal
        The scaled encoders of a postsynaptic neuron, :math: \alpha_j e_j
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    error : Signal
        The error driving the learning, :math:`E`
    n_neurons : int
        the number of neurons in the ensemble, :math:`n`
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    step : Signal
        The integer timestep which the simulator is executing.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    apply_every : float or None
        A scalar indicating how often to apply learning rule.
    correction : Signal
         The scaled negative error
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    error : Signal
        The error driving the learning, :math:`E`
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    n_neurons : int
        the number of neurons in the ensemble, :math:`n`
    step : Signal
        The integer timestep which the simulator is executing.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    encoders : Signal
        The scaled encoders of a postsynaptic neuron, :math: \alpha_j e_j
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, encoders, error, step]``
    4. updates ``[delta, correction]``
    """

    def __init__(self, pre_filtered, delta, correction, encoders,
                 learning_rate, error, n_neurons, apply_every, step,
                 tag=None):
        super(SimPESWeights, self).__init__(
            apply_every=apply_every, step=step, tag=tag)
        self.pre_filtered = pre_filtered
        self.delta = delta
        self.correction = correction
        self.encoders = encoders
        self.learning_rate = learning_rate
        self.error = error
        self.n_neurons = n_neurons

        self.sets += []
        self.incs += []
        self.reads += [pre_filtered, encoders, error]
        self.updates += [delta, correction]
        self.scaled = [delta]

    def _descstr(self):
        return 'pre=%s, err=%s -> %s' % (
            self.pre_filtered, self.error, self.delta)

    def make_concrete_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        delta = signals[self.delta]
        correction = signals[self.correction]
        encoders = signals[self.encoders]
        alpha = self.learning_rate * dt
        n_neurons = self.n_neurons
        error = signals[self.error]

        def step_simpesweights():
                    correction[...] = -alpha / n_neurons * error
                    delta[...] = np.outer(
                        np.dot(encoders, correction), pre_filtered)
        return step_simpesweights


def get_pre_ens(conn):
    return (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
            else conn.pre_obj.ensemble)


def get_post_ens(conn):
    return (conn.post_obj if isinstance(conn.post_obj, (Ensemble, Node))
            else conn.post_obj.ensemble)


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
    if rule.modifies == 'encoders':
        if not conn.is_decoded:
            ValueError("The connection must be decoded in order to use "
                       "encoder learning.")
        post = get_post_ens(conn)
        target = model.sig[post]['encoders']
        tag = "encoders += delta"
        delta = Signal(
            np.zeros((post.n_neurons, post.dimensions)), name='Delta')
    elif rule.modifies in ('decoders', 'weights'):
        pre = get_pre_ens(conn)
        target = model.sig[conn]['weights']
        tag = "weights += delta"
        if not conn.is_decoded:
            post = get_post_ens(conn)
            delta = Signal(
                np.zeros((post.n_neurons, pre.n_neurons)), name='Delta')
        else:
            delta = Signal(
                np.zeros((rule.size_in, pre.n_neurons)), name='Delta')
    else:
        raise BuildError("Unknown target %r" % rule.modifies)

    assert delta.shape == target.shape
    model.add_op(
        ElementwiseInc(model.sig['common'][1], delta, target, tag=tag))
    model.sig[rule]['delta'] = delta

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
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    pre_filtered = model.build(Lowpass(bcm.pre_tau), pre_activities)
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    post_filtered = model.build(Lowpass(bcm.post_tau), post_activities)
    theta = model.build(Lowpass(bcm.theta_tau), post_filtered)

    model.add_op(SimBCM(pre_filtered,
                        post_filtered,
                        theta,
                        model.sig[rule]['delta'],
                        learning_rate=bcm.learning_rate,
                        apply_every=bcm.apply_every,
                        step=model.step))

    # expose these for probes
    model.sig[rule]['theta'] = theta
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered


@Builder.register(Oja)
def build_oja(model, oja, rule):
    """Builds a `.Oja` object into a model.

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
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    post_activities = model.sig[get_post_ens(conn).neurons]['out']
    pre_filtered = model.build(Lowpass(oja.pre_tau), pre_activities)
    post_filtered = model.build(Lowpass(oja.post_tau), post_activities)

    model.add_op(SimOja(pre_filtered,
                        post_filtered,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        learning_rate=oja.learning_rate,
                        beta=oja.beta,
                        apply_every=oja.apply_every,
                        step=model.step))

    # expose these for probes
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered


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
    if voja.post_tau is not None:
        post_filtered = model.build(
            Lowpass(voja.post_tau), model.sig[post]['out'])
    else:
        post_filtered = model.sig[post]['out']

    # Learning signal, defaults to 1 in case no connection is made
    # and multiplied by the learning_rate * dt
    learning = Signal(np.zeros(rule.size_in), name="Voja:learning")
    assert rule.size_in == 1
    model.add_op(Reset(learning, value=1.0))
    model.sig[rule]['in'] = learning  # optional connection will attach here

    scaled_encoders = model.sig[post]['encoders']
    # The gain and radius are folded into the encoders during the ensemble
    # build process, so we need to make sure that the deltas are proportional
    # to this scaling factor
    encoder_scale = model.params[post].gain / post.radius
    assert post_filtered.shape == encoder_scale.shape

    model.add_op(
        SimVoja(pre_decoded=model.sig[conn]['out'],
                post_filtered=post_filtered,
                scaled_encoders=scaled_encoders,
                delta=model.sig[rule]['delta'],
                scale=encoder_scale,
                learning_signal=learning,
                learning_rate=voja.learning_rate,
                apply_every=voja.apply_every,
                step=model.step))

    # expose these for probes
    model.sig[rule]['scaled_encoders'] = scaled_encoders
    model.sig[rule]['post_filtered'] = post_filtered


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
    error = Signal(np.zeros(rule.size_in), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error  # error connection will attach here

    correction = Signal(np.zeros(error.shape), name="PES:correction")
    model.add_op(Reset(correction))

    pre_activities = model.sig[get_pre_ens(conn).neurons]['out']
    pre_filtered = model.build(Lowpass(pes.pre_tau), pre_activities)

    n_neurons = (conn.pre_obj.n_neurons if isinstance(conn.pre_obj, Ensemble)
                 else conn.pre_obj.size_in)

    if not conn.is_decoded:
        model.add_op(
            SimPESWeights(pre_filtered=pre_filtered,
                          delta=model.sig[rule]['delta'],
                          correction=correction,
                          encoders=model.sig[get_post_ens(conn)]['encoders'],
                          learning_rate=pes.learning_rate,
                          error=error,
                          n_neurons=n_neurons,
                          apply_every=pes.apply_every,
                          step=model.step))

    elif isinstance(conn.pre_obj, (Ensemble, Neurons)):
        model.add_op(
            SimPESDecoders(pre_filtered=pre_filtered,
                           delta=model.sig[rule]['delta'],
                           correction=correction,
                           learning_rate=pes.learning_rate,
                           error=error,
                           n_neurons=n_neurons,
                           apply_every=pes.apply_every,
                           step=model.step))
    else:
        raise BuildError("'pre' object '%s' not suitable for PES learning"
                         % (conn.pre_obj))

    # expose these for probes
    model.sig[rule]['error'] = error
    model.sig[rule]['correction'] = correction
    model.sig[rule]['activities'] = pre_filtered
