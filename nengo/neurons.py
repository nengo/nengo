import warnings

import numpy as np

from nengo.dists import Choice, Distribution, get_samples, Uniform
from nengo.exceptions import SimulationError, ValidationError
from nengo.params import DictParam, FrozenObject, NumberParam, Parameter
from nengo.rc import rc
from nengo.utils.numpy import clip, is_array_like


def settled_firingrate(step, J, state, dt=0.001, settle_time=0.1, sim_time=1.0):
    """Compute firing rates (in Hz) for given vector input, ``x``.

    Unlike the default naive implementation, this approach takes into
    account some characteristics of spiking neurons. We start
    by simulating the neurons for a short amount of time, to let any
    initial transients settle. Then, we run the neurons for a second
    and find the average (which should approximate the firing rate).

    Parameters
    ----------
    step : function
        the step function of the neuron type
    J : ndarray
        a vector of currents to generate firing rates from
    state : dict of ndarrays
        additional state needed by the step function
    """
    total = np.zeros_like(J)
    out = state["output"]

    # Simulate for the settle time
    steps = int(settle_time / dt)
    for _ in range(steps):
        step(dt, J, **state)
    # Simulate for sim time, and keep track
    steps = int(sim_time / dt)
    for _ in range(steps):
        step(dt, J, **state)
        total += out
    return total / float(steps)


class NeuronType(FrozenObject):
    """Base class for Nengo neuron models.

    Parameters
    ----------
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.

    Attributes
    ----------
    state : {str: Distribution}
        State variables held by the neuron type during simulation.
        Values in the dict indicate their initial values, or how
        to obtain those initial values. These elements can also be
        probed in the neuron population.
    negative : bool
        Whether the neurons can emit negative outputs (i.e. negative spikes or rates).
    """

    state = {}
    negative = True
    spiking = False

    initial_state = DictParam("initial_state", optional=True)

    def __init__(self, initial_state=None):
        super().__init__()
        self.initial_state = initial_state
        if self.initial_state is not None:
            for name, value in self.initial_state.items():
                if name not in self.state:
                    raise ValidationError(
                        "State variable %r not recognized; should be one of %s"
                        % (name, ", ".join(repr(k) for k in self.state)),
                        attr="initial_state",
                        obj=self,
                    )
                if not (isinstance(value, Distribution) or is_array_like(value)):
                    raise ValidationError(
                        "State variable %r must be a distribution or array-like"
                        % (name,),
                        attr="initial_state",
                        obj=self,
                    )

    @property
    def probeable(self):
        return ("output",) + tuple(self.state)

    def current(self, x, gain, bias):
        """Compute current injected in each neuron given input, gain and bias.

        Note that ``x`` is assumed to be already projected onto the encoders
        associated with the neurons and normalized to radius 1, so the maximum
        expected current for a neuron occurs when input for that neuron is 1.

        Parameters
        ----------
        x : (n_samples,) or (n_samples, n_neurons) array_like
            Scalar inputs for which to calculate current.
        gain : (n_neurons,) array_like
            Gains associated with each neuron.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.

        Returns
        -------
        current : (n_samples, n_neurons)
            Current to be injected in each neuron.
        """
        x = np.array(x, dtype=float, copy=False, ndmin=1)
        gain = np.array(gain, dtype=float, copy=False, ndmin=1)
        bias = np.array(bias, dtype=float, copy=False, ndmin=1)

        if x.ndim == 1:
            x = x[:, np.newaxis]
        elif x.ndim >= 3 or x.shape[1] != gain.shape[0]:
            raise ValidationError(
                "Expected shape (%d, %d); got %s."
                % (x.shape[0], gain.shape[0], x.shape),
                attr="x",
                obj=self,
            )

        return gain * x + bias

    def gain_bias(self, max_rates, intercepts):
        """Compute the gain and bias needed to satisfy max_rates, intercepts.

        This takes the neurons, approximates their response function, and then
        uses that approximation to find the gain and bias value that will give
        the requested intercepts and max_rates.

        Note that this default implementation is very slow! Whenever possible,
        subclasses should override this with a neuron-specific implementation.

        Parameters
        ----------
        max_rates : (n_neurons,) array_like
            Maximum firing rates of neurons.
        intercepts : (n_neurons,) array_like
            X-intercepts of neurons.

        Returns
        -------
        gain : (n_neurons,) array_like
            Gain associated with each neuron. Sometimes denoted alpha.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.
        """
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        J_steps = 101  # Odd number so that 0 is a sample
        max_rate = max_rates.max()

        # Start with dummy gain and bias so x == J in rate calculation
        gain = np.ones(1)
        bias = np.zeros(1)

        # Find range of J that will achieve max rates (assume monotonic)
        J_threshold = None
        J_max = None
        Jr = 10
        for _ in range(10):
            J = np.linspace(-Jr, Jr, J_steps)
            rate = self.rates(J, gain, bias)
            if J_threshold is None and (rate <= 0).any():
                J_threshold = J[np.where(rate <= 0)[0][-1]]
            if J_max is None and (rate >= max_rate).any():
                J_max = J[np.where(rate >= max_rate)[0][0]]

            if J_threshold is not None and J_max is not None:
                break

            Jr *= 2
        else:
            if J_threshold is None:
                raise RuntimeError("Could not find firing threshold")
            if J_max is None:
                raise RuntimeError("Could not find max current")

        J = np.linspace(J_threshold, J_max, J_steps)
        rate = self.rates(J, gain, bias).squeeze(axis=1)

        gain = np.zeros_like(max_rates)
        bias = np.zeros_like(max_rates)
        J_tops = np.interp(max_rates, rate, J)

        gain[:] = (J_threshold - J_tops) / (intercepts - 1)
        bias[:] = J_tops - gain
        return gain, bias

    def make_state(self, n_neurons, rng=np.random, dtype=None):
        dtype = rc.float_dtype if dtype is None else dtype
        state = {}
        initial_state = {} if self.initial_state is None else self.initial_state
        for name in self.state:
            dist = initial_state.get(name, self.state[name])
            state[name] = get_samples(dist, n=n_neurons, d=None, rng=rng).astype(
                dtype, copy=False
            )
        return state

    def max_rates_intercepts(self, gain, bias):
        """Compute the max_rates and intercepts given gain and bias.

        Note that this default implementation is very slow! Whenever possible,
        subclasses should override this with a neuron-specific implementation.

        Parameters
        ----------
        gain : (n_neurons,) array_like
            Gain associated with each neuron. Sometimes denoted alpha.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.

        Returns
        -------
        max_rates : (n_neurons,) array_like
            Maximum firing rates of neurons.
        intercepts : (n_neurons,) array_like
            X-intercepts of neurons.
        """

        max_rates = self.rates(1, gain, bias).squeeze(axis=0)

        x_range = np.linspace(-1, 1, 101)
        rates = self.rates(x_range, gain, bias)
        last_zeros = np.maximum(np.argmax(rates > 0, axis=0) - 1, 0)
        intercepts = x_range[last_zeros]

        return max_rates, intercepts

    def rates(self, x, gain, bias):
        """Compute firing rates (in Hz) for given input ``x``.

        This default implementation takes the naive approach of running the
        step function for a second. This should suffice for most rate-based
        neuron types; for spiking neurons it will likely fail (those models
        should override this function).

        Note that ``x`` is assumed to be already projected onto the encoders
        associated with the neurons and normalized to radius 1, so the maximum
        expected rate for a neuron occurs when input for that neuron is 1.

        Parameters
        ----------
        x : (n_samples,) or (n_samples, n_neurons) array_like
            Scalar inputs for which to calculate rates.
        gain : (n_neurons,) array_like
            Gains associated with each neuron.
        bias : (n_neurons,) array_like
            Bias current associated with each neuron.

        Returns
        -------
        rates : (n_samples, n_neurons) ndarray
            The firing rates at each given value of ``x``.
        """
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        self.step(dt=1.0, J=J, output=out)
        return out

    def step(self, dt, J, output, **state):
        """Implements the differential equation for this neuron type.

        At a minimum, NeuronType subclasses must implement this method.
        That implementation should modify the ``output`` parameter rather
        than returning anything, for efficiency reasons.

        Parameters
        ----------
        dt : float
            Simulation timestep.
        J : (n_neurons,) array_like
            Input currents associated with each neuron.
        output : (n_neurons,) array_like
            Output activity associated with each neuron (e.g., spikes or firing rates).
        state : {str: array_like}
            State variables associated with the population.
        """
        raise NotImplementedError("Neurons must provide step")

    def step_math(self, dt, J, **state):
        warnings.warn(
            "'step_math' has been renamed to 'step'. This alias will be removed "
            "in Nengo 4.0"
        )
        return self.step(dt, J, **state)


class NeuronTypeParam(Parameter):

    equatable = True

    def coerce(self, instance, neurons):
        self.check_type(instance, neurons, NeuronType)
        return super().coerce(instance, neurons)


class Direct(NeuronType):
    """Signifies that an ensemble should simulate in direct mode.

    In direct mode, the ensemble represents and transforms signals perfectly,
    rather than through a neural approximation. Note that direct mode ensembles
    with recurrent connections can easily diverge; most other neuron types will
    instead saturate at a certain high firing rate.
    """

    def gain_bias(self, max_rates, intercepts):
        """Always returns ``None, None``."""
        return None, None

    def max_rates_intercepts(self, gain, bias):
        """Always returns ``None, None``."""
        return None, None

    def rates(self, x, gain, bias):
        """Always returns ``x``."""
        return np.array(x, dtype=float, copy=False, ndmin=1)

    def step(self, dt, J, output):
        """Raises an error if called.

        Rather than calling this function, the simulator will detect that
        the ensemble is in direct mode, and bypass the neural approximation.
        """
        raise SimulationError("Direct mode neurons shouldn't be simulated.")


class RectifiedLinear(NeuronType):
    """A rectified linear neuron model.

    Each neuron is modeled as a rectified line. That is, the neuron's activity
    scales linearly with current, unless it passes below zero, at which point
    the neural activity will stay at zero.

    Parameters
    ----------
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    negative = False

    amplitude = NumberParam("amplitude", low=0, low_open=True)

    def __init__(self, amplitude=1, initial_state=None):
        super().__init__(initial_state)

        self.amplitude = amplitude

    def gain_bias(self, max_rates, intercepts):
        """Determine gain and bias by shifting and scaling the lines."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)
        return max_rates, intercepts

    def step(self, dt, J, output):
        """Implement the rectification nonlinearity."""
        output[...] = self.amplitude * np.maximum(0.0, J)


class SpikingRectifiedLinear(RectifiedLinear):
    """A rectified integrate and fire neuron model.

    Each neuron is modeled as a rectified line. That is, the neuron's activity
    scales linearly with current, unless the current is less than zero, at
    which point the neural activity will stay at zero. This is a spiking
    version of the RectifiedLinear neuron model.

    Parameters
    ----------
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {"voltage": Uniform(low=0, high=1)}
    spiking = True

    def rates(self, x, gain, bias):
        """Use RectifiedLinear to determine rates."""

        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        RectifiedLinear.step(self, dt=1.0, J=J, output=out)
        return out

    def step(self, dt, J, output, voltage):
        """Implement the integrate and fire nonlinearity."""

        voltage += np.maximum(J, 0) * dt
        n_spikes = np.floor(voltage)
        output[:] = (self.amplitude / dt) * n_spikes
        voltage -= n_spikes


class Sigmoid(NeuronType):
    """A non-spiking neuron model whose response curve is a sigmoid.

    Since the tuning curves are strictly positive, the ``intercepts``
    correspond to the inflection point of each sigmoid. That is,
    ``f(intercept) = 0.5`` where ``f`` is the pure sigmoid function.

    Parameters
    ----------
    tau_ref : float
        The neuron refractory period, in seconds. The maximum firing rate of the
        neurons is ``1 / tau_ref``. Must be positive (i.e. ``tau_ref > 0``).
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    negative = False

    tau_ref = NumberParam("tau_ref", low=0, low_open=True)

    def __init__(self, tau_ref=0.0025, initial_state=None):
        super().__init__(initial_state)
        self.tau_ref = tau_ref

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1.0 / self.tau_ref
        if not np.all(max_rates < inv_tau_ref):
            raise ValidationError(
                "Max rates must be below the inverse refractory period (%0.3f)"
                % (inv_tau_ref,),
                attr="max_rates",
                obj=self,
            )

        inverse = -np.log(inv_tau_ref / max_rates - 1.0)
        gain = inverse / (1.0 - intercepts)
        bias = inverse - gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        inverse = gain + bias
        intercepts = 1 - inverse / gain
        max_rates = (1.0 / self.tau_ref) / (1 + np.exp(-inverse))
        return max_rates, intercepts

    def step(self, dt, J, output):
        """Implement the sigmoid nonlinearity."""
        output[...] = (1.0 / self.tau_ref) / (1 + np.exp(-J))


class Tanh(NeuronType):
    """A non-spiking neuron model whose response curve is a hyperbolic tangent.

    Parameters
    ----------
    tau_ref : float
        The neuron refractory period, in seconds. The maximum firing rate of the
        neurons is ``1 / tau_ref``. Must be positive (i.e. ``tau_ref > 0``).
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    tau_ref = NumberParam("tau_ref", low=0, low_open=True)

    def __init__(self, tau_ref=0.0025, initial_state=None):
        super().__init__(initial_state)
        self.tau_ref = tau_ref

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1.0 / self.tau_ref
        if not np.all(max_rates < inv_tau_ref):
            raise ValidationError(
                "Max rates must be below the inverse refractory period (%0.3f)"
                % inv_tau_ref,
                attr="max_rates",
                obj=self,
            )

        inverse = np.arctanh(max_rates * self.tau_ref)
        gain = inverse / (1.0 - intercepts)
        bias = -gain * intercepts
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = -bias / gain
        max_rates = (1.0 / self.tau_ref) * np.tanh(gain + bias)
        return max_rates, intercepts

    def step(self, dt, J, output):
        """Implement the tanh nonlinearity."""
        output[...] = (1.0 / self.tau_ref) * np.tanh(J)


class LIFRate(NeuronType):
    """Non-spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    negative = False

    tau_rc = NumberParam("tau_rc", low=0, low_open=True)
    tau_ref = NumberParam("tau_ref", low=0)
    amplitude = NumberParam("amplitude", low=0, low_open=True)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, amplitude=1, initial_state=None):
        super().__init__(initial_state)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1.0 / self.tau_ref if self.tau_ref > 0 else np.inf
        if not np.all(max_rates < inv_tau_ref):
            raise ValidationError(
                "Max rates must be below the inverse "
                "refractory period (%0.3f)" % inv_tau_ref,
                attr="max_rates",
                obj=self,
            )

        x = 1.0 / (1 - np.exp((self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        gain = (1 - x) / (intercepts - 1.0)
        bias = 1 - gain * intercepts
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = (1 - bias) / gain
        max_rates = 1.0 / (
            self.tau_ref - self.tau_rc * np.log1p(1.0 / (gain * (intercepts - 1) - 1))
        )
        if not np.all(np.isfinite(max_rates)):
            warnings.warn(
                "Non-finite values detected in `max_rates`; this "
                "probably means that `gain` was too small."
            )
        return max_rates, intercepts

    def rates(self, x, gain, bias):
        """Always use LIFRate to determine rates."""
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        # Use LIFRate's step explicitly to ensure rate approximation
        LIFRate.step(self, dt=1, J=J, output=out)
        return out

    def step(self, dt, J, output):
        """Implement the LIFRate nonlinearity."""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = self.amplitude / (
            self.tau_ref + self.tau_rc * np.log1p(1.0 / j[j > 0])
        )
        # the above line is designed to throw an error if any j is nan
        # (nan > 0 -> error), and not pass x < -1 to log1p


class LIF(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike


class AdaptiveLIFRate(LIFRate):
    """Adaptive non-spiking version of the LIF neuron model.

    Works as the LIF model, except with adapation state ``n``, which is
    subtracted from the input current. Its dynamics are::

        tau_n dn/dt = -n

    where ``n`` is incremented by ``inc_n`` when the neuron spikes.

    Parameters
    ----------
    tau_n : float
        Adaptation time constant. Affects how quickly the adaptation state
        decays to zero in the absence of spikes (larger = slower decay).
    inc_n : float
        Adaptation increment. How much the adaptation state is increased after
        each spike.
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.

    References
    ----------
    .. [1] Camera, Giancarlo La, et al. "Minimal models of adapted neuronal
       response to in Vivo-Like input currents." Neural computation
       16.10 (2004): 2101-2124.
    """

    state = {"adaptation": Choice([0])}

    tau_n = NumberParam("tau_n", low=0, low_open=True)
    inc_n = NumberParam("inc_n", low=0)

    def __init__(
        self,
        tau_n=1,
        inc_n=0.01,
        tau_rc=0.02,
        tau_ref=0.002,
        amplitude=1,
        initial_state=None,
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step(self, dt, J, output, adaptation):
        """Implement the AdaptiveLIFRate nonlinearity."""
        n = adaptation
        super().step(dt, J - n, output)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class AdaptiveLIF(LIF):
    """Adaptive spiking version of the LIF neuron model.

    Works as the LIF model, except with adapation state ``n``, which is
    subtracted from the input current. Its dynamics are::

        tau_n dn/dt = -n

    where ``n`` is incremented by ``inc_n`` when the neuron spikes.

    Parameters
    ----------
    tau_n : float
        Adaptation time constant. Affects how quickly the adaptation state
        decays to zero in the absence of spikes (larger = slower decay).
    inc_n : float
        Adaptation increment. How much the adaptation state is increased after
        each spike.
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.

    References
    ----------
    .. [1] Camera, Giancarlo La, et al. "Minimal models of adapted neuronal
       response to in Vivo-Like input currents." Neural computation
       16.10 (2004): 2101-2124.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
        "adaptation": Choice([0]),
    }
    spiking = True

    tau_n = NumberParam("tau_n", low=0, low_open=True)
    inc_n = NumberParam("inc_n", low=0)

    def __init__(
        self,
        tau_n=1,
        inc_n=0.01,
        tau_rc=0.02,
        tau_ref=0.002,
        min_voltage=0,
        amplitude=1,
        initial_state=None,
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            min_voltage=min_voltage,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step(self, dt, J, output, voltage, refractory_time, adaptation):
        """Implement the AdaptiveLIF nonlinearity."""
        n = adaptation
        super().step(dt, J - n, output, voltage, refractory_time)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class Izhikevich(NeuronType):
    """Izhikevich neuron model.

    This implementation is based on the original paper [1]_;
    however, we rename some variables for clarity.
    What was originally 'v' we term 'voltage', which represents the membrane
    potential of each neuron. What was originally 'u' we term 'recovery',
    which represents membrane recovery, "which accounts for the activation
    of K+ ionic currents and inactivation of Na+ ionic currents."
    The 'a', 'b', 'c', and 'd' parameters are also renamed
    (see the parameters below).

    We use default values that correspond to regular spiking ('RS') neurons.
    For other classes of neurons, set the parameters as follows.

    * Intrinsically bursting (IB): ``reset_voltage=-55, reset_recovery=4``
    * Chattering (CH): ``reset_voltage=-50, reset_recovery=2``
    * Fast spiking (FS): ``tau_recovery=0.1``
    * Low-threshold spiking (LTS): ``coupling=0.25``
    * Resonator (RZ): ``tau_recovery=0.1, coupling=0.26``

    Parameters
    ----------
    tau_recovery : float, optional
        (Originally 'a') Time scale of the recovery variable.
    coupling : float, optional
        (Originally 'b') How sensitive recovery is to subthreshold
        fluctuations of voltage.
    reset_voltage : float, optional
        (Originally 'c') The voltage to reset to after a spike, in millivolts.
    reset_recovery : float, optional
        (Originally 'd') The recovery value to reset to after a spike.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.

    References
    ----------
    .. [1] E. M. Izhikevich, "Simple model of spiking neurons."
       IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572.
       (http://www.izhikevich.org/publications/spikes.pdf)
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "recovery": Choice([0]),
    }
    negative = False
    spiking = True

    tau_recovery = NumberParam("tau_recovery", low=0, low_open=True)
    coupling = NumberParam("coupling", low=0)
    reset_voltage = NumberParam("reset_voltage")
    reset_recovery = NumberParam("reset_recovery")

    def __init__(
        self,
        tau_recovery=0.02,
        coupling=0.2,
        reset_voltage=-65.0,
        reset_recovery=8.0,
        initial_state=None,
    ):
        super().__init__(initial_state)
        self.tau_recovery = tau_recovery
        self.coupling = coupling
        self.reset_voltage = reset_voltage
        self.reset_recovery = reset_recovery

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        return settled_firingrate(
            self.step,
            J,
            state={
                "output": np.zeros_like(J),
                "voltage": np.zeros_like(J),
                "recovery": np.zeros_like(J),
            },
            settle_time=0.001,
            sim_time=1.0,
        )

    def step(self, dt, J, output, voltage, recovery):
        """Implement the Izhikevich nonlinearity."""
        # Numerical instability occurs for very low inputs.
        # We'll clip them be greater than some value that was chosen by
        # looking at the simulations for many parameter sets.
        # A more principled minimum value would be better.
        J = np.maximum(-30.0, J)

        dV = (0.04 * voltage ** 2 + 5 * voltage + 140 - recovery + J) * 1000
        voltage[:] += dV * dt

        # We check for spikes and reset the voltage here rather than after,
        # which differs from the original implementation by Izhikevich.
        # However, calculating recovery for voltage values greater than
        # threshold can cause the system to blow up, which we want
        # to avoid at all costs.
        output[:] = (voltage >= 30) / dt
        voltage[output > 0] = self.reset_voltage

        dU = (self.tau_recovery * (self.coupling * voltage - recovery)) * 1000
        recovery[:] += dU * dt
        recovery[output > 0] = recovery[output > 0] + self.reset_recovery


class RatesToSpikesNeuronType(NeuronType):
    """Base class for neuron types that turn rate types into spiking ones."""

    base_type = NeuronTypeParam("base_type")
    amplitude = NumberParam("amplitude", low=0, low_open=True)
    spiking = True

    def __init__(self, base_type, amplitude=1.0, initial_state=None):
        super().__init__(initial_state)

        self.base_type = base_type
        self.amplitude = amplitude
        self.negative = base_type.negative

        if base_type.spiking:
            warnings.warn(
                "'base_type' is type %r, which is a spiking neuron type. We recommend "
                "using the non-spiking equivalent type, if one exists."
                % (type(base_type).__name__)
            )

        for s in self.state:
            if s in self.base_type.state:
                raise ValidationError(
                    "%s and %s have overlapping state variable (%s)"
                    % (self, self.base_type, s),
                    attr="state",
                    obj=self,
                )

    def gain_bias(self, max_rates, intercepts):
        return self.base_type.gain_bias(max_rates, intercepts)

    def max_rates_intercepts(self, gain, bias):
        return self.base_type.max_rates_intercepts(gain, bias)

    def rates(self, x, gain, bias):
        return self.base_type.rates(x, gain, bias)

    def step(self, dt, J, output, **state):
        raise NotImplementedError("Subclasses must implement step")

    @property
    def probeable(self):
        return ("output", "rate_out") + tuple(self.state) + tuple(self.base_type.state)


class RegularSpiking(RatesToSpikesNeuronType):
    """Turn a rate neuron type into a spiking one with regular inter-spike intervals.

    Spikes at regular intervals based on the rates of the base neuron type. [1]_

    Parameters
    ----------
    base_type : NeuronType
        A rate-based neuron type to convert to a regularly spiking neuron.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.

    References
    ----------
    .. [1] Voelker, A. R., Rasmussen, D., & Eliasmith, C. (2020). A Spike in
       Performance: Training Hybrid-Spiking Neural Networks with Quantized Activation
       Functions. arXiv preprint arXiv:2002.03553. (https://arxiv.org/abs/2002.03553)
    """

    state = {"voltage": Uniform(low=0, high=1)}

    def step(self, dt, J, output, voltage):
        # Note: J is the desired output rate, not the input current
        voltage += dt * J
        n_spikes = np.floor(voltage)
        output[...] = (self.amplitude / dt) * n_spikes
        voltage -= n_spikes


class StochasticSpiking(RatesToSpikesNeuronType):
    """Turn a rate neuron type into a spiking one using stochastic rounding.

    The expected number of spikes per timestep ``e = dt * r`` is determined by the
    base type firing rate ``r`` and the timestep ``dt``. Given the fractional part ``f``
    and integer part ``q`` of ``e``, the number of generated spikes is ``q`` with
    probability ``1 - f`` and ``q + 1`` with probability ``f``. For ``e`` much less than
    one, this is very similar to Poisson statistics.

    Parameters
    ----------
    base_type : NeuronType
        A rate-based neuron type to convert to a Poisson spiking neuron.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    def make_state(self, n_neurons, rng=np.random, dtype=None):
        state = super().make_state(n_neurons, rng=rng, dtype=dtype)
        state["rng"] = rng
        return state

    def step(self, dt, J, output, rng, **base_state):
        # Note: J is the desired output rate, not the input current
        if self.negative:
            frac, n_spikes = np.modf(dt * np.abs(J))
        else:
            frac, n_spikes = np.modf(dt * J)

        n_spikes += rng.random_sample(size=frac.shape) < frac

        if self.negative:
            output[...] = (self.amplitude / dt) * n_spikes * np.sign(J)
        else:
            output[...] = (self.amplitude / dt) * n_spikes


class PoissonSpiking(RatesToSpikesNeuronType):
    """Turn a rate neuron type into a spiking one with Poisson spiking statistics.

    Spikes with Poisson probability based on the rates of the base neuron type.

    Parameters
    ----------
    base_type : NeuronType
        A rate-based neuron type to convert to a Poisson spiking neuron.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    def make_state(self, n_neurons, rng=np.random, dtype=None):
        state = super().make_state(n_neurons, rng=rng, dtype=dtype)
        state["rng"] = rng
        return state

    def step(self, dt, J, output, rng, **base_state):
        # Note: J is the desired output rate, not the input current
        if self.negative:
            output[...] = (
                (self.amplitude / dt)
                * rng.poisson(np.abs(J) * dt, output.size)
                * np.sign(J)
            )
        else:
            output[...] = (self.amplitude / dt) * rng.poisson(J * dt, output.size)
