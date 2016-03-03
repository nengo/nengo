from __future__ import division

import logging

import numpy as np

from nengo.exceptions import SimulationError, ValidationError
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.utils.compat import range
from nengo.utils.neurons import settled_firingrate

logger = logging.getLogger(__name__)


class NeuronType(FrozenObject):
    """Base class for Nengo neuron models.

    Attributes
    ----------
    probeable : tuple
        Signals that can be probed in the neuron population.
    """

    probeable = ()

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(self._argreprs))

    @property
    def _argreprs(self):
        return []

    def gain_bias(self, max_rates, intercepts):
        """Compute the gain and bias needed to satisfy max_rates, intercepts.

        This takes the neurons, approximates their response function, and then
        uses that approximation to find the gain and bias value that will give
        the requested intercepts and max_rates.

        Note that this default implementation is very slow! Whenever possible,
        subclasses should override this with a neuron-specific implementation.

        Parameters
        ----------
        max_rates : ndarray(dtype=float64)
            Maximum firing rates of neurons.
        intercepts : ndarray(dtype=float64)
            X-intercepts of neurons.

        Returns
        -------
        gain : ndarray(dtype=float64)
            Gain associated with each neuron. Sometimes denoted alpha.
        bias : ndarray(dtype=float64)
            Bias current associated with each neuron.
        """
        J_max = 0
        J_steps = 101  # Odd number so that 0 is a sample
        max_rate = max_rates.max()

        # Start with dummy gain and bias so x == J in rate calculation
        gain = np.ones(J_steps)
        bias = np.zeros(J_steps)
        rate = np.zeros(J_steps)

        # Find range of J that will achieve max rates
        while rate[-1] < max_rate and J_max < 100:
            J_max += 10
            J = np.linspace(-J_max, J_max, J_steps)
            rate = self.rates(J, gain, bias)
        J_threshold = J[np.where(rate <= 1e-16)[0][-1]]

        gain = np.zeros_like(max_rates)
        bias = np.zeros_like(max_rates)
        for i in range(intercepts.size):
            ix = np.where(rate > max_rates[i])[0]
            if len(ix) == 0:
                ix = -1
            else:
                ix = ix[0]
            if rate[ix] == rate[ix - 1]:
                p = 1
            else:
                p = (max_rates[i] - rate[ix - 1]) / (rate[ix] - rate[ix - 1])
            J_top = p * J[ix] + (1 - p) * J[ix - 1]

            gain[i] = (J_threshold - J_top) / (intercepts[i] - 1)
            bias[i] = J_top - gain[i]

        return gain, bias

    def rates(self, x, gain, bias):
        """Compute firing rates (in Hz) for given vector input, ``x``.

        This default implementation takes the naive approach of running the
        step function for a second. This should suffice for most rate-based
        neuron types; for spiking neurons it will likely fail (those models
        should override this function).

        Parameters
        ----------
        x : ndarray(dtype=float64)
            Vector-space input.
        gain : ndarray(dtype=float64)
            Gains associated with each neuron.
        bias : ndarray(dtype=float64)
            Bias current associated with each neuron.
        """
        J = gain * x + bias
        out = np.zeros_like(J)
        self.step_math(dt=1., J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implements the differential equation for this neuron type.

        At a minimum, NeuronType subclasses must implement this method.
        That implementation should modify the ``output`` parameter rather
        than returning anything, for efficiency reasons.

        Parameters
        ----------
        dt : float
            Simulation timestep.
        J : ndarray(dtype=float64)
            Input currents associated with each neuron.
        output : ndarray(dtype=float64)
            Output activities associated with each neuron.
        """
        raise NotImplementedError("Neurons must provide step_math")


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

    def rates(self, x, gain, bias):
        """Always returns ``x``."""
        return x

    def step_math(self, dt, J, output):
        """Raises an error if called.

        Rather than calling this function, the simulator will detect that
        the ensemble is in direct mode, and bypass the neural approximation.
        """
        raise SimulationError("Direct mode neurons shouldn't be simulated.")

# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class RectifiedLinear(NeuronType):
    """A rectified linear neuron model.

    Each neuron is modeled as a rectified line. That is, the neuron's activity
    scales linearly with current, unless it passes below zero, at which point
    the neural activity will stay at zero.
    """

    probeable = ('rates',)

    def gain_bias(self, max_rates, intercepts):
        """Determine gain and bias by shifting and scaling the lines."""
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def step_math(self, dt, J, output):
        """Implement the rectification nonlinearity."""
        output[...] = np.maximum(0., J)


class Sigmoid(NeuronType):
    """A neuron model whose response curve is a sigmoid."""

    probeable = ('rates',)

    tau_ref = NumberParam('tau_ref', low=0)

    def __init__(self, tau_ref=0.002):
        super(Sigmoid, self).__init__()
        self.tau_ref = tau_ref

    @property
    def _argreprs(self):
        return [] if self.tau_ref == 0.002 else ["tau_ref=%s" % self.tau_ref]

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        lim = 1. / self.tau_ref
        gain = (-2. / (intercepts - 1.0)) * np.log(
            (2.0 * lim - max_rates) / (lim - max_rates))
        bias = -np.log(lim / max_rates - 1) - gain
        return gain, bias

    def step_math(self, dt, J, output):
        """Implement the sigmoid nonlinearity."""
        output[...] = (1. / self.tau_ref) / (1.0 + np.exp(-J))


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
    """

    probeable = ('rates',)

    tau_rc = NumberParam('tau_rc', low=0, low_open=True)
    tau_ref = NumberParam('tau_ref', low=0)

    def __init__(self, tau_rc=0.02, tau_ref=0.002):
        super(LIFRate, self).__init__()
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    @property
    def _argreprs(self):
        args = []
        if self.tau_rc != 0.02:
            args.append("tau_rc=%s" % self.tau_rc)
        if self.tau_ref != 0.002:
            args.append("tau_ref=%s" % self.tau_ref)
        return args

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError("Max rates must be below the inverse "
                                  "refractory period (%0.3f)" % inv_tau_ref,
                                  attr='max_rates', obj=self)

        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        gain = (1 - x) / (intercepts - 1.0)
        bias = 1 - gain * intercepts
        return gain, bias

    def rates(self, x, gain, bias):
        """Always use LIFRate to determine rates."""
        J = gain * x + bias
        out = np.zeros_like(J)
        # Use LIFRate's step_math explicitly to ensure rate approximation
        LIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implement the LIFRate nonlinearity."""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = 1. / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))
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
    """

    probeable = ('spikes', 'voltage', 'refractory_time')

    min_voltage = NumberParam('min_voltage', high=0)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, min_voltage=0):
        super(LIF, self).__init__(tau_rc=tau_rc, tau_ref=tau_ref)
        self.min_voltage = min_voltage

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask / dt

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + self.tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1))

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
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

    References
    ----------
    .. [1] Koch, Christof. Biophysics of Computation: Information Processing
       in Single Neurons. Oxford University Press, 1999. p. 339
    """

    probeable = ('rates', 'adaptation')

    tau_n = NumberParam('tau_n', low=0, low_open=True)
    inc_n = NumberParam('inc_n', low=0)

    def __init__(self, tau_n=1, inc_n=0.01, **lif_args):
        super(AdaptiveLIFRate, self).__init__(**lif_args)
        self.tau_n = tau_n
        self.inc_n = inc_n

    @property
    def _argreprs(self):
        args = super(AdaptiveLIFRate, self)._argreprs
        if self.tau_n != 1:
            args.append("tau_n=%s" % self.tau_n)
        if self.inc_n != 0.01:
            args.append("inc_n=%s" % self.inc_n)
        return args

    def step_math(self, dt, J, output, adaptation):
        """Implement the AdaptiveLIFRate nonlinearity."""
        n = adaptation
        LIFRate.step_math(self, dt, J - n, output)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class AdaptiveLIF(AdaptiveLIFRate, LIF):
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

    References
    ----------
    .. [1] Koch, Christof. Biophysics of Computation: Information Processing
       in Single Neurons. Oxford University Press, 1999. p. 339
    """

    probeable = ('spikes', 'adaptation', 'voltage', 'refractory_time')

    def step_math(self, dt, J, output, voltage, ref, adaptation):
        """Implement the AdaptiveLIF nonlinearity."""
        n = adaptation
        LIF.step_math(self, dt, J - n, output, voltage, ref)
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
    tau_recovery : float, optional (Default: 0.02)
        (Originally 'a') Time scale of the recovery varaible.
    coupling : float, optional (Default: 0.2)
        (Originally 'b') How sensitive recovery is to subthreshold
        fluctuations of voltage.
    reset_voltage : float, optional (Default: -65.)
        (Originally 'c') The voltage to reset to after a spike, in millivolts.
    reset_recovery : float, optional (Default: 8.)
        (Originally 'd') The recovery value to reset to after a spike.

    References
    ----------
    .. [1] E. M. Izhikevich, "Simple model of spiking neurons."
       IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572.
       (http://www.izhikevich.org/publications/spikes.pdf)
    """

    probeable = ('spikes', 'voltage', 'recovery')

    tau_recovery = NumberParam('tau_recovery', low=0, low_open=True)
    coupling = NumberParam('coupling', low=0)
    reset_voltage = NumberParam('reset_voltage')
    reset_recovery = NumberParam('reset_recovery')

    def __init__(self, tau_recovery=0.02, coupling=0.2,
                 reset_voltage=-65., reset_recovery=8.):
        super(Izhikevich, self).__init__()
        self.tau_recovery = tau_recovery
        self.coupling = coupling
        self.reset_voltage = reset_voltage
        self.reset_recovery = reset_recovery

    @property
    def _argreprs(self):
        args = []

        def add(attr, default):
            if getattr(self, attr) != default:
                args.append("%s=%s" % (attr, getattr(self, attr)))
        add("tau_recovery", 0.02)
        add("coupling", 0.2)
        add("reset_voltage", -65.)
        add("reset_recovery", 8.)
        return args

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias.

        Uses the `nengo.utils.neurons.settled_firingrate` helper function.
        """
        J = gain * x + bias
        voltage = np.zeros_like(J)
        recovery = np.zeros_like(J)
        return settled_firingrate(self.step_math, J, [voltage, recovery],
                                  settle_time=0.001, sim_time=1.0)

    def step_math(self, dt, J, spiked, voltage, recovery):
        """Implement the Izhikevich nonlinearity."""
        # Numerical instability occurs for very low inputs.
        # We'll clip them be greater than some value that was chosen by
        # looking at the simulations for many parameter sets.
        # A more principled minimum value would be better.
        J = np.maximum(-30., J)

        dV = (0.04 * voltage ** 2 + 5 * voltage + 140 - recovery + J) * 1000
        voltage[:] += dV * dt

        # We check for spikes and reset the voltage here rather than after,
        # which differs from the original implementation by Izhikevich.
        # However, calculating recovery for voltage values greater than
        # threshold can cause the system to blow up, which we want
        # to avoid at all costs.
        spiked[:] = (voltage >= 30) / dt
        voltage[spiked > 0] = self.reset_voltage

        dU = (self.tau_recovery * (self.coupling * voltage - recovery)) * 1000
        recovery[:] += dU * dt
        recovery[spiked > 0] = recovery[spiked > 0] + self.reset_recovery


class NeuronTypeParam(Parameter):
    def validate(self, instance, neurons):
        if neurons is not None and not isinstance(neurons, NeuronType):
            raise ValidationError("'%s' is not a neuron type" % neurons,
                                  attr=self.name, obj=instance)
        super(NeuronTypeParam, self).validate(instance, neurons)
