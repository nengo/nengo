from __future__ import division

import logging

import numpy as np

from nengo.params import Parameter, NumberParam
from nengo.utils.compat import range
from nengo.utils.neurons import settled_firingrate

logger = logging.getLogger(__name__)


class NeuronType(object):

    probeable = []

    @property
    def _argreprs(self):
        return []

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(self._argreprs))

    def rates(self, x, gain, bias):
        """Compute firing rates (in Hz) for given vector input, ``x``.

        This default implementation takes the naive approach of running the
        step function for a second. This should suffice for most rate-based
        neuron types; for spiking neurons it will likely fail.

        Parameters
        ----------
        x : ndarray
            vector-space input
        gain : ndarray
            gains associated with each neuron
        bias : ndarray
            bias current associated with each neuron
        """
        J = gain * x + bias
        out = np.zeros_like(J)
        self.step_math(dt=1., J=J, output=out)
        return out

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

    def step_math(self, dt, J, output):
        raise NotImplementedError("Neurons must provide step_math")


class Direct(NeuronType):
    """Direct mode. Functions are computed explicitly, instead of in neurons.
    """

    def rates(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        return None, None

    def step_math(self, dt, J, output):
        raise TypeError("Direct mode neurons shouldn't be simulated.")

# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class RectifiedLinear(NeuronType):
    """A rectified linear neuron model."""

    probeable = ['rates']

    def gain_bias(self, max_rates, intercepts):
        """Return gain and bias given maximum firing rate and x-intercept."""
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        output[...] = np.maximum(0., J)


class Sigmoid(NeuronType):
    """Neuron whose response curve is a sigmoid."""

    tau_ref = NumberParam(low=0)
    probeable = ['rates']

    def __init__(self, tau_ref=0.002):
        self.tau_ref = tau_ref

    @property
    def _argreprs(self):
        return [] if self.tau_ref == 0.002 else ["tau_ref=%s" % self.tau_ref]

    def gain_bias(self, max_rates, intercepts):
        """Return gain and bias given maximum firing rate and x-intercept."""
        lim = 1. / self.tau_ref
        gain = (-2. / (intercepts - 1.0)) * np.log(
            (2.0 * lim - max_rates) / (lim - max_rates))
        bias = -np.log(lim / max_rates - 1) - gain
        return gain, bias

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        output[...] = (1. / self.tau_ref) / (1.0 + np.exp(-J))


class LIFRate(NeuronType):
    """Rate version of the leaky integrate-and-fire (LIF) neuron model."""

    tau_rc = NumberParam(low=0, low_open=True)
    tau_ref = NumberParam(low=0)
    probeable = ['rates']

    def __init__(self, tau_rc=0.02, tau_ref=0.002):
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

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        # Use LIFRate's step_math explicitly to ensure rate approximation
        LIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to satisfy max_rates, intercepts.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ----------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.
        """
        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if (max_rates > inv_tau_ref).any():
            raise ValueError(
                "Max rates must be below the inverse refractory period (%0.3f)"
                % (inv_tau_ref))

        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        gain = (1 - x) / (intercepts - 1.0)
        bias = 1 - gain * intercepts
        return gain, bias

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = 1. / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))
        # the above line is designed to throw an error if any j is nan
        # (nan > 0 -> error), and not pass x < -1 to log1p


class LIF(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model."""

    min_voltage = NumberParam(high=0)
    probeable = ['spikes', 'voltage', 'refractory_time']

    def __init__(self, tau_rc=0.02, tau_ref=0.002, min_voltage=0):
        super(LIF, self).__init__(tau_rc=tau_rc, tau_ref=tau_ref)
        self.min_voltage = min_voltage

    def step_math(self, dt, J, spiked, voltage, refractory_time):

        # update voltage using accurate exponential integration scheme
        dV = -np.expm1(-dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < self.min_voltage] = self.min_voltage

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike (if v > 1 set spiked = 1/dt, else 0)
        spiked[:] = (voltage > 1) / dt

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - 1) / dV[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0
        refractory_time[spiked > 0] = self.tau_ref + spiketime
        
class LIF2C(NeuronType):
    
    probeable = ['spikes', 'V_A', 'V_S', 'g_shunt']
    
    V_T = NumberParam()
    V_R = NumberParam()
    C_A = NumberParam()
    C_S = NumberParam()
    g_c = NumberParam()
    g_l = NumberParam()
    
    def __init__(self, g_l=1.0, g_c=20.0, C_A=0.01, C_S=0.2, V_T=1.0, V_R=-1.0, bias=0):
        
        self.g_l = g_l
        self.g_c = g_c
        self.C_A = C_A
        self.C_S = C_S
        self.V_T = V_T
        self.V_R = V_R
        self.bias = bias
        

    def rates(self, x, gain, bias):
        """
        Returns an approximation of the rate, based on the input dimension x. The
        gain and bias determine the cell's linear RF in terms of the input vector.
        
        Have an exact approximation, so no need for calling step math?        
        
        @param x: input vector
        @param gain: RF gain
        @param bias: RF bias
        
        """
                
        J = gain * x + bias
        
        g_shunt = 0 * self.g_c * np.ones_like(J)
        
        m = 1./(self.C_A * (self.V_T - self.V_R))
        b = -self.g_l * (self.V_T + self.V_R) / (2 * (self.C_A * (self.V_T - self.V_R)))
        
        gamma = self.g_c/(self.g_c + g_shunt + self.g_l)

        
        rate = m * gamma * J + b
        rate[rate<1] = 0
        return rate
        
        
    def step_math(self, dt, J, spiked, V_A, V_S, g_shunt):
        """
        Equations for 2C LIF 
        """
        J = J - self.bias
        
        g_shunt_t = g_shunt
        g_shunt_t[g_shunt < 0] = 0
        
        g_shunt_t[g_shunt > 10] = 10
        # making g_shunt inside an exp, so it stays positive always
        dV_A = -self.g_l * V_A + self.g_c * (V_S - V_A) + self.bias
        dV_S = -self.g_l * V_S + self.g_c * (V_A - V_S) - self.g_c * g_shunt_t * V_S + J
        
        # does equation for g_shunt go here or in synapse?
        tau_g_shunt = 0.01
        dg_shunt = (0-g_shunt) / tau_g_shunt
        

        V_A[:] += dV_A * dt / self.C_A
        V_S[:] += dV_S * dt / self.C_S
        g_shunt[:] += dg_shunt * dt
        
        V_A[V_A < self.V_R] = self.V_R
        V_S[V_S < -2] = -2
        V_S[V_S > 10] = 10
        
        spiked[:] = (V_A > self.V_T) / dt
        V_A[spiked > 0] = self.V_R
        #g_shunt[:] += sum(0.1 * (spiked > 0))
        
        


class AdaptiveLIFRate(LIFRate):
    """Adaptive rate version of the LIF neuron model."""

    tau_n = NumberParam(low=0, low_open=True)
    inc_n = NumberParam(low=0)
    probeable = ['rates', 'adaptation']

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
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIFRate.step_math(self, dt, J - n, output)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class AdaptiveLIF(AdaptiveLIFRate, LIF):
    """Adaptive spiking version of the LIF neuron model."""

    probeable = ['spikes', 'adaptation', 'voltage', 'refractory_time']

    def step_math(self, dt, J, output, voltage, ref, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIF.step_math(self, dt, J - n, output, voltage, ref)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class Izhikevich(NeuronType):
    """Izhikevich neuron model.

    Implementation based on the original paper [1]_.
    Note that we rename some variables for our clarity.
    What was originally 'v' we term 'voltage', which represents the membrane
    potential of each neuron. What was originally 'u' we term 'recovery',
    which represents membrane recovery, "which accounts for the activation
    of K+ ionic currents and inactivation of Na+ ionic currents".

    We use default values that correspond to regular spiking ('RS') neurons.
    For other classes of neurons, set the parameters as follows.

    * Intrinsically bursting (IB): ``reset_voltage=-55, reset_recovery=4``
    * Chattering (CH): ``reset_voltage=-50, reset_recovery=2``
    * Fast spiking (FS): ``tau_recovery=0.1``
    * Low-threshold spiking (LTS): ``coupling=0.25``
    * Resonator (RZ): ``tau_recovery=0.1, coupling=0.26``

    Parameters
    ----------
    tau_recovery : float
        (Originally 'a') Ttime scale of the recovery varaible. Default: 0.02
    coupling : float
        (Originally 'b') How sensitive recovery is to subthreshold
        fluctuations of voltage. Default: 0.2
    reset_voltage : float
        (Originally 'c') The voltage to reset to after a spike, in millivolts.
        Default: -65
    reset_recovery : float
        (Originally 'd') The recovery value to reset to after a spike.
        Default: 8.

    References
    ----------
    .. [1] E. M. Izhikevich, "Simple model of spiking neurons."
       IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572.
       (http://www.izhikevich.org/publications/spikes.pdf)
    """

    tau_recovery = NumberParam(low=0, low_open=True)
    coupling = NumberParam(low=0)
    reset_voltage = NumberParam()
    reset_recovery = NumberParam()
    probeable = ['spikes', 'voltage', 'recovery']

    def __init__(self, tau_recovery=0.02, coupling=0.2,
                 reset_voltage=-65, reset_recovery=8):
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
        add("reset_voltage", -65)
        add("reset_recovery", 8)
        return args

    def rates(self, x, gain, bias):
        J = gain * x + bias
        voltage = np.zeros_like(J)
        recovery = np.zeros_like(J)
        return settled_firingrate(self.step_math, J, [voltage, recovery],
                                  settle_time=0.001, sim_time=1.0)

    def step_math(self, dt, J, spiked, voltage, recovery):
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
            raise ValueError("'%s' is not a neuron type" % neurons)
        super(NeuronTypeParam, self).validate(instance, neurons)
