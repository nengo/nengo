from __future__ import division

import logging

import numpy as np

from nengo.params import Parameter
from nengo.utils.compat import range

logger = logging.getLogger(__name__)


class NeuronType(object):

    probeable = []

    def rates(self, x, gain, bias):
        """Compute firing rates (in Hz) for given vector input, ``x``.

        This default implementation takes the naive approach of running the
        step function for a second. This should suffice for most rate-based
        neuron types; for spiking neurons it will likely fail.

        Parameters
        ---------
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
        ---------
        max_rates : ndarray(dtype=float64)
            Maximum firing rates of neurons.
        intercepts : ndarray(dtype=float64)
            X-intercepts of neurons.
        """
        J_max = 0
        J_steps = 101
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

    probeable = ['rates']

    def __init__(self, tau_ref=0.002):
        self.tau_ref = tau_ref

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

    probeable = ['rates']

    def __init__(self, tau_rc=0.02, tau_ref=0.002):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

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
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.
        """
        inv_tau_ref = 1. / self.tau_ref
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

    probeable = ['spikes', 'voltage', 'refractory_time']

    def step_math(self, dt, J, spiked, voltage, refractory_time):

        # update voltage using accurate exponential integration scheme
        dV = -np.expm1(-dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < 0] = 0  # clip values below zero

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


class AdaptiveLIFRate(LIFRate):
    """Adaptive rate version of the LIF neuron model."""

    probeable = ['rates', 'adaptation']

    def __init__(self, tau_n=1, inc_n=10e-3, **lif_args):
        super(AdaptiveLIFRate, self).__init__(**lif_args)
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step_math(self, dt, J, output, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIFRate.step_math(self, dt, J - n, output)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class AdaptiveLIF(LIF):
    """Adaptive spiking version of the LIF neuron model."""

    probeable = ['spikes', 'adaptation', 'voltage', 'refractory_time']

    def __init__(self, tau_n=1, inc_n=10e-3, **lif_args):
        super(AdaptiveLIF, self).__init__(**lif_args)
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step_math(self, dt, J, output, voltage, ref, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIF.step_math(self, dt, J - n, output, voltage, ref)
        n += (dt / self.tau_n) * (self.inc_n * output - n)


class NeuronTypeParam(Parameter):
    def validate(self, instance, neurons):
        if neurons is not None and not isinstance(neurons, NeuronType):
            raise ValueError("'%s' is not a neuron type" % neurons)
        super(NeuronTypeParam, self).validate(instance, neurons)
