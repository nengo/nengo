from __future__ import division

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NeuronType(object):

    probeable = []

    def rates(self, x, gain, bias):
        raise NotImplementedError("Neurons must provide rates")

    def gain_bias(self, max_rates, intercepts):
        raise NotImplementedError("Neurons must provide gain_bias")


class Direct(NeuronType):
    """Direct mode. Functions are computed explicitly, instead of in neurons.
    """

    def rates(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        return None, None


# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class _LIFBase(NeuronType):
    """Abstract base class for LIF neuron types."""

    probeable = ['neuron_output', 'spikes']

    def __init__(self, tau_rc=0.02, tau_ref=0.002):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def rates_from_current(self, J):
        """LIF firing rates in Hz for input current (incl. bias)"""
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            j = J - 1    # because we're using log1p instead of log
            r = 1. / (self.tau_ref + self.tau_rc * np.log1p(1. / j))
            # NOTE: There is a known bug in numpy that np.log1p(inf) returns
            #   NaN instead of inf: https://github.com/numpy/numpy/issues/4225
            r[j <= 0] = 0
        finally:
            np.seterr(**old)
        return r

    def rates(self, x, gain, bias):
        """LIF firing rates in Hz for vector space

        Parameters
        ---------
        x: ndarray of any shape
            vector-space inputs
        """
        J = gain * x + bias
        return self.rates_from_current(J)

    def gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        """
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
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


class LIFRate(_LIFBase):
    """Rate version of the leaky integrate-and-fire (LIF) neuron model."""

    def step_math(self, dt, J, output):
        """Compute rates for input current (incl. bias)"""

        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = dt / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))
        # the above line is designed to throw an error if any j is nan
        # (nan > 0 -> error), and not pass x < -1 to log1p


class LIF(_LIFBase):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model."""

    probeable = ['neuron_output', 'spikes', 'voltage']

    def step_math(self, dt, J, spiked, voltage, refractory_time):

        # update voltage using Euler's method
        dV = (dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < 0] = 0  # clip values below zero

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike (if v > 1 set spiked = 1, else 0)
        spiked[:] = (voltage > 1)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - 1) / dV[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0
        refractory_time[spiked > 0] = self.tau_ref + spiketime


class AdaptiveLIFRate(LIFRate):
    """Adaptive rate version of the LIF neuron model."""

    def __init__(self, tau_n=1, inc_n=10e-3, **lif_args):
        super(AdaptiveLIFRate, self).__init__(**lif_args)
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step_math(self, dt, J, output, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIFRate.step_math(self, dt, J - n, output)
        n += (dt / self.tau_n) * ((self.inc_n / dt) * output - n)


class AdaptiveLIF(LIF):
    """Adaptive spiking version of the LIF neuron model."""

    def __init__(self, tau_n=1, inc_n=10e-3, **lif_args):
        super(AdaptiveLIF, self).__init__(**lif_args)
        self.tau_n = tau_n
        self.inc_n = inc_n

    def step_math(self, dt, J, output, voltage, ref, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        LIF.step_math(self, dt, J - n, output, voltage, ref)
        n += (dt / self.tau_n) * ((self.inc_n / dt) * output - n)
