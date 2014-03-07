import logging

import numpy as np

from nengo.objects import Neurons
from nengo.utils.distributions import UniformHypersphere

logger = logging.getLogger(__name__)


class Direct(Neurons):

    def __init__(self, n_neurons=None, label=None):
        # n_neurons is ignored, but accepted to maintain compatibility
        # with other neuron types
        Neurons.__init__(self, 0, label=label)

    def default_encoders(self, dimensions, rng):
        return np.identity(dimensions)

    def rates(self, x):
        return x

    def set_gain_bias(self, max_rates, intercepts):
        pass


# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class _LIFBase(Neurons):

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, label=None):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        Neurons.__init__(self, n_neurons, label=label)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def default_encoders(self, dimensions, rng):
        sphere = UniformHypersphere(dimensions, surface=True)
        return sphere.sample(self.n_neurons, rng=rng)

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

    def rates(self, x):
        """LIF firing rates in Hz for vector space

        Parameters
        ---------
        x: ndarray of any shape
            vector-space inputs
        """
        J = self.gain * x + self.bias
        return self.rates_from_current(J)

    def set_gain_bias(self, max_rates, intercepts):
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
        logging.debug("Setting gain and bias on %s", self.label)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts


class LIFRate(_LIFBase):

    def math(self, dt, J):
        """Compute rates for input current (incl. bias)"""
        return dt * self.rates_from_current(J)


class LIF(_LIFBase):

    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

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
