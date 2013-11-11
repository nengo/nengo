import logging

import numpy as np

from . import decoders

logger = logging.getLogger(__name__)


class PythonFunction(object):
    def __init__(self, fn, n_in, n_out=None, name=None):
        if name is None:
            name = "<Direct%d>" % id(self)
        self.name = name
        self.n_in = n_in
        if n_out is None:
            self.n_out = np.asarray(fn(np.ones(n_in))).size
        self.fn = fn

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.items():
                if k == 'fn':
                    rval.fn = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval


class Neurons(object):
    def __init__(self, n_neurons, bias=None, gain=None, name=None):
        self.n_neurons = n_neurons
        self.bias = bias
        self.gain = gain
        if name is None:
            name = "<%s%d>" % (self.__class__.__name__, id(self))
        self.name = name

    def __str__(self):
        r = self.__class__.__name__ + "("
        r += self.name if hasattr(self, 'name') else "id " + str(id(self))
        r += ", %dN)" if hasattr(self, 'n_neurons') else ")"
        return r

    def __repr__(self):
        return str(self)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def default_encoders(self, dimensions, rng):
        raise NotImplementedError("Neurons must provide default_encoders")

    def rates(self, J_without_bias):
        raise NotImplementedError("Neurons must provide rates")

    def set_gain_bias(self, max_rates, intercepts):
        raise NotImplementedError("Neurons must provide set_gain_bias")

class Direct(Neurons):
    def __init__(self, n_neurons=None, name=None):
        # n_neurons is ignored, but accepted to maintain compatibility
        # with other neuron types
        Neurons.__init__(self, 0, name=name)

    @property
    def n_in(self):
        # Dimensions are set in the build process
        return self.dimensions

    @property
    def n_out(self):
        # Dimensions are set in the build process
        return self.dimensions

    def default_encoders(self, dimensions, rng):
        return np.eye(dimensions)

    def rates(self, J_without_bias):
        return J_without_bias

    def set_gain_bias(self, max_rates, intercepts):
        pass


# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class _LIFBase(Neurons):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, name=None):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        Neurons.__init__(self, n_neurons, name=name)

    def default_encoders(self, dimensions, rng):
        return decoders.sample_hypersphere(
            dimensions, self.n_neurons, rng, surface=True)

    def rates(self, J_without_bias):
        """LIF firing rates in Hz

        Parameters
        ---------
        J_without_bias: ndarray of any shape
            membrane currents, without bias voltage
        """
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            J = J_without_bias + self.bias
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A

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
        logging.debug("Setting gain and bias on %s", self.name)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts


class LIFRate(_LIFBase):
    def math(self, dt, J):
        """Compute rates for input current (incl. bias)"""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            r = dt / (self.tau_ref + self.tau_rc * np.log1p(1./j))
        finally:
            np.seterr(**old)
        return r


class LIF(_LIFBase):
    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        old = np.seterr(all='ignore')
        try:
            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked[:] = (v > 1) * 1.0

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time
