import logging

import numpy as np

from .builder import Signal, Constant
from . import simulator as sim

logger = logging.getLogger(__name__)


class Nonlinearity(object):
    operator = None

    def __str__(self):
        return "Nonlinearity (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        # XXX: do we still need to append signals to model?
        model.signals.append(self.bias_signal)
        model.signals.append(self.input_signal)
        model.signals.append(self.output_signal)
        model._operators.append(
            self.operator(
                output=self.output_signal,
                J=self.input_signal,
                nl=self))
        # -- encoders will be scheduled between this copy
        #    and nl_op
        model._operators.append(
            sim.Copy(dst=self.input_signal, src=self.bias_signal))


class Direct(Nonlinearity):

    operator = sim.SimDirect

    def __init__(self, n_in, n_out, fn, name=None):
        if name is None:
            name = "<Direct%d>" % id(self)
        self.name = name

        self.input_signal = Signal(n_in, name=name + '.input')
        self.output_signal = Signal(n_out, name=name + '.output')
        self.bias_signal = Constant(np.zeros(n_in),
                                    name=name + '.bias')

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(
                    self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.items():
                if k == 'fn':
                    rval.fn = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval

    def __str__(self):
        return "Direct (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'fn': self.fn.__name__,
        }


class GainNonlinearity(Nonlinearity):
    _gain = None

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain


class _LIFBase(GainNonlinearity):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, name=None):
        if name is None:
            name = "<%s%d>" % (self.__class__.__name__, id(self))
        self.input_signal = Signal(n_neurons, name=name + '.input')
        self.output_signal = Signal(n_neurons, name=name + '.output')
        self.bias_signal = Constant(np.zeros(n_neurons), name=name + '.bias')

        self.name = name
        self.n_neurons = n_neurons
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = None

    def __str__(self):
        return "%s (id %d, %dN)" % (
            self.__class__.__name__, id(self), self.n_neurons)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'n_neurons': self.n_neurons,
            'tau_rc': self.tau_rc,
            'tau_ref': self.tau_ref,
            'gain': self.gain.tolist(),
        }

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.input_signal.name = value + '.input'
        self.output_signal.name = value + '.output'
        self.bias_signal.name = value + '.bias'

    @property
    def bias(self):
        return self.bias_signal.value

    @bias.setter
    def bias(self, value):
        self.bias_signal.value[...] = value

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, _n_neurons):
        self._n_neurons = _n_neurons
        self.input_signal.n = _n_neurons
        self.output_signal.n = _n_neurons
        self.bias_signal.n = _n_neurons
        self.bias_signal.value = np.zeros(_n_neurons)

    @property
    def n_out(self):
        return self.n_neurons

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


class LIFRate(_LIFBase):
    operator = sim.SimLIFRate
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
    operator = sim.SimLIF
    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample
        self.voltage = Signal(n_neurons)
        self.refractory_time = Signal(n_neurons)

    def add_to_model(self, model):
        # XXX: do we still need to append signals to model?
        model.signals.append(self.bias_signal)
        model.signals.append(self.input_signal)
        model.signals.append(self.output_signal)
        model._operators.append(
            self.operator(
                output=self.output_signal,
                J=self.input_signal,
                nl=self,
                voltage=self.voltage,
                refractory_time=self.refractory_time))
        # -- encoders will be scheduled between this copy
        #    and nl_op
        model._operators.append(
            sim.Copy(dst=self.input_signal, src=self.bias_signal))

    def to_json(self):
        d = _LIFBase.to_json(self)
        d['upsample'] = self.upsample
        return d

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
