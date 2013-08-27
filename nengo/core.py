"""
Low-level objects
=================

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

"""
import logging

import numpy as np


logger = logging.getLogger(__name__)


"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


def filter_coefs(pstc, dt):
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
    pass


class SignalView(object):
    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)
        if name is not None:
            self._name = name

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def __repr__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    @property
    def dtype(self):
        return np.dtype(self.base._dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise TODO('reshape of strided view')

    def transpose(self, neworder=None):
        raise TODO('transpose')

    def __getitem__(self, item):
        # -- copy the shape and strides
        shape = list(self.shape)
        elemstrides = list(self.elemstrides)
        offset = self.offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            elemstrides = self.elemstrides[1:]
            offset = self.offset + item * self.elemstrides[0]
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            if self.base is self:
                return '<anon%d>' % id(self)
            else:
                return 'View(%s)' % self.base.name

    @name.setter
    def name(self, value):
        self._name = value

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'base': self.base.name,
            'shape': list(self.shape),
            'elemstrides': list(self.elemstrides),
            'offset': self.offset,
        }


class Signal(SignalView):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1, dtype=np.float64, name=None):
        self.n = n
        self._dtype = dtype
        if name is not None:
            self._name = name
        if assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", " + str(self.n) + "D)"
        except AttributeError:
            return "Signal (id " + str(id(self)) + ", " + str(self.n) + "D)"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return (self.n,)

    @property
    def elemstrides(self):
        return (1,)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

    def add_to_model(self, model):
        model.signals.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'n': self.n,
            'dtype': str(self.dtype),
        }


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return "Probing " + str(self.sig)

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.probes.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'dt': self.dt,
        }


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value, name=None):
        Signal.__init__(self, n, name=name)
        self.value = np.asarray(value)
        # TODO: change constructor to get n from value
        assert self.value.size == n

    def __str__(self):
        if self.name is not None:
            return "Constant(" + self.name + ")"
        return "Constant(id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'value': self.value.tolist(),
        }


def is_constant(sig):
    """
    Return True iff `sig` is (or is a view of) a Constant signal.
    """
    return isinstance(sig.base, Constant)


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        alpha = np.asarray(alpha)
        if hasattr(outsig, 'value'):
            raise TypeError('transform destination is constant')
        if is_constant(insig):
            raise TypeError('constant input (use filter instead)')

        name = insig.name + ">" + outsig.name + ".tf_alpha"

        self.alpha_signal = Constant(n=alpha.size, value=alpha, name=name)
        self.insig = insig
        self.outsig = outsig
        if self.alpha_signal.size == 1:
            if self.insig.shape != self.outsig.shape:
                raise ShapeMismatch()
        else:
            if self.alpha_signal.shape != (
                    self.outsig.shape + self.insig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.insig.shape,
                        self.outsig.shape,
                        )

    def __str__(self):
        return ("Transform (id " + str(id(self)) + ")"
                " from " + str(self.insig) + " to " + str(self.outsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.add(self.alpha_signal)
        model.transforms.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'insig': self.insig.name,
            'outsig': self.outsig.name,
        }


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        if hasattr(newsig, 'value'):
            raise TypeError('filter destination is constant')
        alpha = np.asarray(alpha)

        name = oldsig.name + ">" + newsig.name + ".f_alpha"

        self.alpha_signal = Constant(n=alpha.size, value=alpha, name=name)
        self.oldsig = oldsig
        self.newsig = newsig

        if self.alpha_signal.size == 1:
            if self.oldsig.shape != self.newsig.shape:
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )
        else:
            if self.alpha_signal.shape != (
                    self.newsig.shape + self.oldsig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )

    def __str__(self):
        return ("Filter (id " + str(id(self)) + ")"
                " from " + str(self.oldsig) + " to " + str(self.newsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.add(self.alpha_signal)
        model.filters.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'oldsig': self.oldsig.name,
            'newsig': self.newsig.name,
        }


class Encoder(object):
    """A linear transform from a signal to a population"""
    def __init__(self, sig, pop, weights):
        self.sig = sig
        self.pop = pop
        weights = np.asarray(weights)
        if weights.shape != (pop.n_in, sig.size):
            raise ValueError('weight shape', weights.shape)
        name = sig.name + ".encoders"
        self.weights_signal = Constant(n=weights.size, value=weights, name=name)

    def __str__(self):
        return ("Encoder (id " + str(id(self)) + ")"
                " of " + str(self.sig) + " to " + str(self.pop))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.encoders.add(self)
        model.signals.add(self.weights_signal)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'pop': self.pop.name,
            'weights': self.weights.tolist(),
        }


class Decoder(object):
    """A linear transform from a population to a signal"""
    def __init__(self, pop, sig, weights):
        self.pop = pop
        self.sig = sig
        weights = np.asarray(weights)
        if weights.shape != (sig.size, pop.n_out):
            raise ValueError('weight shape', weights.shape)
        name = sig.name + ".decoders"
        self.weights_signal = Constant(n=weights.size, value=weights, name=name)

    def __str__(self):
        return ("Decoder (id " + str(id(self)) + ")"
                " of " + str(self.pop) + " to " + str(self.sig))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.decoders.add(self)
        model.signals.add(self.weights_signal)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'pop': self.pop.name,
            'sig': self.sig.name,
            'weights': self.weights.tolist(),
        }


class Nonlinearity(object):
    def __str__(self):
        return "Nonlinearity (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.nonlinearities.add(self)
        model.signals.add(self.bias_signal)
        model.signals.add(self.input_signal)
        model.signals.add(self.output_signal)


class Direct(Nonlinearity):
    def __init__(self, n_in, n_out, fn, name=None):
        if name is None:
            name = "<Direct%d>" % id(self)
        self.name = name

        self.input_signal = Signal(n_in, name=name + '.input')
        self.output_signal = Signal(n_out, name=name + '.output')
        self.bias_signal = Constant(n=n_in,
                                    value=np.zeros(n_in),
                                    name=name + '.bias')

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __str__(self):
        return "Direct (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def fn(self, J):
        return J

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'fn': inspect.getsource(self.fn),
        }


class _LIFBase(Nonlinearity):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, name=None):
        if name is None:
            name = "<%s%d>" % (self.__class__.__name__, id(self))
        self.input_signal = Signal(n_neurons, name=name + '.input')
        self.output_signal = Signal(n_neurons, name=name + '.output')
        self.bias_signal = Constant(
            n=n_neurons, value=np.zeros(n_neurons), name=name + '.bias')

        self._name = name
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
        """Return LIF firing rates for current J in Hz

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
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
    def math(self, J):
        """Compute rates for input current (incl. bias)"""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            r = 1. / (self.tau_ref + self.tau_rc * np.log1p(1./j))
        finally:
            np.seterr(**old)
        return r


class LIF(_LIFBase):
    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    def to_json(self):
        d = _LIFBase.to_json(self)
        d['upsample'] = upsample
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

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked[:] = (v > 1) * 1.0

        old = np.seterr(all='ignore')
        try:

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
