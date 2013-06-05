"""
simulator_objects.py: model description classes

These classes are used to describe a Nengo model to be simulated.
Model is the input to a *simulator* (see e.g. simulator.py).

"""
import numpy as np


random_weight_rng = np.random.RandomState(12345)

class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""


class SignalView(object):
    """Interpretable, vector-valued quantity within NEF
    """
    def __init__(self, base, shape, elemstrides, offset):
        assert base
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)

    def __len__(self):
        return self.shape[0]

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


class Signal(SignalView):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1, dtype=np.float64):
        self.n = n
        self._dtype = dtype

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


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value):
        Signal.__init__(self, n)
        self.value = np.asarray(value)
        # TODO: change constructor to get n from value
        assert self.value.size == n

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))


class Nonlinearity(object):
    def __init__(self, input_signal, output_signal, bias_signal):
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.bias_signal = bias_signal


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        alpha = np.asarray(alpha)
        if hasattr(outsig, 'value'):
            raise TypeError('transform destination is constant')
        self.alpha_signal = Constant(n=alpha.size, value=alpha)
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
                        self.outsig.shape,
                        self.insig.shape,
                        )


    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        if hasattr(newsig, 'value'):
            raise TypeError('filter destination is constant')
        alpha = np.asarray(alpha)
        self.alpha_signal = Constant(n=alpha.size, value=alpha)
        self.oldsig = oldsig
        self.newsig = newsig

    def __str__(self):
        return '%s{%s, %s, %s}' % (
            self.__class__.__name__,
            self.alpha, self.oldsig, self.newsig)

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

class Encoder(object):
    """A linear transform from a signal to a population"""
    def __init__(self, sig, pop, weights=None):
        self.sig = sig
        self.pop = pop
        if weights is None:
            weights = random_weight_rng.randn(pop.n_in, sig.size)
        else:
            weights = np.asarray(weights)
            if weights.shape != (pop.n_in, sig.size):
                raise ValueError('weight shape', weights.shape)
        self.weights_signal = Constant(n=weights.size, value=weights)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

class Decoder(object):
    """A linear transform from a population to a signal"""
    def __init__(self, pop, sig, weights=None):
        self.pop = pop
        self.sig = sig
        if weights is None:
            weights = random_weight_rng.randn(sig.size, pop.n_out)
        else:
            weights = np.asarray(weights)
            if weights.shape != (sig.size, pop.n_out):
                raise ValueError('weight shape', weights.shape)
        self.weights_signal = Constant(n=weights.size, value=weights)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value


class SimModel(object):
    """
    A container for model components.
    """
    def __init__(self, dt=0.001):
        self.dt = dt
        self.signals = []
        self.nonlinearities = []
        self.encoders = []
        self.decoders = []
        self.transforms = []
        self.filters = []
        self.probes = []

    def signal(self, n=1, value=None):
        """Add a signal to the model"""
        if value is None:
            rval = Signal(n)
        else:
            rval = Constant(n, value)
        self.signals.append(rval)
        return rval

    def probe(self, sig, dt):
        """Add a probe to the model"""
        rval = Probe(sig, dt)
        self.probes.append(rval)
        return rval

    def nonlinearity(self, nl):
        """Add a nonlinearity (some computation) to the model"""
        self.nonlinearities.append(nl)
        assert nl.bias_signal not in self.signals
        assert nl.input_signal not in self.signals
        assert nl.output_signal not in self.signals
        self.signals.append(nl.bias_signal)
        self.signals.append(nl.input_signal)
        self.signals.append(nl.output_signal)
        self.transform(1.0, nl.output_signal, nl.output_signal)
        return nl

    def encoder(self, sig, pop, weights=None):
        """Add an encoder to the model"""
        rval = Encoder(sig, pop, weights=weights)
        self.encoders.append(rval)
        if rval.weights_signal not in self.signals:
            self.signals.append(rval.weights_signal)
        return rval

    def decoder(self, pop, sig, weights=None):
        """Add a decoder to the model"""
        rval = Decoder(pop, sig, weights=weights)
        self.decoders.append(rval)
        if rval.weights_signal not in self.signals:
            self.signals.append(rval.weights_signal)
        return rval

    def neuron_connection(self, src, dst, weights=None):
        """Connect two nonlinearities"""
        print "Deprecated: use encoder(src.output_signal) for neuron_connection"
        return self.encoder(src.output_signal, dst, weights)

    def transform(self, alpha, insig, outsig):
        """Add a transform to the model"""
        rval = Transform(alpha, insig, outsig)
        if rval.alpha_signal not in self.signals:
            self.signals.append(rval.alpha_signal)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        """Add a filter to the model"""
        rval = Filter(alpha, oldsig, newsig)
        if rval.alpha_signal not in self.signals:
            self.signals.append(rval.alpha_signal)
        self.filters.append(rval)
        return rval
