"""
simulator_objects.py: model description classes

These classes are used to describe a Nengo model to be simulated.
Model is the input to a *simulator* (see e.g. simulator.py).

"""
import numpy as np


random_weight_rng = np.random.RandomState(12345)

"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""


class SignalView(object):
    """Interpretable, vector-valued quantity within NEF
    """
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
                return '<anon>'
            else:
                return 'View(%s)' % self.base.name

    @name.setter
    def name(self, value):
        self._name = value



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
        return "Signal (" + str(self.n) + "D, id " + str(id(self)) + ")"

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



class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value, name=None):
        Signal.__init__(self, n, name=name)
        self.value = np.asarray(value)
        # TODO: change constructor to get n from value
        assert self.value.size == n

    def __str__(self):
        return "Constant (" + str(self.value) + ", id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        alpha = np.asarray(alpha)
        if hasattr(outsig, 'value'):
            raise TypeError('transform destination is constant')
        self.alpha_signal = Constant(n=alpha.size,
                                     value=alpha,
                                     name='tf_alpha')
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


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        if hasattr(newsig, 'value'):
            raise TypeError('filter destination is constant')
        alpha = np.asarray(alpha)
        self.alpha_signal = Constant(n=alpha.size, value=alpha,
                                     name='f_alpha')
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
