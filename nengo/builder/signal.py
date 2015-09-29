"""Signals represent values that will be used in the simulation.
"""

import numpy as np

import nengo.utils.numpy as npext
from nengo.utils.compat import StringIO, is_integer


class Signal(object):
    """Represents data or views onto data within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None, base=None, readonly=False):
        self._value = np.asarray(value).view()
        self._value.setflags(write=False)

        if base is not None:
            assert isinstance(base, Signal) and not base.is_view
            # make sure value uses the same data as base.value
            assert npext.array_base(value) is npext.array_base(base.value)
        self._base = base

        if self.assert_named_signals:
            assert name
        self._name = name

        self._readonly = bool(readonly)

    def __repr__(self):
        return "Signal(%s, shape=%s)" % (self._name, self.shape)

    @property
    def base(self):
        return self if self._base is None else self._base

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def elemstrides(self):
        return tuple(s / self.itemsize for s in self.value.strides)

    @property
    def is_view(self):
        return self._base is not None

    @property
    def itemsize(self):
        return self._value.itemsize

    @property
    def name(self):
        return self._name if self._name is not None else ("0x%x" % id(self))

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def offset(self):
        return npext.array_offset(self.value) / self.itemsize

    @property
    def readonly(self):
        return self._readonly

    @readonly.setter
    def readonly(self, readonly):
        self._readonly = bool(readonly)

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        raise RuntimeError("Cannot change signal value after initialization")

    def __getitem__(self, item):
        """Index or slice into array"""
        if not isinstance(item, tuple):
            item = (item,)

        if not all(is_integer(i) or isinstance(i, slice) for i in item):
            raise ValueError("Can only index or slice into signals")

        if all(map(is_integer, item)):
            # turn one index into slice to get a view from numpy
            item = item[:-1] + (slice(item[-1], item[-1]+1),)

        return Signal(self._value[item],
                      name="%s[%s]" % (self.name, item),
                      base=self.base)

    def reshape(self, *shape):
        return Signal(self._value.reshape(*shape),
                      name="%s.reshape(%s)" % (self.name, shape),
                      base=self.base)

    def column(self):
        """Reshape into a column vector."""
        return self.reshape((self.size, 1))

    def row(self):
        """Reshape into a row vector."""
        return self.reshape((1, self.size))

    def may_share_memory(self, other):
        return np.may_share_memory(self.value, other.value)


class SignalDict(dict):
    """Map from Signal -> ndarray

    This dict subclass ensures that the ndarray values aren't overwritten,
    and instead data are written into them, which ensures that
    these arrays never get copied, which wastes time and space.

    Use ``init`` to set the ndarray initially.
    """

    def __setitem__(self, key, val):
        """Ensures that ndarrays stay in the same place in memory.

        Unlike normal dicts, this means that you cannot add a new key
        to a SignalDict using __setitem__. This is by design, to avoid
        silent typos when debugging Simulator. Every key must instead
        be explicitly initialized with SignalDict.init.
        """
        self[key][...] = val

    def __str__(self):
        """Pretty-print the signals and current values."""
        sio = StringIO()
        for k in self:
            sio.write("%s %s\n" % (repr(k), repr(self[k])))
        return sio.getvalue()

    def init(self, signal):
        """Set up a permanent mapping from signal -> ndarray."""
        if signal in self:
            raise ValueError("Cannot add signal twice")

        if signal.is_view:
            if signal.base not in self:
                self.init(signal.base)

            # get a view onto the base data
            v = signal.value
            offset = npext.array_offset(v)
            view = np.ndarray(shape=v.shape, strides=v.strides, offset=offset,
                              dtype=v.dtype, buffer=self[signal.base].data)
            view.setflags(write=not signal.readonly)
            dict.__setitem__(self, signal, view)
        else:
            val = signal.value
            val = val.view() if signal.readonly else val.copy()
            dict.__setitem__(self, signal, val)

    def reset(self, signal):
        """Reset ndarray to the base value of the signal that maps to it"""
        if not signal.readonly:
            self[signal] = signal.value
