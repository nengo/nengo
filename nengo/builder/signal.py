from io import StringIO

import numpy as np

from nengo.exceptions import SignalError
from nengo.rc import rc
from nengo.transforms import SparseMatrix
import nengo.utils.numpy as npext


def is_sparse(obj):
    """Check if ``obj`` is a sparse matrix."""
    return isinstance(obj, SparseMatrix) or npext.is_spmatrix(obj)


class Signal:
    """Represents data or views onto data within a Nengo simulation.

    Signals are tightly coupled to NumPy arrays, which is how live data is
    represented in a Nengo simulation. Signals provide a view onto the
    important metadata of the live NumPy array, and maintain the original
    value of the array in order to reset the simulation to the initial state.

    Parameters
    ----------
    initial_value : array_like
        The initial value of the signal. Much of the metadata tracked by the
        Signal is based on this array as well (e.g., dtype).
    name : str, optional
        Name of the signal. Primarily used for debugging.
        If None, the memory location of the Signal will be used.
    base : Signal, optional
        The base signal, if this signal is a view on another signal.
        Linking the two signals with the ``base`` argument is necessary
        to ensure that their live data is also linked.
    readonly : bool, optional
        Whether this signal and its related live data should be marked as
        readonly. Writing to these arrays will raise an exception.
    offset : int, optional
        For a signal view this gives the offset of the view from the base
        ``initial_value`` in bytes. This might differ from the offset
        of the NumPy array view provided as ``initial_value`` if the base
        is a view already (in which case the signal base offset will be 0
        because it starts where the view starts. That NumPy view can have
        an offset of itself).
    """

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(
        self,
        initial_value=None,
        shape=None,
        name=None,
        base=None,
        readonly=False,
        offset=0,
    ):
        self._initial_value = None  # set as None temporarily, for initial_value setter

        if self.assert_named_signals:
            assert name is not None
        self._name = name

        if initial_value is None:
            assert shape is not None
            initial_value = np.zeros(shape, dtype=rc.float_dtype)
        elif shape is not None:
            assert initial_value.shape == shape

        self._base = base
        self._offset = offset
        self._readonly = bool(readonly)

        self.initial_value = initial_value

    def __getstate__(self):
        state = dict(self.__dict__)

        if not self.sparse:
            # For normal arrays, the initial value could be a view on another
            # signal's data. To make sure we do not make a copy of the data,
            # we store the underlying metadata in a tuple, which pickle can
            # inspect to see that v.base is the same in different signals
            # and avoid serializing it multiple times.
            v = self._initial_value
            state["_initial_value"] = (
                v.shape,
                v.base,
                npext.array_offset(v),
                v.strides,
            )

        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        if not self.sparse:
            shape, base, offset, strides = self._initial_value
            self._initial_value = np.ndarray(
                shape, buffer=base, dtype=base.dtype, offset=offset, strides=strides
            )
            self._initial_value.setflags(write=False)

    def __getitem__(self, item):
        """Index or slice into array."""
        if self.sparse:
            raise SignalError("Attempting to create a view of a sparse Signal")

        if item is Ellipsis or (isinstance(item, slice) and item == slice(None)):
            return self

        if not isinstance(item, tuple):
            item = (item,)

        if not all(npext.is_integer(i) or isinstance(i, slice) for i in item):
            raise SignalError("Can only index or slice into signals")

        if all(npext.is_integer(i) for i in item):
            # turn one index into slice to get a view from numpy
            item = item[:-1] + (slice(item[-1], item[-1] + 1),)

        view = self._initial_value[item]
        offset = npext.array_offset(view) - npext.array_offset(self._initial_value)
        return Signal(
            view, name="%s[%s]" % (self.name, item), base=self.base, offset=offset
        )

    def __repr__(self):
        return "Signal(name=%s, shape=%s)" % (self._name, self.shape)

    @property
    def base(self):
        """(Signal) The base signal, if this signal is a view.

        Linking the two signals with the ``base`` argument is necessary
        to ensure that their live data is also linked.
        """
        return self if self._base is None else self._base

    @property
    def dtype(self):
        """(numpy.dtype) Data type of the signal (e.g., float64)."""
        return self.initial_value.dtype

    @property
    def elemoffset(self):
        """(int) Offset of data from base in elements."""
        return self.offset // self.itemsize

    @property
    def elemstrides(self):
        """(int) Strides of data in elements."""
        return None if self.sparse else tuple(s // self.itemsize for s in self.strides)

    @property
    def initial_value(self):
        """(numpy.ndarray) Initial value of the signal.

        Much of the metadata tracked by the Signal is based on this array
        as well (e.g., dtype).
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        if self._initial_value is not None and self.shape != initial_value.shape:
            raise SignalError(
                "Replacement shape %s must equal original shape %s"
                % (initial_value.shape, self.shape)
            )

        self._initial_value = initial_value
        if np.any(
            np.isnan(self._initial_value.data if self.sparse else self._initial_value)
        ):
            raise SignalError("%r contains NaNs." % self)

        if self.sparse:
            assert initial_value.ndim == 2
            assert self.offset == 0
            assert not self.is_view
            if npext.is_spmatrix(initial_value):
                self._initial_value.data.setflags(write=False)
        else:
            # To ensure we do not modify data passed into the signal,
            # we make a view of the data and mark it as not writeable.
            # Consumers (like SignalDict) are responsible for making copies
            # that can be modified, or using the readonly view appropriately.
            readonly_view = np.asarray(self._initial_value)
            if readonly_view.ndim > 0 and self.base is None:
                readonly_view = np.ascontiguousarray(readonly_view)
            # Ensure we have a view and aren't modifying the original's flags
            readonly_view = readonly_view.view()
            readonly_view.setflags(write=False)
            self._initial_value = readonly_view

        if self.is_view:
            assert isinstance(self.base, Signal) and not self.base.is_view
            # make sure initial_value uses the same data as base.initial_value
            assert initial_value.base is self.base.initial_value.base

    @property
    def is_view(self):
        """(bool) True if this Signal is a view on another Signal."""
        return self._base is not None

    @property
    def itemsize(self):
        """(int) Size of an array element in bytes."""
        return self.dtype.itemsize

    @property
    def name(self):
        """(str) Name of the signal. Primarily used for debugging."""
        return self._name if self._name is not None else ("0x%x" % id(self))

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def nbytes(self):
        """(int) Number of bytes consumed by the signal."""
        return self.itemsize * self.size

    @property
    def ndim(self):
        """(int) Number of array dimensions."""
        return self.initial_value.ndim

    @property
    def offset(self):
        """(int) Offset of data from base in bytes.

        For a signal view this gives the offset of the view from the base
        ``initial_value`` in bytes. This might differ from the offset
        of the NumPy array view provided as ``initial_value`` if the base
        is a view already (in which case the signal base offset will be 0
        because it starts where the view starts. That NumPy view can have
        an offset of itself).
        """
        return self._offset

    @property
    def readonly(self):
        """(bool) Whether associated live data can be changed."""
        return self._readonly

    @readonly.setter
    def readonly(self, readonly):
        self._readonly = bool(readonly)

    @property
    def shape(self):
        """(tuple) Tuple of array dimensions."""
        return self.initial_value.shape

    @property
    def size(self):
        """(int) Total number of elements."""
        return self.initial_value.size

    @property
    def strides(self):
        """(tuple) Strides of data in bytes."""
        return None if self.sparse else self.initial_value.strides

    @property
    def sparse(self):
        """(bool) Whether the signal is sparse."""
        return is_sparse(self.initial_value)

    def may_share_memory(self, other):
        """Determine if two signals might overlap in memory.

        This comparison is not exact and errs on the side of false positives.
        See `numpy.may_share_memory` for more details.

        Parameters
        ----------
        other : Signal
            The other signal we are investigating.
        """
        return (self.is_view or other.is_view) and np.may_share_memory(
            self.initial_value, other.initial_value
        )

    def reshape(self, *shape):
        """Return a view on this signal with a different shape.

        Note that ``reshape`` cannot change the overall size of the signal.
        See `numpy.reshape` for more details.

        Any number of integers can be passed to this method,
        describing the desired shape of the returned signal.
        """

        if self.sparse:
            raise SignalError("Attempting to create a view of a sparse Signal")

        if len(shape) == 1:
            shape = shape[0]  # in case a tuple is passed in
        initial_value = self.initial_value.view()
        try:
            # this raises AttributeError if cannot reshape without copying
            initial_value.shape = shape
        except AttributeError:
            raise SignalError(
                "Reshaping %s to %s would require the array to be copied "
                "(because it is not contiguous), which is not supported" % (self, shape)
            )
        return Signal(
            initial_value,
            name="%s.reshape(%s)" % (self.name, shape),
            base=self.base,
            offset=self.offset,
        )


class SignalDict(dict):
    """Map from Signal -> ndarray.

    This dict subclass ensures that the ndarray values aren't overwritten,
    and instead data are written into them, which ensures that
    these arrays never get copied, which wastes time and space.

    Use ``init`` to set the ndarray initially.
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if isinstance(key, Signal) and key.base is not key:
                # return a view on the base signal
                base = dict.__getitem__(self, key.base)
                return np.ndarray(
                    buffer=base,
                    dtype=key.dtype,
                    shape=key.shape,
                    offset=key.offset,
                    strides=key.strides,
                )
            else:
                raise

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
        """Set up a permanent mapping from signal -> data."""
        if signal in self:
            raise SignalError("Cannot add signal twice")

        assert isinstance(signal, Signal)
        data = signal.initial_value
        if isinstance(data, SparseMatrix):
            data = data.allocate()

        if signal.is_view:
            if signal.base not in self:
                self.init(signal.base)

            # get a view onto the base data
            view = np.ndarray(
                shape=data.shape,
                strides=data.strides,
                offset=signal.offset,
                dtype=data.dtype,
                buffer=self[signal.base].data,
            )
            assert np.array_equal(view, data)
            view.setflags(write=not signal.readonly)
            dict.__setitem__(self, signal, view)
        else:
            data = data if signal.readonly else data.copy()
            dict.__setitem__(self, signal, data)

    def reset(self, signal):
        """Reset ndarray to the base value of the signal that maps to it."""
        if not signal.readonly:
            self[signal] = signal.initial_value
