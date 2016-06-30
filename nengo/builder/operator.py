"""Operators represent calculations that will occur in the simulation.

This code adapted from sigops/operator.py and sigops/operators.py
(https://github.com/jaberg/sigops).
This modified code is included under the terms of their license:

Copyright (c) 2014, James Bergstra
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

from nengo.builder.signal import Signal
import nengo.utils.numpy as npext
from nengo.exceptions import BuildError, SimulationError


class Operator(object):
    """Base class for operator instances understood by Nengo.

    During one simulator timestep, a `.Signal` can experience

    1. at most one set operator (optional)
    2. any number of increments
    3. any number of reads
    4. at most one update

    in this specific order.

    A ``set`` defines the state of the signal at time :math:`t`, the start
    of the simulation timestep. That state can then be modified by
    ``increment`` operations. A signal's state will only be ``read`` after
    all increments are complete. The state is then finalized by an ``update``,
    which denotes the state that the signal should be at time :math:`t + dt`.

    Each operator must keep track of the signals that it manipulates,
    and which of these four types of manipulations is done to each signal
    so that the simulator can order all of the operators properly.

    .. note:: There are intentionally no default values for the
              `~.Operator.reads`, `~.Operator.sets`, `~.Operator.incs`,
              and `~.Operator.updates` properties to ensure that subclasses
              explicitly set these values.

    Parameters
    ----------
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    tag : str or None
        A label associated with the operator, for debugging purposes.
    """

    def __init__(self, tag=None):
        self.tag = tag

    def __repr__(self):
        return "<%s%s at 0x%x>" % (
            type(self).__name__, self._tagstr(), id(self))

    def __str__(self):
        strs = (s for s in (self._descstr(), self._tagstr()) if s)
        return "%s{%s}" % (type(self).__name__, ' '.join(strs))

    def _descstr(self):
        return ''

    def _tagstr(self):
        return ('"%s"' % self.tag) if self.tag is not None else ''

    @property
    def all_signals(self):
        return self.reads + self.sets + self.incs + self.updates

    @property
    def incs(self):
        """Signals incremented by this operator.

        Increments will be applied after sets (if it is set), and before reads.
        """
        return self._incs

    @incs.setter
    def incs(self, val):
        self._incs = val

    @property
    def reads(self):
        """Signals that are read and not modified by this operator.

        Reads occur after increments, and before updates.
        """
        return self._reads

    @reads.setter
    def reads(self, val):
        self._reads = val

    @property
    def sets(self):
        """Signals set by this operator.

        Sets occur first, before increments. A signal that is set here cannot
        be set or updated by any other operator.
        """
        return self._sets

    @sets.setter
    def sets(self, val):
        self._sets = val

    @property
    def updates(self):
        """Signals updated by this operator.

        Updates are the last operation to occur to a signal.
        """
        return self._updates

    @updates.setter
    def updates(self, val):
        self._updates = val

    def init_signals(self, signals):
        """Initialize the signals associated with this operator.

        The signals will be initialized into ``signals``.
        Operator subclasses that use extra buffers should create them here.

        Parameters
        ----------
        signals : SignalDict
            A mapping from signals to their associated live ndarrays.
        """
        for sig in self.all_signals:
            if sig not in signals:
                signals.init(sig)

    def make_step(self, signals, dt, rng):
        """Returns a callable that performs the desired computation.

        This method must be implemented by subclasses. To fully understand what
        an operator does, look at its implementation of ``make_step``.

        Parameters
        ----------
        signals : SignalDict
            A mapping from signals to their associated live ndarrays.
        dt : float
            Length of each simulation timestep, in seconds.
        rng : `numpy.random.RandomState`
            Random number generator for stochastic operators.
        """
        raise NotImplementedError("subclasses must implement this method.")

    @classmethod
    def supports_merge(cls):
        """Returns whether this operator type supports merges at all."""
        return False

    def can_merge(self, other):
        """Checks if this signal can be merged with another signal.

        Will return ``False`` by default. Override this method if an operator
        supports merging.

        This function is expected to be transitive and symmetric.
        """
        return False

    def merge(self, others):
        """Merge this operator with `others`.

        May lead to undefined behaviour if ``can_merge`` returns ``False`` for
        any of the elements in ``others``.

        Returns
        -------
        Operator
            The merged operator.
        dict
            Dictionary mapping old signals to new signals to update the
            signals of other operators.
        """
        raise NotImplementedError("Merge not supported by operator.")


class TimeUpdate(Operator):
    """Updates the simulation step and time.

    Implements ``step[...] += 1`` and ``time[...] = step * dt``.

    A separate operator is used (rather than a combination of `.Copy` and
    `.DotInc`) so that other backends can manage these important parts of the
    simulation state separately from other signals.

    Parameters
    ----------
    step : Signal
        The signal associated with the integer step counter.
    time : Signal
        The signal associated with the time (a float, in seconds).
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    step : Signal
        The signal associated with the integer step counter.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    time : Signal
        The signal associated with the time (a float, in seconds).

    Notes
    -----
    1. sets ``[step, time]``
    2. incs ``[]``
    3. reads ``[]``
    4. updates ``[]``
    """

    def __init__(self, step, time, tag=None):
        super(TimeUpdate, self).__init__(tag=tag)
        self.step = step
        self.time = time

        self.sets = [step, time]
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        step = signals[self.step]
        time = signals[self.time]

        def step_timeupdate():
            step[...] += 1
            time[...] = step * dt

        return step_timeupdate

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        return (self.__class__ is other.__class__)

    def merge(self, others):
        replacements = {}
        step = Signal.merge_signals_or_views(
            [self.step] + [o.step for o in others], replacements)
        time = Signal.merge_signals_or_views(
            [self.time] + [o.time for o in others], replacements)
        return TimeUpdate(step, time), replacements


class PreserveValue(Operator):
    """Marks a signal as ``set`` for the graph checker.

    This operator does no computation. It simply marks a signal as ``set``,
    allowing us to apply other ops to signals that we want to preserve their
    value across multiple time steps. It is used primarily for learning rules.

    Parameters
    ----------
    dst : Signal
        The signal whose value we want to preserve.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    dst : Signal
        The signal whose value we want to preserve.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[dst]``
    2. incs ``[]``
    3. reads ``[]``
    4. updates ``[]``
    """
    def __init__(self, dst, tag=None):
        super(PreserveValue, self).__init__(tag=tag)
        self.dst = dst

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def _descstr(self):
        return str(self.dst)

    def make_step(self, signals, dt, rng):
        def step_preservevalue():
            pass
        return step_preservevalue


class Reset(Operator):
    """Assign a constant value to a Signal.

    Implements ``dst[...] = value``.

    Parameters
    ----------
    dst : Signal
        The Signal to reset.
    value : float, optional (Default: 0)
        The constant value to which ``dst`` is set.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    dst : Signal
        The Signal to reset.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    value : float
        The constant value to which ``dst`` is set.

    Notes
    -----
    1. sets ``[dst]``
    2. incs ``[]``
    3. reads ``[]``
    4. updates ``[]``
    """

    def __init__(self, dst, value=0, tag=None):
        super(Reset, self).__init__(tag=tag)
        self.dst = dst
        self.value = float(value)

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def _descstr(self):
        return str(self.dst)

    def make_step(self, signals, dt, rng):
        target = signals[self.dst]
        value = self.value

        def step_reset():
            target[...] = value
        return step_reset

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        return (
            self.__class__ is other.__class__ and
            Signal.compatible([self.dst, other.dst]) and
            self.value == other.value)

    def merge(self, others):
        replacements = {}
        dst = Signal.merge_signals_or_views(
            [self.dst] + [o.dst for o in others], replacements)
        return Reset(dst, self.value), replacements


class Copy(Operator):
    """Assign the value of one signal to another.

    Implements ``dst[...] = src``.

    Parameters
    ----------
    src : Signal
        The signal that will be copied (read).
    dst : Signal
        The signal that will be assigned to (set).
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    dst : Signal
        The signal that will be assigned to (set).
    src : Signal
        The signal that will be copied (read).
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[dst]``
    2. incs ``[]``
    3. reads ``[src]``
    4. updates ``[]``
    """

    def __init__(self, src, dst, tag=None):
        super(Copy, self).__init__(tag=tag)
        self.src = src
        self.dst = dst

        self.sets = [dst]
        self.incs = []
        self.reads = [src]
        self.updates = []

    def _descstr(self):
        return '%s -> %s' % (self.src, self.dst)

    def make_step(self, signals, dt, rng):
        src = signals[self.src]
        dst = signals[self.dst]

        def step_copy():
            dst[...] = src
        return step_copy

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        return (
            self.__class__ is other.__class__ and
            Signal.compatible([self.dst, other.dst]) and
            Signal.compatible([self.src, other.src]))

    def merge(self, others):
        replacements = {}
        dst = Signal.merge_signals_or_views(
            [self.dst] + [o.dst for o in others], replacements)
        src = Signal.merge_signals_or_views(
            [self.src] + [o.src for o in others], replacements)
        return Copy(src, dst), replacements


class SlicedCopy(Operator):
    """Assign the value of a slice of one signal to another slice.

    Implements ``dst[dst_slice] = src[src_slice]``.

    This operator can also implement ``dst[dst_slice] += src[src_slice]``
    using the parameter ``inc``.

    Parameters
    ----------
    dst : Signal
        The signal that will be assigned to (set).
    src : Signal
        The signal that will be copied (read).
    dst_slice : slice or Ellipsis, optional (Default: Ellipsis)
        Slice associated with ``dst``.
    src_slice : slice or Ellipsis, optional (Default: Ellipsis)
        Slice associated with ``src``
    inc : bool, optional (Default: False)
        Whether this should be an increment rather than a copy.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    dst : Signal
        The signal that will be assigned to (set).
    dst_slice : list or Ellipsis
        Indices associated with ``dst``.
    src : Signal
        The signal that will be copied (read).
    src_slice : list or Ellipsis
        Indices associated with ``src``
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[] if inc else [dst]``
    2. incs ``[dst] if inc else []``
    3. reads ``[src]``
    4. updates ``[]``
    """

    def __init__(self, src, dst, src_slice=Ellipsis, dst_slice=Ellipsis,
                 inc=False, tag=None):
        super(SlicedCopy, self).__init__(tag=tag)

        if isinstance(src_slice, slice):
            src = src[src_slice]
            src_slice = Ellipsis
        if isinstance(dst_slice, slice):
            dst = dst[dst_slice]
            dst_slice = Ellipsis
        # ^ src_slice and dst_slice are now either lists of indices or Ellipsis

        self.src = src
        self.dst = dst
        self.src_slice = src_slice
        self.dst_slice = dst_slice
        self.inc = inc

        self.sets = [] if inc else [dst]
        self.incs = [dst] if inc else []
        self.reads = [src]
        self.updates = []

    def _descstr(self):
        return '%s[%s] -> %s[%s], inc=%s' % (
            self.src, self.src_slice, self.dst, self.dst_slice, self.inc)

    def make_step(self, signals, dt, rng):
        src = signals[self.src]
        dst = signals[self.dst]
        src_slice = self.src_slice
        dst_slice = self.dst_slice
        inc = self.inc

        def step_slicedcopy():
            if inc:
                dst[dst_slice] += src[src_slice]
            else:
                dst[dst_slice] = src[src_slice]
        return step_slicedcopy

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        return (
            self.__class__ is other.__class__ and
            Signal.compatible([self.src, other.src]) and
            Signal.compatible([self.dst, other.dst]) and
            self.src_slice is Ellipsis and self.dst_slice is Ellipsis and
            other.src_slice is Ellipsis and other.dst_slice is Ellipsis and
            self.inc == other.inc)

    def _merged_slice(self, signals, slices):
        if all(s is Ellipsis for s in slices):
            return Ellipsis
        elif any(s is Ellipsis for s in slices):
            raise ValueError("Mixed Ellipsis with list of indices.")

        offset = 0
        merged_slice = []
        for sig, sl in zip(signals, slices):
            merged_slice.extend([i + offset for i in sl])
            offset += sig.size
        return merged_slice

    def merge(self, others):
        src_sigs = [self.src] + [o.src for o in others]
        dst_sigs = [self.dst] + [o.dst for o in others]

        replacements = {}
        src = Signal.merge_signals_or_views(src_sigs, replacements)
        dst = Signal.merge_signals_or_views(dst_sigs, replacements)
        src_slice = self._merged_slice(
            src_sigs, [self.src_slice] + [o.src_slice for o in others])
        dst_slice = self._merged_slice(
            dst_sigs, [self.dst_slice] + [o.dst_slice for o in others])
        return SlicedCopy(
            src, dst, src_slice=src_slice, dst_slice=dst_slice,
            inc=self.inc), replacements


class ElementwiseInc(Operator):
    """Increment signal ``Y`` by ``A * X`` (with broadcasting).

    Implements ``Y[...] += A * X``.

    Parameters
    ----------
    A : Signal
        The first signal to be multiplied.
    X : Signal
        The second signal to be multiplied.
    Y : Signal
        The signal to be incremented.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    A : Signal
        The first signal to be multiplied.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    X : Signal
        The second signal to be multiplied.
    Y : Signal
        The signal to be incremented.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[A, X]``
    4. updates ``[]``
    """

    def __init__(self, A, X, Y, tag=None):
        super(ElementwiseInc, self).__init__(tag=tag)
        self.A = A
        self.X = X
        self.Y = Y

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    def _descstr(self):
        return '%s, %s -> %s' % (self.A, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        A = signals[self.A]
        X = signals[self.X]
        Y = signals[self.Y]

        # check broadcasting shapes
        Ashape = npext.broadcast_shape(A.shape, 2)
        Xshape = npext.broadcast_shape(X.shape, 2)
        Yshape = npext.broadcast_shape(Y.shape, 2)
        assert all(len(s) == 2 for s in [Ashape, Xshape, Yshape])
        for da, dx, dy in zip(Ashape, Xshape, Yshape):
            if not (da in [1, dy] and dx in [1, dy] and max(da, dx) == dy):
                raise BuildError("Incompatible shapes in ElementwiseInc: "
                                 "Trying to do %s += %s * %s" %
                                 (Yshape, Ashape, Xshape))

        def step_elementwiseinc():
            Y[...] += A * X
        return step_elementwiseinc

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        return (
            self.__class__ is other.__class__ and
            Signal.compatible([self.A, other.A], axis=self.A.ndim - 1) and
            Signal.compatible([self.X, other.X], axis=self.X.ndim - 1) and
            Signal.compatible([self.Y, other.Y], axis=self.Y.ndim - 1))

    def merge(self, others):
        replacements = {}
        A = Signal.merge_signals_or_views(
            [self.A] + [o.A for o in others], replacements,
            axis=self.A.ndim - 1)
        X = Signal.merge_signals_or_views(
            [self.X] + [o.X for o in others], replacements,
            axis=self.X.ndim - 1)
        Y = Signal.merge_signals_or_views(
            [self.Y] + [o.Y for o in others], replacements,
            axis=self.Y.ndim - 1)
        return ElementwiseInc(A, X, Y), replacements


def reshape_dot(A, X, Y, tag=None):
    """Checks if the dot product needs to be reshaped.

    Also does a bunch of error checking based on the shapes of A and X.
    """
    badshape = False
    ashape = (1,) if A.shape == () else A.shape
    xshape = (1,) if X.shape == () else X.shape

    if A.shape == ():
        incshape = X.shape
    elif X.shape == ():
        incshape = A.shape
    elif X.ndim == 1:
        badshape = ashape[-1] != xshape[0]
        incshape = ashape[:-1]
    else:
        badshape = ashape[-1] != xshape[-2]
        incshape = ashape[:-1] + xshape[:-2] + xshape[-1:]

    if (badshape or incshape != Y.shape) and incshape != ():
        raise BuildError("shape mismatch in %s: %s x %s -> %s"
                         % (tag, A.shape, X.shape, Y.shape))

    # Reshape to handle case when np.dot(A, X) and Y are both scalars
    return (np.dot(A, X)).size == Y.size == 1


class DotInc(Operator):
    """Increment signal ``Y`` by ``dot(A, X)``.

    Implements ``Y[...] += np.dot(A, X)``.

    .. note:: Currently, this only supports matrix-vector multiplies
              for compatibility with Nengo OCL.

    Parameters
    ----------
    A : Signal
        The first signal to be multiplied (a matrix).
    X : Signal
        The second signal to be multiplied (a vector).
    Y : Signal
        The signal to be incremented.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    A : Signal
        The first signal to be multiplied.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    X : Signal
        The second signal to be multiplied.
    Y : Signal
        The signal to be incremented.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[A, X]``
    4. updates ``[]``
    """

    def __init__(self, A, X, Y, tag=None):
        super(DotInc, self).__init__(tag=tag)

        if X.ndim >= 2 and any(d > 1 for d in X.shape[1:]):
            raise BuildError("X must be a column vector")
        if Y.ndim >= 2 and any(d > 1 for d in Y.shape[1:]):
            raise BuildError("Y must be a column vector")

        self.A = A
        self.X = X
        self.Y = Y

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    def _descstr(self):
        return '%s, %s -> %s' % (self.A, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step_dotinc():
            inc = np.dot(A, X)
            if reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step_dotinc

    @classmethod
    def supports_merge(cls):
        return True

    def can_merge(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.X is other.X:
            # simple merge might be possible
            return (Signal.compatible([self.Y, other.Y]) and
                    Signal.compatible([self.A, other.A]))

        # check if BSR merge is possible
        try:
            # Not using Signal.compatible for A, because A must not be a view.
            Signal.check_signals_mergeable([self.A, other.A])
            from scipy.sparse import bsr_matrix
            assert bsr_matrix
        except (ValueError, ImportError):
            return False
        return (Signal.compatible([self.X, other.X]) and
                Signal.compatible([self.Y, other.Y]) and
                self.A.shape == other.A.shape)

    def merge(self, others):
        replacements = {}

        # Simple merge if all X are the same.
        if all(o.X is self.X for o in others):
            A = Signal.merge_signals_or_views(
                [self.A] + [o.A for o in others], replacements)
            Y = Signal.merge_signals_or_views(
                [self.Y] + [o.Y for o in others], replacements)
            return DotInc(A, self.X, Y), replacements

        # BSR merge if X differ
        X = Signal.merge_signals_or_views(
            [self.X] + [o.X for o in others], replacements)
        Y = Signal.merge_signals_or_views(
            [self.Y] + [o.Y for o in others], replacements)

        # Construct sparse A representation
        data = np.stack(
            [self.A.initial_value] + [o.A.initial_value for o in others])
        indptr = np.arange(len(others) + 2, dtype=int)
        indices = np.arange(len(others) + 1, dtype=int)
        name = 'bsr_merged<{first}, ..., {last}>'.format(
            first=self.A.name, last=others[-1].A.name)
        readonly = all([self.A.readonly] + [o.A.readonly for o in others])
        A = Signal(data, name=name, readonly=readonly)
        for i, s in enumerate([self.A] + [o.A for o in others]):
            replacements[s] = Signal(
                data[i], name="%s[%i]" % (s.name, i), base=A)
            assert np.all(s.initial_value == replacements[s].initial_value)
            assert s.shape == replacements[s].shape

        reshape = reshape_dot(
            self.A.initial_value, self.X.initial_value, self.Y.initial_value,
            tag=self.tag)
        return (
            BsrDotInc(
                A, X, Y, indices=indices, indptr=indptr, reshape=reshape),
            replacements)


class BsrDotInc(Operator):
    """Increment signal Y by dot(A, X) where is a matrix in block sparse row
    format.

    Requires SciPy.

    Currently, this only supports matrix-vector multiplies for compatibility
    with NengoOCL.

    Parameters
    ----------
    A : (k, r, c) Signal
        The signal providing the k data blocks with r rows and c columns.
    X : (k * c) Signal
        The signal providing the k column vectors to multiply with.
    Y : (k * r) Signal
        The signal providing the k column vectors to update.
    indices : ndarray
        Column indices, see `scipy.sparse.bsr_matrix` for details.
    indptr : ndarray
        Column index pointers, see `scipy.sparse.bsr_matrix` for details.
    reshape : bool
        Whether to reshape the result.
    """

    def __init__(self, A, X, Y, indices, indptr, reshape, tag=None):
        super(BsrDotInc, self).__init__(tag=tag)

        if X.ndim >= 2 and any(d > 1 for d in X.shape[1:]):
            raise BuildError("X must be a column vector")
        if Y.ndim >= 2 and any(d > 1 for d in Y.shape[1:]):
            raise BuildError("Y must be a column vector")

        self.A = A
        self.X = X
        self.Y = Y
        self.indices = indices
        self.indptr = indptr
        self.reshape = reshape
        self.tag = tag

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    def _descstr(self):
        return '%s, %s -> %s' % (self.A, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]

        def step_dotinc():
            from scipy.sparse import bsr_matrix
            mat_A = bsr_matrix((A, self.indices, self.indptr))
            inc = mat_A.dot(X)
            if self.reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step_dotinc


class SimPyFunc(Operator):
    """Apply a Python function to a signal, with optional arguments.

    Implements ``output[...] = fn(*args)`` where ``args`` can
    include the current simulation time ``t`` and an input signal ``x``.

    Note that ``output`` may also be None, in which case the function is
    called but no output is captured.

    Parameters
    ----------
    output : Signal or None
        The signal to be set. If None, the function is still called.
    fn : callable
        The function to call.
    t : Signal or None
        The signal associated with the time (a float, in seconds).
        If None, the time will not be passed to ``fn``.
    x : Signal or None
        An input signal to pass to ``fn``.
        If None, an input signal will not be passed to ``fn``.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    fn : callable
        The function to call.
    output : Signal or None
        The signal to be set. If None, the function is still called.
    t : Signal or None
        The signal associated with the time (a float, in seconds).
        If None, the time will not be passed to ``fn``.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    x : Signal or None
        An input signal to pass to ``fn``.
        If None, an input signal will not be passed to ``fn``.

    Notes
    -----
    1. sets ``[] if output is None else [output]``
    2. incs ``[]``
    3. reads ``([] if t is None else [t]) + ([] if x is None else [x])``
    4. updates ``[]``
    """

    def __init__(self, output, fn, t, x, tag=None):
        super(SimPyFunc, self).__init__(tag=tag)
        self.output = output
        self.fn = fn
        self.t = t
        self.x = x

        self.sets = [] if output is None else [output]
        self.incs = []
        self.reads = ([] if t is None else [t]) + ([] if x is None else [x])
        self.updates = []

    def _descstr(self):
        return '%s -> %s, fn=%r' % (self.x, self.output, self.fn.__name__)

    def make_step(self, signals, dt, rng):
        fn = self.fn
        output = signals[self.output] if self.output is not None else None
        t = signals[self.t] if self.t is not None else None
        x = signals[self.x] if self.x is not None else None

        def step_simpyfunc():
            args = (np.copy(x),) if x is not None else ()
            y = fn(t.item(), *args) if t is not None else fn(*args)
            if output is not None:
                if y is None:  # required since Numpy turns None into NaN
                    raise SimulationError(
                        "Function %r returned None" % fn.__name__)
                try:
                    output[...] = y
                except ValueError:
                    raise SimulationError("Function %r returned invalid value "
                                          "%r" % (fn.__name__, y))

        return step_simpyfunc
