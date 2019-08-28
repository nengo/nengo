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

from nengo import npext
from nengo.exceptions import BuildError, SimulationError
from nengo.pyext import function_name


class Operator:
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

    .. note:: There are intentionally no valid default values for the
              `~.Operator.reads`, `~.Operator.sets`, `~.Operator.incs`,
              and `~.Operator.updates` properties to ensure that subclasses
              explicitly set these values.

    Parameters
    ----------
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    tag : str or None
        A label associated with the operator, for debugging purposes.
    """

    def __init__(self, tag=None):
        self.tag = tag

        self._sets = None
        self._incs = None
        self._reads = None
        self._updates = None

    def __repr__(self):
        return "<%s%s at 0x%x>" % (
            type(self).__name__,
            "" if self.tag is None else " %r" % self.tag,
            id(self),
        )

    def __str__(self):
        return "%s{%s%s}" % (
            type(self).__name__,
            self._descstr,
            "" if self.tag is None else " %r" % self.tag,
        )

    @property
    def _descstr(self):
        return ""

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
    tag : str, optional
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
        super().__init__(tag=tag)
        self.sets = [step, time]
        self.incs = []
        self.reads = []
        self.updates = []

    @property
    def step(self):
        return self.sets[0]

    @property
    def time(self):
        return self.sets[1]

    def make_step(self, signals, dt, rng):
        step = signals[self.step]
        time = signals[self.time]

        def step_timeupdate():
            step[...] += 1
            time[...] = step * dt

        return step_timeupdate


class Reset(Operator):
    """Assign a constant value to a Signal.

    Implements ``dst[...] = value``.

    Parameters
    ----------
    dst : Signal
        The Signal to reset.
    value : float, optional
        The constant value to which ``dst`` is set.
    tag : str, optional
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
        super().__init__(tag=tag)
        self.value = float(value)

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    @property
    def dst(self):
        return self.sets[0]

    @property
    def _descstr(self):
        return str(self.dst)

    def make_step(self, signals, dt, rng):
        target = signals[self.dst]
        value = self.value

        def step_reset():
            target[...] = value

        return step_reset


class Copy(Operator):
    """Assign the value of one signal to another, with optional slicing.

    Implements:

    - ``dst[:] = src``
    - ``dst[dst_slice] = src[src_slice]``
      (when ``dst_slice`` or ``src_slice`` is not None)
    - ``dst[dst_slice] += src[src_slice]`` (when ``inc=True``)

    Parameters
    ----------
    dst : Signal
        The signal that will be assigned to (set).
    src : Signal
        The signal that will be copied (read).
    dst_slice : slice or list, optional
        Slice or list of indices associated with ``dst``.
    src_slice : slice or list, optional
        Slice or list of indices associated with ``src``
    inc : bool, optional
        Whether this should be an increment rather than a copy.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    dst : Signal
        The signal that will be assigned to (set).
    dst_slice : list or None
        Indices associated with ``dst``.
    src : Signal
        The signal that will be copied (read).
    src_slice : list or None
        Indices associated with ``src``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[] if inc else [dst]``
    2. incs ``[dst] if inc else []``
    3. reads ``[src]``
    4. updates ``[]``
    """

    def __init__(self, src, dst, src_slice=None, dst_slice=None, inc=False, tag=None):
        super().__init__(tag=tag)

        if isinstance(src_slice, slice):
            src = src[src_slice]
            src_slice = None
        if isinstance(dst_slice, slice):
            dst = dst[dst_slice]
            dst_slice = None
        # ^ src_slice and dst_slice are now either lists of indices or None

        self.src_slice = src_slice
        self.dst_slice = dst_slice
        self.inc = inc

        self.sets = [] if inc else [dst]
        self.incs = [dst] if inc else []
        self.reads = [src]
        self.updates = []

    @property
    def dst(self):
        return self.incs[0] if self.inc else self.sets[0]

    @property
    def src(self):
        return self.reads[0]

    @property
    def _descstr(self):
        def sigstring(sig, sl):
            return "%s%s" % (sig, ("[%s]" % (sl,)) if sl is not None else "")

        return "%s -> %s, inc=%s" % (
            sigstring(self.src, self.src_slice),
            sigstring(self.dst, self.dst_slice),
            self.inc,
        )

    def make_step(self, signals, dt, rng):
        src = signals[self.src]
        dst = signals[self.dst]
        src_slice = self.src_slice if self.src_slice is not None else Ellipsis
        dst_slice = self.dst_slice if self.dst_slice is not None else Ellipsis
        inc = self.inc

        # If there are repeated indices in dst_slice, special handling needed.
        repeats = False
        if npext.is_array_like(dst_slice):
            # copy because we might modify it
            dst_slice = np.array(dst_slice)
            if dst_slice.dtype.kind != "b":
                # get canonical, positive indices first
                dst_slice[dst_slice < 0] += len(dst)
                repeats = len(np.unique(dst_slice)) < len(dst_slice)

        if inc and repeats:

            def step_copy():
                np.add.at(dst, dst_slice, src[src_slice])

        elif inc:

            def step_copy():
                dst[dst_slice] += src[src_slice]

        elif repeats:
            raise BuildError(
                "%s: Cannot have repeated indices in "
                "``dst_slice`` when copy is not an increment" % self
            )
        else:

            def step_copy():
                dst[dst_slice] = src[src_slice]

        return step_copy


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
    tag : str, optional
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
        super().__init__(tag=tag)
        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    @property
    def A(self):
        return self.reads[0]

    @property
    def X(self):
        return self.reads[1]

    @property
    def Y(self):
        return self.incs[0]

    @property
    def _descstr(self):
        return "%s, %s -> %s" % (self.A, self.X, self.Y)

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
                raise BuildError(
                    "Incompatible shapes in ElementwiseInc: "
                    "Trying to do %s += %s * %s" % (Yshape, Ashape, Xshape)
                )

        def step_elementwiseinc():
            Y[...] += A * X

        return step_elementwiseinc


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
        raise BuildError(
            "shape mismatch in %s: %s x %s -> %s" % (tag, A.shape, X.shape, Y.shape)
        )

    # Reshape to handle case when np.dot(A, X) and Y are both scalars
    return A.dot(X).size == Y.size == 1


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
    tag : str, optional
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

    def __init__(self, A, X, Y, reshape=None, tag=None):
        super().__init__(tag=tag)

        if X.ndim >= 2 and any(d > 1 for d in X.shape[1:]):
            raise BuildError("X must be a column vector")
        if Y.ndim >= 2 and any(d > 1 for d in Y.shape[1:]):
            raise BuildError("Y must be a column vector")

        self.reshape = reshape
        if self.reshape is None:
            self.reshape = reshape_dot(
                A.initial_value, X.initial_value, Y.initial_value, self.tag
            )

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    @property
    def A(self):
        return self.reads[0]

    @property
    def X(self):
        return self.reads[1]

    @property
    def Y(self):
        return self.incs[0]

    @property
    def _descstr(self):
        return "%s, %s -> %s" % (self.A, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]

        if self.reshape:

            def step_dotinc_reshape():
                Y[...] += A.dot(X).reshape(Y.shape)

            return step_dotinc_reshape

        else:

            def step_dotinc():
                Y[...] += A.dot(X)

            return step_dotinc


class SparseDotInc(DotInc):
    """Like `.DotInc` but ``A`` is a sparse matrix.

    .. versionadded:: 3.0.0
    """

    def __init__(self, A, X, Y, tag=None):
        if not A.sparse:
            raise BuildError("%s: A must be a sparse Signal")

        # Disallow reshaping
        super().__init__(A, X, Y, reshape=False, tag=tag)


class BsrDotInc(DotInc):
    """Increment signal Y by dot(A, X) using block sparse row format.

    Implements ``Y[...] += np.dot(A, X)``, where ``A`` is an instance
    of `scipy.sparse.bsr_matrix`.

    .. note:: Requires SciPy.

    .. note:: Currently, this only supports matrix-vector multiplies
              for compatibility with Nengo OCL.

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
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    A : (k, r, c) Signal
        The signal providing the k data blocks with r rows and c columns.
    indices : ndarray
        Column indices, see `scipy.sparse.bsr_matrix` for details.
    indptr : ndarray
        Column index pointers, see `scipy.sparse.bsr_matrix` for details.
    reshape : bool
        Whether to reshape the result.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    X : (k * c) Signal
        The signal providing the k column vectors to multiply with.
    Y : (k * r) Signal
        The signal providing the k column vectors to update.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[Y]``
    3. reads ``[A, X]``
    4. updates ``[]``
    """

    def __init__(self, A, X, Y, indices, indptr, reshape=None, tag=None):
        from scipy.sparse import bsr_matrix  # pylint: disable=import-outside-toplevel

        self.bsr_matrix = bsr_matrix

        super().__init__(A, X, Y, reshape=reshape, tag=tag)

        self.indices = indices
        self.indptr = indptr

    def make_step(self, signals, dt, rng):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]

        def step_dotinc():
            mat_A = self.bsr_matrix((A, self.indices, self.indptr))
            inc = mat_A.dot(X)
            if self.reshape:
                inc = inc.reshape(Y.shape)
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
    tag : str, optional
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
        super().__init__(tag=tag)
        self.fn = fn
        self.t_passed = t is not None
        self.x_passed = x is not None

        self.sets = [] if output is None else [output]
        self.incs = []
        self.reads = ([] if t is None else [t]) + ([] if x is None else [x])
        self.updates = []

    @property
    def output(self):
        if len(self.sets) == 1:
            return self.sets[0]
        return None

    @property
    def t(self):
        return self.reads[0] if self.t_passed else None

    @property
    def x(self):
        return self.reads[-1] if self.x_passed else None

    @property
    def _descstr(self):
        return "%s -> %s, fn=%r" % (self.x, self.output, function_name(self.fn))

    def make_step(self, signals, dt, rng):
        fn = self.fn
        output = signals[self.output] if self.output is not None else None
        t = signals[self.t] if self.t is not None else None
        x = signals[self.x] if self.x is not None else None

        def step_simpyfunc():
            args = (np.copy(x),) if x is not None else ()
            y = fn(t.item(), *args) if t is not None else fn(*args)

            if output is not None:
                try:
                    # required since Numpy turns None into NaN
                    if y is None or not np.all(np.isfinite(y)):
                        raise SimulationError(
                            "Function %r returned non-finite value"
                            % function_name(self.fn)
                        )

                    output[...] = y

                except (TypeError, ValueError):
                    raise SimulationError(
                        "Function %r returned a value "
                        "%r of invalid type %r" % (function_name(self.fn), y, type(y))
                    )

        return step_simpyfunc
