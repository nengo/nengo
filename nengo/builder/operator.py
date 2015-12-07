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

import nengo.utils.numpy as npext
from nengo.exceptions import BuildError, SimulationError


class Operator(object):
    """Base class for operator instances understood by nengo.Simulator.

    The lifetime of a Signal during one simulator timestep:

    0. at most one set operator (optional)
    1. any number of increments
    2. any number of reads
    3. at most one update

    A signal that is only read can be considered a "constant".

    A signal that is both set *and* updated can be a problem:
    since reads must come after the set, and the set will destroy
    whatever were the contents of the update, it can be the case
    that the update is completely hidden and rendered irrelevant.
    There are however at least two reasons to use both a set and an update:

    - to use a signal as scratch space (updating means destroying it)
    - to use sets and updates on partly overlapping views of the same
      memory.

    N.B.: It is done on purpose that there are no default values for
    reads, sets, incs, and updates.

    Each operator should explicitly set each of these properties.
    """

    def __init__(self, tag=None):
        self.tag = tag

    def __repr__(self):
        return "<%s%s at 0x%x>" % (
            self.__class__.__name__, self._tagstr, id(self))

    @property
    def _tagstr(self):
        return ' "%s"' % self.tag if self.tag is not None else ''

    @property
    def reads(self):
        """Signals that are read and not modified"""
        return self._reads

    @reads.setter
    def reads(self, val):
        self._reads = val

    @property
    def sets(self):
        """Signals assigned by this operator

        A signal that is set here cannot be set or updated
        by any other operator.
        """
        return self._sets

    @sets.setter
    def sets(self, val):
        self._sets = val

    @property
    def incs(self):
        """Signals incremented by this operator

        Increments will be applied after this signal has been
        set (if it is set), and before reads.
        """
        return self._incs

    @incs.setter
    def incs(self, val):
        self._incs = val

    @property
    def updates(self):
        """Signals assigned their value for time t + 1

        This operator will be scheduled so that updates appear after
        all sets, increments and reads of this signal.
        """
        return self._updates

    @updates.setter
    def updates(self, val):
        self._updates = val

    @property
    def all_signals(self):
        return self.reads + self.sets + self.incs + self.updates

    def init_signals(self, signals):
        """Initialize simulator.signals

        Install any buffers into the signals view that
        this operator will need. Classes for neurons
        that use extra buffers should create them here.
        """
        for sig in self.all_signals:
            if sig not in signals:
                signals.init(sig)


class TimeUpdate(Operator):
    """Updates the simulation time"""

    def __init__(self, step, time):
        self.step = step
        self.time = time

        self.sets = [step, time]
        self.incs = []
        self.reads = []
        self.updates = []

    def __str__(self):
        return 'TimeUpdate()'

    def make_step(self, signals, dt, rng):
        step = signals[self.step]
        time = signals[self.time]

        def step_timeupdate():
            step[...] += 1
            time[...] = step * dt

        return step_timeupdate


class PreserveValue(Operator):
    """Marks a signal as `set` for the graph checker.

    This operator does no computation. It simply marks a signal as `set`,
    allowing us to apply other ops to signals that we want to preserve their
    value across multiple time steps. It is used primarily for learning rules.
    """
    def __init__(self, dst, tag=None):
        self.dst = dst
        self.tag = tag

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def __str__(self):
        return 'PreserveValue(%s%s)' % (self.dst, self._tagstr)

    def make_step(self, signals, dt, rng):
        def step_preservevalue():
            pass
        return step_preservevalue


class Reset(Operator):
    """Assign a constant value to a Signal."""

    def __init__(self, dst, value=0, tag=None):
        self.dst = dst
        self.value = float(value)
        self.tag = tag

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def __str__(self):
        return 'Reset(%s%s)' % (self.dst, self._tagstr)

    def make_step(self, signals, dt, rng):
        target = signals[self.dst]
        value = self.value

        def step_reset():
            target[...] = value
        return step_reset


class Copy(Operator):
    """Assign the value of one signal to another."""

    def __init__(self, dst, src, tag=None):
        self.dst = dst
        self.src = src
        self.tag = tag

        self.sets = [dst]
        self.incs = []
        self.reads = [src]
        self.updates = []

    def __str__(self):
        return 'Copy(%s -> %s%s)' % (self.src, self.dst, self._tagstr)

    def make_step(self, signals, dt, rng):
        dst = signals[self.dst]
        src = signals[self.src]

        def step_copy():
            dst[...] = src
        return step_copy


class SlicedCopy(Operator):
    """Copy from `a` to `b` with slicing: `b[b_slice] = a[a_slice]`"""
    def __init__(self, a, b, a_slice=Ellipsis, b_slice=Ellipsis,
                 inc=False, tag=None):
        if isinstance(a_slice, slice):
            a = a[a_slice]
            a_slice = Ellipsis
        if isinstance(b_slice, slice):
            b = b[b_slice]
            b_slice = Ellipsis
        # ^ a_slice and b_slice are now either lists of indices or `Ellipsis`

        self.a = a
        self.b = b
        self.a_slice = a_slice
        self.b_slice = b_slice
        self.inc = inc
        self.tag = tag

        self.sets = [] if inc else [b]
        self.incs = [b] if inc else []
        self.reads = [a]
        self.updates = []

    def __str__(self):
        return 'SlicedCopy(%s[%s] -> %s[%s], inc=%s%s)' % (
            self.a, self.a_slice, self.b, self.b_slice, self.inc, self._tagstr)

    def make_step(self, signals, dt, rng):
        a = signals[self.a]
        b = signals[self.b]
        a_slice = self.a_slice
        b_slice = self.b_slice
        inc = self.inc

        def step_slicedcopy():
            if inc:
                b[b_slice] += a[a_slice]
            else:
                b[b_slice] = a[a_slice]
        return step_slicedcopy


class ElementwiseInc(Operator):
    """Increment signal Y by A * X (with broadcasting)"""

    def __init__(self, A, X, Y, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.tag = tag

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    def __str__(self):
        return 'ElementwiseInc(%s, %s -> %s%s)' % (
            str(self.A), str(self.X), str(self.Y), self._tagstr)

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
    """Increment signal Y by dot(A, X)

    Currently, this only supports matrix-vector multiplies for compatibility
    with NengoOCL.
    """

    def __init__(self, A, X, Y, tag=None):
        if X.ndim >= 2 and any(d > 1 for d in X.shape[1:]):
            raise BuildError("X must be a column vector")
        if Y.ndim >= 2 and any(d > 1 for d in Y.shape[1:]):
            raise BuildError("Y must be a column vector")

        self.A = A
        self.X = X
        self.Y = Y
        self.tag = tag

        self.sets = []
        self.incs = [Y]
        self.reads = [A, X]
        self.updates = []

    def __str__(self):
        return 'DotInc(%s, %s -> %s%s)' % (
            self.A, self.X, self.Y, self._tagstr)

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


class SimPyFunc(Operator):
    """Set signal `output` by some Python function of x, possibly t."""

    def __init__(self, output, fn, t, x, tag=None):
        self.output = output
        self.fn = fn
        self.t = t
        self.x = x
        self.tag = tag

        self.sets = [] if output is None else [output]
        self.incs = []
        self.reads = ([] if t is None else [t]) + ([] if x is None else [x])
        self.updates = []

    def __str__(self):
        return "SimPyFunc(%s -> %s, fn='%s'%s)" % (
            self.x, self.output, self.fn.__name__, self._tagstr)

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
