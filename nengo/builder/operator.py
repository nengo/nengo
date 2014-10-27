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


class Operator(object):
    """Base class for operator instances understood by nengo.Simulator.

    The lifetime of a Signal during one simulator timestep:
    0) at most one set operator (optional)
    1) any number of increments
    2) any number of reads
    3) at most one update

    A signal that is only read can be considered a "constant".

    A signal that is both set *and* updated can be a problem:
    since reads must come after the set, and the set will destroy
    whatever were the contents of the update, it can be the case
    that the update is completely hidden and rendered irrelevant.
    There are however at least two reasons to use both a set and an update:
    (a) to use a signal as scratch space (updating means destroying it)
    (b) to use sets and updates on partly overlapping views of the same
        memory.

    N.B.: It is done on purpose that there are no default values for
    reads, sets, incs, and updates.

    Each operator should explicitly set each of these properties.
    """

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
            if sig.base not in signals:
                signals.init(sig.base)


class PreserveValue(Operator):
    """Marks a signal as `set` for the graph checker.

    This is a silly operator that does no computation. It simply marks
    a signal as `set`, allowing us to apply other ops to signals that
    we want to preserve their value across multiple time steps. It is
    used primarily for learning rules.
    """
    def __init__(self, dst):
        self.dst = dst

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            pass
        return step


class Reset(Operator):
    """Assign a constant value to a Signal."""

    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

        self.sets = [dst]
        self.incs = []
        self.reads = []
        self.updates = []

    def __str__(self):
        return 'Reset(%s)' % str(self.dst)

    def make_step(self, signals, dt, rng):
        target = signals[self.dst]
        value = self.value

        def step():
            target[...] = value
        return step


class Copy(Operator):
    """Assign the value of one signal to another."""

    def __init__(self, dst, src, as_update=False, tag=None):
        self.dst = dst
        self.src = src
        self.as_update = as_update
        self.tag = tag

        self.sets = [] if as_update else [dst]
        self.incs = []
        self.reads = [src]
        self.updates = [dst] if as_update else []

    def __str__(self):
        return 'Copy(%s -> %s, as_update=%s)' % (
            str(self.src), str(self.dst), self.as_update)

    def make_step(self, signals, dt, rng):
        dst = signals[self.dst]
        src = signals[self.src]

        def step():
            dst[...] = src
        return step


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
        return 'ElementwiseInc(%s, %s -> %s "%s")' % (
            str(self.A), str(self.X), str(self.Y), self.tag)

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
            assert da in [1, dy] and dx in [1, dy] and max(da, dx) == dy

        def step():
            Y[...] += A * X
        return step


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
        raise ValueError('shape mismatch in %s: %s x %s -> %s' % (
            tag, A.shape, X.shape, Y.shape))

    # Reshape to handle case when np.dot(A, X) and Y are both scalars
    return (np.dot(A, X)).size == Y.size == 1


class DotInc(Operator):
    """Increment signal Y by dot(A, X)

    Currently, this only supports matrix-vector multiplies for compatibility
    with NengoOCL.
    """

    def __init__(self, A, X, Y, as_update=False, tag=None):
        if X.ndim >= 2 and any(d > 1 for d in X.shape[1:]):
            raise ValueError("X must be a column vector")
        if Y.ndim >= 2 and any(d > 1 for d in Y.shape[1:]):
            raise ValueError("Y must be a column vector")

        self.A = A
        self.X = X
        self.Y = Y
        self.as_update = as_update
        self.tag = tag

        self.sets = []
        self.incs = [] if as_update else [Y]
        self.reads = [A, X]
        self.updates = [Y] if as_update else []

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
            self.A, self.X, self.Y, self.tag)

    def make_step(self, signals, dt, rng):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            inc = np.dot(A, X)
            if reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step


class SimNoise(Operator):
    def __init__(self, output, distribution):
        self.output = output
        self.distribution = distribution

        self.sets = []
        self.incs = [output]
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        Y = signals[self.output]
        dist = self.distribution
        n = Y.size
        Yview = Y.reshape(-1)

        def step():
            Yview[...] += dist.sample(n, rng=rng)

        return step
