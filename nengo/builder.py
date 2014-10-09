"""Reference implementation for building a nengo.Network.

Signals and some Operators are adapted from sigops
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


import collections
import logging
import warnings

import numpy as np

from nengo.connection import Connection
from nengo.ensemble import Ensemble, Neurons
from nengo.learning_rules import BCM, Oja, PES
from nengo.network import Network
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate, Direct, LIF, LIFRate
from nengo.node import Node
from nengo.probe import Probe
from nengo.synapses import Alpha, LinearFilter, Lowpass, Synapse
from nengo.utils.distributions import Distribution
from nengo.utils.builder import default_n_eval_points, full_transform
from nengo.utils.compat import is_iterable, is_number, StringIO
from nengo.utils.filter_design import cont2discrete
import nengo.utils.numpy as npext

logger = logging.getLogger(__name__)


class SignalView(object):

    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base is not None
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

    def view_like_self_of(self, newbase, name=None):
        if newbase.base != newbase:
            raise NotImplementedError()
        if newbase.structure != self.base.structure:
            raise NotImplementedError('technically ok but should not happen',
                                      (self.base, newbase))
        return SignalView(newbase,
                          self.shape,
                          self.elemstrides,
                          self.offset,
                          name)

    @property
    def structure(self):
        return (self.shape, self.elemstrides, self.offset)

    def same_view_as(self, other):
        return self.structure == other.structure and self.base == other.base

    @property
    def dtype(self):
        return np.dtype(self.base.dtype)

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
                raise ValueError(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        elif self.size == 1:
            # -- scalars can be reshaped to any number of (1, 1, 1...)
            size = int(np.prod(shape))
            if size != self.size:
                raise ValueError(shape, self.shape)
            elemstrides = [1] * len(shape)
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
            raise NotImplementedError('reshape of strided view')

    def transpose(self, neworder=None):
        if neworder:
            raise NotImplementedError()
        return SignalView(
            self.base,
            reversed(self.shape),
            reversed(self.elemstrides),
            self.offset,
            self.name + '.T'
        )

    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

    def __getitem__(self, item):  # noqa: C901
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
                return 'View(%s[%d])' % (self.base.name, self.offset)

    @name.setter
    def name(self, value):
        self._name = value

    def is_contiguous(self):
        shape, strides, offset = self.structure
        if len(shape) == 0:
            return True, offset, offset + 1
        elif len(shape) == 1:
            if strides[0] == 1:
                return True, offset, offset + shape[0]
            else:
                return False, None, None
        elif len(shape) == 2:
            if strides == (1, shape[0]) or strides == (shape[1], 1):
                return True, offset, offset + shape[0] * shape[1]
            else:
                return False, None, None
        else:
            raise NotImplementedError()
        # if self.ndim == 1 and self.elemstrides[0] == 1:
            # return self.offset, self.offset + self.size

    def shares_memory_with(self, other):  # noqa: C901
        # TODO: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        if self.base is not other.base or self.size == 0 or other.size == 0:
            return False
        elif self is other or self.same_view_as(other):
            return True
        elif self.ndim < other.ndim:
            return other.shares_memory_with(self)

        assert self.ndim > 0
        if self.ndim == 1:
            # -- self is a vector view
            #    and other is either a scalar or vector view
            ae0, = self.elemstrides
            be0, = other.elemstrides
            amin = self.offset
            amax = amin + self.shape[0] * ae0
            bmin = other.offset
            bmax = bmin + other.shape[0] * be0
            if amin <= amax <= bmin <= bmax or bmin <= bmax <= amin <= amax:
                return False
            elif ae0 == be0 == 1:
                # -- strides are equal, and we've already checked for
                #    non-overlap. They do overlap, so they are aliased.
                return True
            # TODO: look for common divisor of ae0 and be0
            raise NotImplementedError('1d', (self.structure, other.structure))
        elif self.ndim == 2:
            # -- self is a matrix view
            #    and other is either a scalar, vector or matrix view
            a_contig, amin, amax = self.is_contiguous()
            b_contig, bmin, bmax = other.is_contiguous()

            if a_contig and b_contig:
                # -- both have a contiguous memory layout,
                #    from min up to but not including max
                return (not (amin <= amax <= bmin <= bmax)
                        and not (bmin <= bmax <= amin <= amax))
            elif a_contig:
                # -- only a contiguous
                raise NotImplementedError('2d self:contig, other:discontig',
                                          (self.structure, other.structure))
            else:
                raise NotImplementedError('2d',
                                          (self.structure, other.structure))
        raise NotImplementedError()

    @property
    def value(self):
        """Returns a view on the base array's value."""
        try:
            # for some installations, this works
            itemsize = int(self.dtype.itemsize)
        except TypeError:
            # other installations, this...
            itemsize = int(self.dtype().itemsize)
        byteoffset = itemsize * self.offset
        bytestrides = [itemsize * s for s in self.elemstrides]
        view = np.ndarray(shape=self.shape,
                          dtype=self.dtype,
                          buffer=self.base._value.data,
                          offset=byteoffset,
                          strides=bytestrides)
        return view


class Signal(SignalView):
    """Interpretable, vector-valued quantity within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None):
        self._value = np.asarray(value, dtype=np.float64)
        if name is not None:
            self._name = name
        if Signal.assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", shape=" + str(self.shape) + ")"
        except AttributeError:
            return ("Signal(id " + str(id(self)) + ", shape="
                    + str(self.shape) + ")")

    def __repr__(self):
        return str(self)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(int(si / self.dtype.itemsize) for si in s)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

    @property
    def value(self):
        return self._value


class SignalDict(dict):
    """Map from Signal -> ndarray

    This dict subclass ensures that the ndarray values aren't overwritten,
    and instead data are written into them, which ensures that
    these arrays never get copied, which wastes time and space.

    Use ``init`` to set the ndarray initially.
    """

    def __getitem__(self, obj):
        """SignalDict overrides __getitem__ for two reasons.

        1. so that scalars are returned as 0-d ndarrays
        2. so that a SignalView lookup returns a views of its base
        """
        if obj in self:
            return dict.__getitem__(self, obj)
        elif obj.base in self:
            # look up views as a fallback
            # --work around numpy's special case behaviour for scalars
            base_array = self[obj.base]
            try:
                # for some installations, this works
                itemsize = int(obj.dtype.itemsize)
            except TypeError:
                # other installations, this...
                itemsize = int(obj.dtype().itemsize)
            byteoffset = itemsize * obj.offset
            bytestrides = [itemsize * s for s in obj.elemstrides]
            view = np.ndarray(shape=obj.shape,
                              dtype=obj.dtype,
                              buffer=base_array.data,
                              offset=byteoffset,
                              strides=bytestrides)
            return view
        else:
            raise KeyError("%s has not been initialized. Please call "
                           "SignalDict.init first." % (str(obj)))

    def __setitem__(self, key, val):
        """Ensures that ndarrays stay in the same place in memory.

        Unlike normal dicts, this means that you cannot add a new key
        to a SignalDict using __setitem__. This is by design, to avoid
        silent typos when debugging Simulator. Every key must instead
        be explicitly initialized with SignalDict.init.
        """
        self.__getitem__(key)[...] = val

    def __str__(self):
        """Pretty-print the signals and current values."""
        sio = StringIO()
        for k in self:
            sio.write("%s %s\n" % (repr(k), repr(self[k])))
        return sio.getvalue()

    def init(self, signal):
        """Set up a permanent mapping from signal -> ndarray."""
        # Make a copy of base.value to start
        dict.__setitem__(self, signal.base, np.array(signal.base.value))

    def reset(self, signal):
        """Reset ndarray to the base value of the signal that maps to it"""
        self[signal] = signal.value


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

    def init_signals(self, signals, dt):
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

    def make_step(self, signals, dt):
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

    def make_step(self, signals, dt):
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

    def make_step(self, signals, dt):
        dst = signals[self.dst]
        src = signals[self.src]

        def step():
            dst[...] = src
        return step


class ElementwiseInc(Operator):
    """Increment signal Y by A * X"""

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

    def make_step(self, signals, dt):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        if X.shape != Y.shape:
            raise ValueError("Shape mismatch in %s: %s != %s"
                             % (self.tag, X.shape, Y.shape))
        if A.size > 1 and A.shape != X.shape:
            raise ValueError("Shape mismatch in %s: %s != %s"
                             % (self.tag, A.shape, X.shape))

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
    """Increment signal Y by dot(A, X)"""

    def __init__(self, A, X, Y, as_update=False, tag=None):
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

    def make_step(self, signals, dt):
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


class SimPyFunc(Operator):
    """Set signal `output` by some non-linear function of x, possibly t"""

    def __init__(self, output, fn, t_in, x):
        self.output = output
        self.fn = fn
        self.t_in = t_in
        self.x = x

        self.sets = [] if output is None else [output]
        self.incs = []
        self.reads = [] if x is None else [x]
        self.updates = []

    def __str__(self):
        return "SimPyFunc(%s -> %s '%s')" % (self.x, self.output, self.fn)

    def make_step(self, signals, dt):
        if self.output is not None:
            output = signals[self.output]
        fn = self.fn
        args = [signals['__time__']] if self.t_in else []
        args += [signals[self.x]] if self.x is not None else []

        def step():
            y = fn(*args)
            if self.output is not None:
                if y is None:
                    raise ValueError(
                        "Function '%s' returned invalid value" % fn.__name__)
                output[...] = y

        return step


class SimNeurons(Operator):
    """Set output to neuron model output for the given input current."""

    def __init__(self, neurons, J, output, states=[]):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.states = states

        self.sets = [output] + states
        self.incs = []
        self.reads = [J]
        self.updates = []

    def make_step(self, signals, dt):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step():
            self.neurons.step_math(dt, J, output, *states)
        return step


class SimFilterSynapse(Operator):
    """Simulate a discrete-time LTI system.

    Implements a discrete-time LTI system using the difference equation [1]_
    for the given transfer function (num, den).

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Digital_filter#Difference_equation
    """
    def __init__(self, input, output, num, den):
        self.input = input
        self.output = output
        self.num = num
        self.den = den

        self.sets = []
        self.incs = []
        self.reads = [input]
        self.updates = [output]

    def make_step(self, signals, dt):
        input = signals[self.input]
        output = signals[self.output]
        num, den = self.num, self.den

        if len(num) == 1 and len(den) == 0:
            def step(input=input, output=output, b=num[0]):
                output[:] = b * input
        elif len(num) == 1 and len(den) == 1:
            def step(input=input, output=output, a=den[0], b=num[0]):
                output *= -a
                output += b * input
        else:
            x = collections.deque(maxlen=len(num))
            y = collections.deque(maxlen=len(den))

            def step(input=input, output=output, x=x, y=y, num=num, den=den):
                output[:] = 0

                x.appendleft(np.array(input))
                for k, xk in enumerate(x):
                    output += num[k] * xk
                for k, yk in enumerate(y):
                    output -= den[k] * yk
                y.appendleft(np.array(output))

        return step


class SimBCM(Operator):
    """Change the transform according to the BCM rule."""
    def __init__(self, pre_filtered, post_filtered, theta, delta,
                 learning_rate):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.theta = theta
        self.delta = delta
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta]
        self.updates = [delta]

    def make_step(self, signals, dt):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta), pre_filtered)
        return step


class SimOja(Operator):
    """
    Change the transform according to the OJA rule
    """
    def __init__(self, pre_filtered, post_filtered, transform, delta,
                 learning_rate, beta):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.transform = transform
        self.delta = delta
        self.learning_rate = learning_rate
        self.beta = beta

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, transform]
        self.updates = [delta]

    def make_step(self, signals, dt):
        transform = signals[self.transform]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate
        beta = self.beta

        def step():
            # perform forgetting
            post_squared = alpha * post_filtered * post_filtered
            delta[...] = -beta * transform * post_squared[:, None]

            # perform update
            delta[...] += np.outer(alpha * post_filtered, pre_filtered)

        return step


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt=0.001, label=None):
        # We want to keep track of the toplevel network
        self.toplevel = None

        # Resources used by the build process.
        self.operators = []
        self.params = {}
        self.seeds = {}
        self.probes = []
        self.sig = collections.defaultdict(dict)

        self.dt = dt
        self.label = label

    def __str__(self):
        return "Model: %s" % self.label

    def add_op(self, op):
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        op.init_signals(signals, self.dt)
        op.make_step(signals, self.dt)

    def has_built(self, obj):
        """Returns true iff obj has been processed by build."""
        return obj in self.params

BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform', 'solver_info'])
BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble', ['eval_points', 'encoders', 'intercepts', 'max_rates',
                      'scaled_encoders', 'gain', 'bias'])


class Builder(object):
    builders = {}

    @classmethod
    def register_builder(cls, build_fn, nengo_class):
        cls.builders[nengo_class] = build_fn

    @classmethod
    def build(cls, obj, *args, **kwargs):
        model = kwargs.setdefault('model', Model())

        if model.has_built(obj):
            # TODO: Prevent this at pre-build validation time.
            warnings.warn("Object %s has already been built." % obj)
            return

        for obj_cls in obj.__class__.__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise TypeError("Cannot build object of type '%s'." %
                            obj.__class__.__name__)
        cls.builders[obj_cls](obj, *args, **kwargs)
        return model


def build_network(network, model):  # noqa: C901
    """Takes a Network object and returns a Model.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

    1) Ensembles, Nodes, Neurons
    2) Subnetworks (recursively)
    3) Connections
    4) Learning Rules
    5) Probes
    """
    def get_seed(obj, rng):
        # Generate a seed no matter what, so that setting a seed or not on
        # one object doesn't affect the seeds of other objects.
        seed = rng.randint(npext.maxint)
        return (seed if not hasattr(obj, 'seed') or obj.seed is None
                else obj.seed)

    if model.toplevel is None:
        model.toplevel = network
        model.sig['common'][0] = Signal(0.0, name='Common: Zero')
        model.sig['common'][1] = Signal(1.0, name='Common: One')
        model.seeds[network] = get_seed(network, np.random)

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    sorted_types = sorted(list(network.objects.keys()),
                          key=lambda t: t.__name__)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeds[obj] = get_seed(obj, rng)

    logger.info("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        Builder.build(obj, model=model, config=network.config)

    logger.info("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        Builder.build(subnetwork, model=model)

    logger.info("Network step 3: Building connections")
    for conn in network.connections:
        Builder.build(conn, model=model, config=network.config)

    logger.info("Network step 4: Building learning rules")
    for conn in network.connections:
        if is_iterable(conn.learning_rule):
            for learning_rule in conn.learning_rule:
                Builder.build(learning_rule, conn,
                              model=model, config=network.config)
        elif conn.learning_rule is not None:
            Builder.build(conn.learning_rule, conn,
                          model=model, config=network.config)

    logger.info("Network step 5: Building probes")
    for probe in network.probes:
        Builder.build(probe, model=model, config=network.config)

    model.params[network] = None

Builder.register_builder(build_network, Network)


def sample(dist, n_samples, rng):
    if isinstance(dist, Distribution):
        return dist.sample(n_samples, rng=rng)
    return np.array(dist)


def build_ensemble(ens, model, config):  # noqa: C901
    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    # Generate eval points
    if isinstance(ens.eval_points, Distribution):
        n_points = ens.n_eval_points
        if n_points is None:
            n_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
        eval_points = ens.eval_points.sample(n_points, ens.dimensions, rng)
        # eval_points should be in the ensemble's representational range
        eval_points *= ens.radius
    else:
        if (ens.n_eval_points is not None
                and ens.eval_points.shape[0] != ens.n_eval_points):
            warnings.warn("Number of eval_points doesn't match "
                          "n_eval_points. Ignoring n_eval_points.")
        eval_points = np.array(ens.eval_points, dtype=np.float64)

    # Set up signal
    model.sig[ens]['in'] = Signal(np.zeros(ens.dimensions),
                                  name="%s.signal" % ens)
    model.add_op(Reset(model.sig[ens]['in']))

    # Set up encoders
    if isinstance(ens.neuron_type, Direct):
        encoders = np.identity(ens.dimensions)
    elif isinstance(ens.encoders, Distribution):
        encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Determine max_rates and intercepts
    max_rates = sample(ens.max_rates, ens.n_neurons, rng=rng)
    intercepts = sample(ens.intercepts, ens.n_neurons, rng=rng)

    # Build the neurons
    if ens.gain is not None and ens.bias is not None:
        gain = sample(ens.gain, ens.n_neurons, rng=rng)
        bias = sample(ens.bias, ens.n_neurons, rng=rng)
    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError("gain or bias set for %s, but not both. "
                                  "Solving for one given the other is not "
                                  "implemented yet." % ens)
    else:
        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)

    if isinstance(ens.neuron_type, Direct):
        model.sig[ens.neurons]['in'] = Signal(
            np.zeros(ens.dimensions), name='%s.neuron_in' % ens)
        model.sig[ens.neurons]['out'] = model.sig[ens.neurons]['in']
        model.add_op(Reset(model.sig[ens.neurons]['in']))
    else:
        model.sig[ens.neurons]['in'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_in" % ens)
        model.sig[ens.neurons]['out'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_out" % ens)
        model.add_op(Copy(src=Signal(bias, name="%s.bias" % ens),
                          dst=model.sig[ens.neurons]['in']))
        # This adds the neuron's operator and sets other signals
        Builder.build(ens.neuron_type, ens.neurons, model=model, config=config)

    # Scale the encoders
    if isinstance(ens.neuron_type, Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    model.sig[ens]['encoders'] = Signal(
        scaled_encoders, name="%s.scaled_encoders" % ens)

    # Create output signal, using built Neurons
    model.add_op(DotInc(
        model.sig[ens]['encoders'],
        model.sig[ens]['in'],
        model.sig[ens.neurons]['in'],
        tag="%s encoding" % ens))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens.neurons]['out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)

Builder.register_builder(build_ensemble, Ensemble)


def build_lifrate(lifrate, neurons, model, config):
    model.add_op(SimNeurons(neurons=lifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))

Builder.register_builder(build_lifrate, LIFRate)


def build_lif(lif, neurons, model, config):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))

Builder.register_builder(build_lif, LIF)


def build_alifrate(alifrate, neurons, model, config):
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['adaptation']]))

Builder.register_builder(build_alifrate, AdaptiveLIFRate)


def build_alif(alif, neurons, model, config):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['refractory_time'],
                                    model.sig[neurons]['adaptation']]))

Builder.register_builder(build_alif, AdaptiveLIF)


def build_node(node, model, config):
    # Get input
    if node.output is None or callable(node.output):
        if node.size_in > 0:
            model.sig[node]['in'] = Signal(
                np.zeros(node.size_in), name="%s.signal" % node)
            # Reset input signal to 0 each timestep
            model.add_op(Reset(model.sig[node]['in']))

    # Provide output
    if node.output is None:
        model.sig[node]['out'] = model.sig[node]['in']
    elif not callable(node.output):
        model.sig[node]['out'] = Signal(node.output, name=str(node))
    else:
        sig_in, sig_out = build_pyfunc(fn=node.output,
                                       t_in=True,
                                       n_in=node.size_in,
                                       n_out=node.size_out,
                                       label="%s.pyfn" % node,
                                       model=model)
        if sig_in is not None:
            model.add_op(DotInc(model.sig[node]['in'],
                                model.sig['common'][1],
                                sig_in,
                                tag="%s input" % node))
        if sig_out is not None:
            model.sig[node]['out'] = sig_out

    model.params[node] = None

Builder.register_builder(build_node, Node)


def conn_probe(probe, model, config):
    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeds[conn] = model.seeds[probe]

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    Builder.build(conn, model=model, config=config)


def synapse_probe(key, probe, model, config):
    sig = model.sig[probe.obj][key]
    if isinstance(probe.slice, slice):
        sig = sig[probe.slice]
    else:
        raise NotImplementedError("Indexing slices not implemented")

    if probe.synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['in'] = filtered_signal(
            probe, sig, probe.synapse, model=model, config=config)

probemap = {
    Ensemble: {'decoded_output': None,
               'input': 'in'},
    Neurons: {'output': None,
              'spikes': None,
              'rates': None,
              'input': 'in'},
    Node: {'output': None},
    Connection: {'output': 'out',
                 'input': 'in'},
}


def build_probe(probe, model, config):
    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in probemap.items():
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise ValueError("Type '%s' is not probeable" % type(probe.obj))

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(probe, model, config)
    else:
        synapse_probe(key, probe, model, config)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []

Builder.register_builder(build_probe, Probe)


def build_linear_system(conn, model):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = conn.eval_points
    if eval_points is None:
        eval_points = npext.array(
            model.params[conn.pre_obj].eval_points, min_dims=2)
    else:
        eval_points = npext.array(eval_points, min_dims=2)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    activities = model.dt * conn.pre_obj.neuron_type.rates(x, gain, bias)
    if np.count_nonzero(activities) == 0:
        raise RuntimeError(
            "Building %s: 'activites' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            targets[i] = conn.function(ep)

    return eval_points, activities, targets


def build_connection(conn, model, config):  # noqa: C901
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get input and output connections from pre and post
    def get_prepost_signal(is_pre):
        target = conn.pre_obj if is_pre else conn.post_obj
        key = 'out' if is_pre else 'in'

        if target not in model.sig:
            raise ValueError("Building %s: the '%s' object %s "
                             "is not in the model, or has a size of zero."
                             % (conn, 'pre' if is_pre else 'post', target))
        if key not in model.sig[target]:
            raise ValueError("Error building %s: the '%s' object %s "
                             "has a '%s' size of zero." %
                             (conn, 'pre' if is_pre else 'post', target, key))

        return model.sig[target][key]

    model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
    model.sig[conn]['out'] = get_prepost_signal(is_pre=False)

    decoders = None
    eval_points = None
    solver_info = None
    transform = full_transform(conn, slice_pre=False)

    # Figure out the signal going across this connection
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        if (conn.function is None and isinstance(conn.pre_slice, slice) and
                (conn.pre_slice.step is None or conn.pre_slice.step == 1)):
            signal = model.sig[conn]['in'][conn.pre_slice]
        else:
            sig_in, signal = build_pyfunc(
                fn=(lambda x: x[conn.pre_slice]) if conn.function is None else
                   (lambda x: conn.function(x[conn.pre_slice])),
                t_in=False,
                n_in=model.sig[conn]['in'].size,
                n_out=conn.size_mid,
                label=str(conn),
                model=model)
            model.add_op(DotInc(model.sig[conn]['in'],
                                model.sig['common'][1],
                                sig_in,
                                tag="%s input" % conn))
    elif isinstance(conn.pre_obj, Ensemble):
        # Normal decoded connection
        eval_points, activities, targets = build_linear_system(conn, model)

        if conn.solver.weights:
            # account for transform
            targets = np.dot(targets, transform.T)
            transform = np.array(1., dtype=np.float64)

            decoders, solver_info = conn.solver(
                activities, targets, rng=rng,
                E=model.params[conn.post_obj].scaled_encoders.T)
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = model.sig[conn]['out'].size
        else:
            decoders, solver_info = conn.solver(activities, targets, rng=rng)
            signal_size = conn.size_mid

        # Add operator for decoders
        decoders = decoders.T

        model.sig[conn]['decoders'] = Signal(
            decoders, name="%s.decoders" % conn)
        signal = Signal(np.zeros(signal_size), name=str(conn))
        model.add_op(Reset(signal))
        model.add_op(DotInc(model.sig[conn]['decoders'],
                            model.sig[conn]['in'],
                            signal,
                            tag="%s decoding" % conn))
    else:
        # Direct connection
        signal = model.sig[conn]['in']

    # Add operator for filtering
    if conn.synapse is not None:
        signal = filtered_signal(conn, signal, conn.synapse, model, config)

    if conn.modulatory:
        # Make a new signal, effectively detaching from post
        model.sig[conn]['out'] = Signal(
            np.zeros(model.sig[conn]['out'].size),
            name="%s.mod_output" % conn)
        model.add_op(Reset(model.sig[conn]['out']))

    # Add operator for transform
    if isinstance(conn.post_obj, Neurons):
        if not model.has_built(conn.post_obj.ensemble):
            # Since it hasn't been built, it wasn't added to the Network,
            # which is most likely because the Neurons weren't associated
            # with an Ensemble.
            raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                               "that are not a part of any Ensemble." % (
                                   conn, conn.post_obj))

        if conn.post_slice != slice(None):
            raise NotImplementedError(
                "Post-slices on connections to neurons are not implemented")

        gain = model.params[conn.post_obj.ensemble].gain[conn.post_slice]
        if transform.ndim < 2:
            transform = transform * gain
        else:
            transform *= gain[:, np.newaxis]

    model.sig[conn]['transform'] = Signal(transform,
                                          name="%s.transform" % conn)
    if transform.ndim < 2:
        model.add_op(ElementwiseInc(model.sig[conn]['transform'],
                                    signal,
                                    model.sig[conn]['out'],
                                    tag=str(conn)))
    else:
        model.add_op(DotInc(model.sig[conn]['transform'],
                            signal,
                            model.sig[conn]['out'],
                            tag=str(conn)))

    if conn.learning_rule:
        # Forcing update of signal that is modified by learning rules.
        # Learning rules themselves apply DotIncs.

        if isinstance(conn.pre_obj, Neurons):
            modified_signal = model.sig[conn]['transform']
        elif isinstance(conn.pre_obj, Ensemble):
            if conn.solver.weights:
                # TODO: make less hacky.
                # Have to do this because when a weight_solver
                # is provided, then learning rules should operators on
                # "decoders" which is really the weight matrix.
                model.sig[conn]['transform'] = model.sig[conn]['decoders']
                modified_signal = model.sig[conn]['transform']
            else:
                modified_signal = model.sig[conn]['decoders']
        else:
            raise TypeError("Can't apply learning rules to connections of "
                            "this type. pre type: %s, post type: %s"
                            % (type(conn.pre_obj).__name__,
                               type(conn.post_obj).__name__))

        model.add_op(PreserveValue(modified_signal))

    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)

Builder.register_builder(build_connection, Connection)


def filtered_signal(owner, sig, synapse, model, config):
    # Note: we add a filter here even if synapse < dt,
    # in order to avoid cycles in the op graph. If the filter
    # is explicitly set to None (e.g. for a passthrough node)
    # then cycles can still occur.
    if is_number(synapse):
        synapse = Lowpass(synapse)
    assert isinstance(synapse, Synapse)
    Builder.build(synapse, owner, sig, model=model, config=config)
    return model.sig[owner]['synapse_out']


def build_pyfunc(fn, t_in, n_in, n_out, label, model):
    if n_in:
        sig_in = Signal(np.zeros(n_in), name="%s.input" % label)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    if n_out > 0:
        sig_out = Signal(np.zeros(n_out), name="%s.output" % label)
    else:
        sig_out = None

    model.add_op(SimPyFunc(output=sig_out, fn=fn, t_in=t_in, x=sig_in))

    return sig_in, sig_out


def build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config):
    model.sig[owner]['synapse_in'] = input_signal
    model.sig[owner]['synapse_out'] = Signal(
        np.zeros(input_signal.size),
        name="%s.%s" % (input_signal.name, synapse))

    model.add_op(SimFilterSynapse(input=model.sig[owner]['synapse_in'],
                                  output=model.sig[owner]['synapse_out'],
                                  num=num, den=den))


def build_filter_synapse(synapse, owner, input_signal, model, config):
    num, den, _ = cont2discrete(
        (synapse.num, synapse.den), model.dt, method='zoh')
    num = num.flatten()
    num = num[1:] if num[0] == 0 else num
    den = den[1:]  # drop first element (equal to 1)
    build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config)

Builder.register_builder(build_filter_synapse, LinearFilter)


def build_lowpass_synapse(synapse, owner, input_signal, model, config):
    if synapse.tau > 0.03 * model.dt:
        d = -np.expm1(-model.dt / synapse.tau)
        num, den = [d], [d - 1]
    else:
        num, den = [1.], []

    build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config)

Builder.register_builder(build_lowpass_synapse, Lowpass)


def build_alpha_synapse(synapse, owner, input_signal, model, config):
    if synapse.tau > 0.03 * model.dt:
        a = model.dt / synapse.tau
        ea = np.exp(-a)
        num, den = [-a*ea + (1 - ea), ea*(a + ea - 1)], [-2 * ea, ea**2]
    else:
        num, den = [1.], []  # just copy the input

    build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config)

Builder.register_builder(build_alpha_synapse, Alpha)


def build_pes(pes, conn, model, config):
    if isinstance(conn.pre_obj, Neurons):
        activities = model.sig[conn.pre_obj.ensemble]['out']
    else:
        activities = model.sig[conn.pre_obj]['out']
    error = model.sig[pes.error_connection]['out']

    scaled_error = Signal(np.zeros(error.shape),
                          name="PES:error * learning rate")
    scaled_error_view = SignalView(scaled_error, (error.size, 1), (1, 1), 0,
                                   name="PES:scaled_error_view")
    activities_view = SignalView(activities, (1, activities.size), (1, 1), 0,
                                 name="PES:activities_view")

    lr_sig = Signal(pes.learning_rate * model.dt, name="PES:learning_rate")

    model.add_op(Reset(scaled_error))
    model.add_op(DotInc(lr_sig, error, scaled_error, tag="PES:scale error"))

    if conn.solver.weights or isinstance(conn.pre_obj, Neurons):
        outer_product = Signal(np.zeros((error.size, activities.size)),
                               name="PES: outer prod")
        transform = model.sig[conn]['transform']
        if isinstance(conn.post_obj, Neurons):
            encoders = model.sig[conn.post_obj.ensemble]['encoders']
        else:
            encoders = model.sig[conn.post_obj]['encoders']

        model.add_op(Reset(outer_product))
        model.add_op(DotInc(scaled_error_view, activities_view, outer_product,
                            tag="PES:Outer Prod"))
        model.add_op(DotInc(encoders, outer_product, transform,
                            tag="PES:Inc Decoder"))

    else:
        assert isinstance(conn.pre_obj, Ensemble)
        decoders = model.sig[conn]['decoders']

        model.add_op(DotInc(scaled_error_view, activities_view, decoders,
                            tag="PES:Inc Decoder"))

    model.params[pes] = None

Builder.register_builder(build_pes, PES)


def build_bcm(bcm, conn, model, config):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    transform = model.sig[conn]['transform']
    pre_activities = model.sig[pre.neurons]['out']
    post_activities = model.sig[post.neurons]['out']
    pre_filtered = filtered_signal(
        bcm, pre_activities, bcm.pre_tau, model, config)
    post_filtered = filtered_signal(
        bcm, post_activities, bcm.post_tau, model, config)
    theta = filtered_signal(
        bcm, post_filtered, bcm.theta_tau, model, config)
    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)),
                   name='BCM: Delta')

    model.add_op(SimBCM(pre_filtered, post_filtered, theta, delta,
                        learning_rate=bcm.learning_rate))
    model.add_op(DotInc(model.sig['common'][1], delta, transform,
                        tag="BCM: DotInc"))


Builder.register_builder(build_bcm, BCM)


def build_oja(oja, conn, model, config):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    transform = model.sig[conn]['transform']
    pre_activities = model.sig[pre.neurons]['out']
    post_activities = model.sig[post.neurons]['out']
    pre_filtered = filtered_signal(
        oja, pre_activities, oja.pre_tau, model, config)
    post_filtered = filtered_signal(
        oja, post_activities, oja.post_tau, model, config)
    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)),
                   name='Oja: Delta')

    model.add_op(SimOja(pre_filtered, post_filtered, transform, delta,
                        learning_rate=oja.learning_rate, beta=oja.beta))
    model.add_op(DotInc(model.sig['common'][1], delta, transform,
                        tag="Oja: DotInc"))

    model.params[oja] = None

Builder.register_builder(build_oja, Oja)
