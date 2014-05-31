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

import nengo.decoders
import nengo.neurons
import nengo.objects
import nengo.synapses
import nengo.utils.distributions as dists
import nengo.utils.numpy as npext
from nengo.utils.compat import is_callable, is_integer, is_number, StringIO
from nengo.utils.filter_design import cont2discrete

logger = logging.getLogger(__name__)


class ShapeMismatch(ValueError):
    pass


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
                raise ShapeMismatch(shape, self.shape)
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
                raise ShapeMismatch(shape, self.shape)
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


class Signal(SignalView):
    """Interpretable, vector-valued quantity within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None):
        self.value = np.asarray(value, dtype=np.float64)
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

    def init(self, signal, ndarray):
        """Set up a permanent mapping from signal -> ndarray."""
        dict.__setitem__(self, signal, ndarray)


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
                signals.init(sig.base,
                             np.asarray(
                                 np.zeros(sig.base.shape,
                                          dtype=sig.base.dtype)
                                 + sig.base.value))


class Reset(Operator):
    """Assign a constant value to a Signal."""

    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

        self.reads = []
        self.incs = []
        self.updates = []
        self.sets = [dst]

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
        self.tag = tag
        self.as_update = True

        self.reads = [src]
        self.sets = [] if as_update else [dst]
        self.updates = [dst] if as_update else []
        self.incs = []

    def __str__(self):
        return 'Copy(%s -> %s, as_update=%s)' % (
            str(self.src), str(self.dst), self.as_update)

    def make_step(self, signals, dt):
        dst = signals[self.dst]
        src = signals[self.src]

        def step():
            dst[...] = src
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

    # If the result is scalar, we'll reshape it so Y[...] += inc works
    return incshape == ()


class DotInc(Operator):
    """Increment signal Y by dot(A, X)"""

    def __init__(self, A, X, Y, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.tag = tag

        self.reads = [self.A, self.X]
        self.incs = [self.Y]
        self.sets = []
        self.updates = []

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
            str(self.A), str(self.X), str(self.Y), self.tag)

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


class ProdUpdate(Operator):
    """Sets Y <- dot(A, X) + B * Y"""

    def __init__(self, A, X, B, Y, tag=None):
        self.A = A
        self.X = X
        self.B = B
        self.Y = Y
        self.tag = tag

        self.reads = [self.A, self.X, self.B]
        self.updates = [self.Y]
        self.incs = []
        self.sets = []

    def __str__(self):
        return 'ProdUpdate(%s, %s, %s, -> %s "%s")' % (
            str(self.A), str(self.X), str(self.B), str(self.Y), self.tag)

    def make_step(self, signals, dt):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        B = signals[self.B]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            val = np.dot(A, X)
            if reshape:
                val = np.asarray(val).reshape(Y.shape)
            Y[...] *= B
            Y[...] += val
        return step


class SimPyFunc(Operator):
    """Set signal `output` by some non-linear function of x, possibly t"""

    def __init__(self, output, fn, t_in, x):
        self.output = output
        self.fn = fn
        self.t_in = t_in
        self.x = x

        self.reads = [] if x is None else [x]
        self.updates = [] if output is None else [output]
        self.sets = []
        self.incs = []

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

        self.reads = [J]
        self.updates = [output] + states
        self.sets = []
        self.incs = []

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

        self.reads = [input]
        self.updates = [output]
        self.sets = []
        self.incs = []

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


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt=0.001, label=None, seed=None):
        # We want to keep track of the toplevel network
        self.toplevel = None

        # Resources used by the build process.
        self.operators = []
        self.params = {}
        self.probes = []
        self.sig = collections.defaultdict(dict)

        self.dt = dt
        self.label = label
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.rng = np.random.RandomState(self.seed)

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

    def next_seed(self):
        """Yields a seed to use for RNG during build computations."""
        return self.rng.randint(npext.maxint)


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
            # If we've already built the obj, we'll ignore it.

            # TODO: Prevent this at pre-build validation time.
            warnings.warn("Object '%s' has already been built." % obj)
            return

        for obj_cls in obj.__class__.__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise TypeError("Cannot build object of type '%s'." %
                            obj.__class__.__name__)
        cls.builders[obj_cls](obj, *args, **kwargs)
        # if obj not in model.params:
        #     raise RuntimeError(
        #         "Build function '%s' did not add '%s' to model.params"
        #         % (cls.builders[obj_cls].__name__, str(obj)))
        return model


def build_network(network, model):
    """Takes a Network object and returns a Model.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

    1) Ensembles, Nodes, Neurons
    2) Subnetworks (recursively)
    3) Connections
    4) Probes
    """
    if model.toplevel is None:
        model.toplevel = network

    logger.info("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        Builder.build(obj, model=model, config=network.config)

    logger.info("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        Builder.build(subnetwork, model=model)

    logger.info("Network step 3: Building connections")
    for conn in network.connections:
        Builder.build(conn, model=model, config=network.config)

    logger.info("Network step 4: Building probes")
    for probe in network.probes:
        Builder.build(probe, model=model, config=network.config)

    model.params[network] = None

Builder.register_builder(build_network, nengo.objects.Network)


def pick_eval_points(ens, n_points, rng):
    if n_points is None:
        # use a heuristic to pick the number of points
        dims, neurons = ens.dimensions, ens.n_neurons
        n_points = max(np.clip(500 * dims, 750, 2500), 2 * neurons)
    return dists.UniformHypersphere(ens.dimensions).sample(
        n_points, rng=rng) * ens.radius


def build_ensemble(ens, model, config):  # noqa: C901
    # Create random number generator
    seed = model.next_seed() if ens.seed is None else ens.seed
    rng = np.random.RandomState(seed)

    # Generate eval points
    if ens.eval_points is None or is_integer(ens.eval_points):
        eval_points = pick_eval_points(
            ens=ens, n_points=ens.eval_points, rng=rng)
    else:
        eval_points = npext.array(
            ens.eval_points, dtype=np.float64, min_dims=2)

    # Set up signal
    model.sig[ens]['in'] = Signal(np.zeros(ens.dimensions),
                                  name="%s.signal" % ens.label)
    model.add_op(Reset(model.sig[ens]['in']))

    # Set up encoders
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        encoders = np.identity(ens.dimensions)
    elif ens.encoders is None:
        sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
        encoders = sphere.sample(ens.n_neurons, rng=rng)
    else:
        encoders = np.array(ens.encoders, dtype=np.float64)
        enc_shape = (ens.n_neurons, ens.dimensions)
        if encoders.shape != enc_shape:
            raise ShapeMismatch(
                "Encoder shape is %s. Should be (n_neurons, dimensions); "
                "in this case %s." % (encoders.shape, enc_shape))
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Determine max_rates and intercepts
    if isinstance(ens.max_rates, dists.Distribution):
        max_rates = ens.max_rates.sample(ens.n_neurons, rng=rng)
    else:
        max_rates = np.array(ens.max_rates)
    if isinstance(ens.intercepts, dists.Distribution):
        intercepts = ens.intercepts.sample(ens.n_neurons, rng=rng)
    else:
        intercepts = np.array(ens.intercepts)

    # Build the neurons
    gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.dimensions), name='%s.neuron_in' % ens.label)
        model.sig[ens]['neuron_out'] = model.sig[ens]['neuron_in']
        model.add_op(Reset(model.sig[ens]['neuron_in']))
    else:
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_in" % ens.label)
        model.sig[ens]['neuron_out'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_out" % ens.label)
        model.add_op(Copy(src=Signal(bias, name="%s.bias" % ens.label),
                          dst=model.sig[ens]['neuron_in']))
        # This adds the neuron's operator and sets other signals
        Builder.build(ens.neuron_type, ens, model=model, config=config)

    # Scale the encoders
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    # Create output signal, using built Neurons
    model.add_op(DotInc(
        Signal(scaled_encoders, name="%s.scaled_encoders" % ens.label),
        model.sig[ens]['in'],
        model.sig[ens]['neuron_in'],
        tag="%s encoding" % ens.label))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens]['neuron_out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)

Builder.register_builder(build_ensemble, nengo.objects.Ensemble)


def build_lifrate(lif, ens, model, config):
    model.add_op(SimNeurons(neurons=lif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out']))

Builder.register_builder(build_lifrate, nengo.neurons.LIFRate)


def build_lif(lif, ens, model, config):
    model.sig[ens]['voltage'] = Signal(
        np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
    model.sig[ens]['refractory_time'] = Signal(
        np.zeros(ens.n_neurons), name="%s.refractory_time" % ens.label)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[ens]['neuron_in'],
        output=model.sig[ens]['neuron_out'],
        states=[model.sig[ens]['voltage'], model.sig[ens]['refractory_time']]))

Builder.register_builder(build_lif, nengo.neurons.LIF)


def build_alifrate(alif, ens, model, config):
    model.sig[ens]['adaptation'] = Signal(
        np.zeros(ens.n_neurons), name="%s.adaptation" % ens.label)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out'],
                            states=[model.sig[ens]['adaptation']]))

Builder.register_builder(build_alifrate, nengo.neurons.AdaptiveLIFRate)


def build_alif(alif, ens, model, config):
    model.sig[ens]['voltage'] = Signal(
        np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
    model.sig[ens]['refractory_time'] = Signal(
        np.zeros(ens.n_neurons), name="%s.refractory_time" % ens.label)
    model.sig[ens]['adaptation'] = Signal(
        np.zeros(ens.n_neurons), name="%s.adaptation" % ens.label)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out'],
                            states=[model.sig[ens]['voltage'],
                                    model.sig[ens]['refractory_time'],
                                    model.sig[ens]['adaptation']]))

Builder.register_builder(build_alif, nengo.neurons.AdaptiveLIF)


def build_node(node, model, config):
    # Get input
    if node.output is None or is_callable(node.output):
        if node.size_in > 0:
            model.sig[node]['in'] = Signal(
                np.zeros(node.size_in), name="%s.signal" % node.label)
            # Reset input signal to 0 each timestep
            model.add_op(Reset(model.sig[node]['in']))

    # Provide output
    if node.output is None:
        model.sig[node]['out'] = model.sig[node]['in']
    elif not is_callable(node.output):
        model.sig[node]['out'] = Signal(node.output, name=node.label)
    else:
        sig_in, sig_out = build_pyfunc(fn=node.output,
                                       t_in=True,
                                       n_in=node.size_in,
                                       n_out=node.size_out,
                                       label="%s.pyfn" % node.label,
                                       model=model)
        if sig_in is not None:
            model.add_op(DotInc(model.sig[node]['in'],
                                Signal(1.0, name="1"),
                                sig_in,
                                tag="%s input" % node.label))
        if sig_out is not None:
            model.sig[node]['out'] = sig_out

    model.params[node] = None

Builder.register_builder(build_node, nengo.objects.Node)


def conn_probe(pre, probe, **conn_args):
    return nengo.Connection(pre, probe, **conn_args)


def synapse_probe(sig, probe, model, config):
    # We can use probe.conn_args here because we don't modify synapse
    synapse = probe.conn_args.get('synapse', None)

    if is_number(synapse):
        synapse = nengo.synapses.Lowpass(synapse)

    if synapse is None:
        model.sig[probe]['in'] = sig
    else:
        assert isinstance(synapse, nengo.synapses.Synapse)
        Builder.build(synapse, probe, sig, model=model, config=config)
        model.sig[probe]['in'] = model.sig[probe]['synapse_out']


def probe_ensemble(probe, conn_args, model, config):
    ens = probe.target
    if probe.attr == 'decoded_output':
        return conn_probe(ens, probe, **conn_args)
    elif probe.attr in ('neuron_output', 'spikes'):
        return conn_probe(ens.neurons, probe, transform=1.0, **conn_args)
    elif probe.attr == 'voltage':
        return synapse_probe(model.sig[ens]['voltage'], probe, model, config)


def probe_node(probe, conn_args, model, config):
    if probe.attr == 'output':
        return conn_probe(probe.target, probe,  **conn_args)


def probe_connection(probe, conn_args, model, config):
    if probe.attr == 'signal':
        sig_out = model.sig[probe.target]['out']
        return synapse_probe(sig_out, probe, model, config)


def build_probe(probe, model, config):
    # Make a copy so as not to modify the probe
    conn_args = probe.conn_args.copy()
    # If we make a connection, we won't add it to a network
    conn_args['add_to_container'] = False

    if isinstance(probe.target, nengo.Ensemble):
        conn = probe_ensemble(probe, conn_args, model, config)
    elif isinstance(probe.target, nengo.Node):
        conn = probe_node(probe, conn_args, model, config)
    elif isinstance(probe.target, nengo.Connection):
        conn = probe_connection(probe, conn_args, model, config)

    # Most probes are implemented as connections
    if conn is not None:
        # Make a sink signal for the connection
        model.sig[probe]['in'] = Signal(np.zeros(conn.dimensions),
                                        name=probe.label)
        model.add_op(Reset(model.sig[probe]['in']))
        # Build the connection
        Builder.build(conn, model=model, config=config)

    # Let the model know
    model.probes.append(probe)

    # We put a list here so that the simulator can fill it
    # as it simulates the model
    model.params[probe] = []

Builder.register_builder(build_probe, nengo.objects.Probe)


def build_connection(conn, model, config):  # noqa: C901
    rng = np.random.RandomState(model.next_seed())

    if isinstance(conn.pre, nengo.objects.Neurons):
        model.sig[conn]['in'] = model.sig[conn.pre.ensemble]["neuron_out"]
    else:
        model.sig[conn]['in'] = model.sig[conn.pre]["out"]

    if isinstance(conn.post, nengo.objects.Neurons):
        model.sig[conn]['out'] = model.sig[conn.post.ensemble]["neuron_in"]
    else:
        model.sig[conn]['out'] = model.sig[conn.post]["in"]

    decoders = None
    eval_points = None
    solver_info = None
    transform = np.array(conn.transform_full, dtype=np.float64)

    # Figure out the signal going across this connection
    if (isinstance(conn.pre, nengo.objects.Ensemble)
            and isinstance(conn.pre.neuron_type, nengo.neurons.Direct)):
        # Decoded connection in directmode
        if conn.function is None:
            signal = model.sig[conn]['in']
        else:
            sig_in, signal = build_pyfunc(
                fn=conn.function,
                t_in=False,
                n_in=model.sig[conn]['in'].size,
                n_out=conn.dimensions,
                label=conn.label,
                model=model)
            model.add_op(DotInc(model.sig[conn]['in'],
                                Signal(1.0, name="1"),
                                sig_in,
                                tag="%s input" % conn.label))
    elif isinstance(conn.pre, nengo.objects.Ensemble):
        # Normal decoded connection
        encoders = model.params[conn.pre].encoders
        gain = model.params[conn.pre].gain
        bias = model.params[conn.pre].bias

        eval_points = conn.eval_points
        if eval_points is None:
            eval_points = npext.array(
                model.params[conn.pre].eval_points, min_dims=2)
        elif is_integer(eval_points):
            eval_points = pick_eval_points(
                ens=conn.pre, n_points=eval_points, rng=rng)
        else:
            eval_points = npext.array(eval_points, min_dims=2)

        x = np.dot(eval_points, encoders.T / conn.pre.radius)
        activities = model.dt * conn.pre.neuron_type.rates(x, gain, bias)
        if np.count_nonzero(activities) == 0:
            raise RuntimeError(
                "In '%s', for '%s', 'activites' matrix is all zero. "
                "This is because no evaluation points fall in the firing "
                "ranges of any neurons." % (str(conn), str(conn.pre)))

        if conn.function is None:
            targets = eval_points
        else:
            targets = np.zeros((len(eval_points), conn.function_size))
            for i, ep in enumerate(eval_points):
                targets[i] = conn.function(ep)

        if conn.weight_solver is not None:
            if conn.decoder_solver is not None:
                raise ValueError("Cannot specify both 'weight_solver' "
                                 "and 'decoder_solver'.")

            # account for transform
            targets = np.dot(targets, transform.T)
            transform = np.array(1., dtype=np.float64)

            decoders, solver_info = conn.weight_solver(
                activities, targets, rng=rng,
                E=model.params[conn.post].scaled_encoders.T)
            model.sig[conn]['out'] = model.sig[conn.post]['neuron_in']
            signal_size = model.sig[conn]['out'].size
        else:
            solver = (conn.decoder_solver if conn.decoder_solver is
                      not None else nengo.decoders.lstsq_L2nz)
            decoders, solver_info = solver(activities, targets, rng=rng)
            signal_size = conn.dimensions

        # Add operator for decoders and filtering
        decoders = decoders.T

        decoder_signal = Signal(decoders, name="%s.decoders" % conn.label)
        signal = Signal(np.zeros(signal_size), name=conn.label)
        model.add_op(ProdUpdate(decoder_signal,
                                model.sig[conn]['in'],
                                Signal(0, name="decay"),
                                signal,
                                tag="%s decoding" % conn.label))
    else:
        # Direct connection
        signal = model.sig[conn]['in']

    # Add operator for filtering
    if conn.synapse is not None:
        # Note: we add a filter here even if synapse < dt,
        # in order to avoid cycles in the op graph. If the filter
        # is explicitly set to None (e.g. for a passthrough node)
        # then cycles can still occur.
        synapse = (nengo.synapses.Lowpass(conn.synapse)
                   if is_number(conn.synapse) else conn.synapse)
        assert isinstance(synapse, nengo.synapses.Synapse)
        Builder.build(synapse, conn, signal, model=model, config=config)
        signal = model.sig[conn]['synapse_out']
    else:
        synapse = None

    if conn.modulatory:
        # Make a new signal, effectively detaching from post
        model.sig[conn]['out'] = Signal(
            np.zeros(model.sig[conn]['out'].size),
            name="%s.mod_output" % conn.label)
        model.add_op(Reset(model.sig[conn]['out']))

    # Add operator for transform
    if isinstance(conn.post, nengo.objects.Neurons):
        if not model.has_built(conn.post.ensemble):
            # Since it hasn't been built, it wasn't added to the Network,
            # which is most likely because the Neurons weren't associated
            # with an Ensemble.
            raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                               "that are not a part of any Ensemble." % (
                                   conn, conn.post))
        transform *= model.params[conn.post.ensemble].gain[:, np.newaxis]

    model.add_op(DotInc(Signal(transform, name="%s.transform" % conn.label),
                        signal,
                        model.sig[conn]['out'],
                        tag=conn.label))

    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)

Builder.register_builder(build_connection, nengo.objects.Connection)


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

Builder.register_builder(build_filter_synapse, nengo.synapses.LinearFilter)


def build_lowpass_synapse(synapse, owner, input_signal, model, config):
    if synapse.tau > 0.03 * model.dt:
        d = -np.expm1(-model.dt / synapse.tau)
        num, den = [d], [d - 1]
    else:
        num, den = [1.], []

    build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config)

Builder.register_builder(build_lowpass_synapse, nengo.synapses.Lowpass)


def build_alpha_synapse(synapse, owner, input_signal, model, config):
    if synapse.tau > 0.03 * model.dt:
        a = model.dt / synapse.tau
        ea = np.exp(-a)
        num, den = [-a*ea + (1 - ea), ea*(a + ea - 1)], [-2 * ea, ea**2]
    else:
        num, den = [1.], []  # just copy the input

    build_discrete_filter_synapse(
        synapse, owner, input_signal, num, den, model, config)

Builder.register_builder(build_alpha_synapse, nengo.synapses.Alpha)
