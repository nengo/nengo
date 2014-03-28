"""Reference implementation for building a nengo.Network."""

import collections
import logging

import numpy as np

import nengo.decoders
import nengo.neurons
import nengo.objects
import nengo.utils.distributions as dists
import nengo.utils.numpy as npext
from nengo.utils.compat import is_callable, is_integer

logger = logging.getLogger(__name__)

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

    def __getitem__(self, item):  # noqa
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

    def shares_memory_with(self, other):  # noqa
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
    """Interpretable, vector-valued quantity within NEF"""

    def __init__(self, value, name=None):
        self.value = np.asarray(value, dtype=np.float64)
        if name is not None:
            self._name = name
        if assert_named_signals:
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


class Operator(object):
    """
    Base class for operator instances understood by the reference simulator.
    """

    # The lifetime of a Signal during one simulator timestep:
    # 0) at most one set operator (optional)
    # 1) any number of increments
    # 2) any number of reads
    # 3) at most one update
    #
    # A signal that is only read can be considered a "constant"
    #
    # A signal that is both set *and* updated can be a problem: since
    # reads must come after the set, and the set will destroy
    # whatever were the contents of the update, it can be the case
    # that the update is completely hidden and rendered irrelevant.
    # There are however at least two reasons to use both a set and an update:
    # (a) to use a signal as scratch space (updating means destroying it)
    # (b) to use sets and updates on partly overlapping views of the same
    #     memory.
    #
    # N.B. It is done on purpose that there are no default values for
    # reads, sets, incs, and updates.
    #
    # Each operator should explicitly set each of these properties.

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
        """
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
    """
    Assign a constant value to a Signal.
    """

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
    """
    Assign the value of one signal to another
    """

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

    def make_step(self, dct, dt):
        dst = dct[self.dst]
        src = dct[self.src]

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
    """
    Increment signal Y by dot(A, X)
    """

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

    def make_step(self, dct, dt):
        X = dct[self.X]
        A = dct[self.A]
        Y = dct[self.Y]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            inc = np.dot(A, X)
            if reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step


class ProdUpdate(Operator):
    """
    Sets Y <- dot(A, X) + B * Y
    """

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

    def make_step(self, dct, dt):
        X = dct[self.X]
        A = dct[self.A]
        Y = dct[self.Y]
        B = dct[self.B]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            val = np.dot(A, X)
            if reshape:
                val = np.asarray(val).reshape(Y.shape)
            Y[...] *= B
            Y[...] += val
        return step


class SimPyFunc(Operator):
    """Set signal `output` by some non-linear function of J
    (and possibly other things too.)
    """

    def __init__(self, output, fn, t_in, x):
        self.output = output
        self.fn = fn
        self.t_in = t_in
        self.x = x

        self.reads = [] if x is None else [x]
        self.updates = [output]
        self.sets = []
        self.incs = []

    def __str__(self):
        return "SimPyFunc(%s -> %s '%s')" % (self.x, self.output, self.fn)

    def make_step(self, dct, dt):
        output = dct[self.output]
        fn = self.fn
        args = [dct['__time__']] if self.t_in else []
        args += [dct[self.x]] if self.x is not None else []

        def step():
            y = fn(*args)
            if y is None:
                raise ValueError(
                    "Function '%s' returned invalid value" % fn.__name__)
            output[...] = y

        return step


class SimLIF(Operator):
    """
    Set output to spikes generated by an LIF model.
    """

    def __init__(self, output, J, nl, voltage, refractory_time):
        self.nl = nl
        self.output = output
        self.J = J
        self.voltage = voltage
        self.refractory_time = refractory_time

        self.reads = [J]
        self.updates = [self.voltage, self.refractory_time, output]
        self.sets = []
        self.incs = []

    def init_signals(self, signals, dt):
        Operator.init_signals(self, signals, dt)
        signals.init(self.voltage,
                     np.zeros(self.nl.n_in, dtype=self.voltage.dtype))
        signals.init(self.refractory_time,
                     np.zeros(self.nl.n_in, dtype=self.refractory_time.dtype))

    def make_step(self, dct, dt):
        J = dct[self.J]
        output = dct[self.output]
        v = dct[self.voltage]
        rt = dct[self.refractory_time]

        def step():
            self.nl.step_math(dt, J, v, rt, output)
        return step


class SimLIFRate(Operator):
    """
    Set output to spike rates of an LIF model.
    """

    def __init__(self, output, J, nl):
        self.output = output
        self.J = J
        self.nl = nl

        self.reads = [J]
        self.updates = [output]
        self.sets = []
        self.incs = []

    def make_step(self, dct, dt):
        J = dct[self.J]
        output = dct[self.output]

        def step():
            self.nl.step_math(dt, J, output)
        return step


def random_maxint(rng):
    """Returns rng.randint(x) where x is the maximum 32-bit integer."""
    return rng.randint(np.iinfo(np.int32).max)


_builder_func_dict = {}  # Nengo object -> builder method; set by @builds


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt, label="Model", seed=None):
        self.operators = []
        self.probes = []
        self.sig_in = {}
        self.sig_out = {}

        self.dt = dt
        self.label = label
        self.seed = seed
        self.params = None

    def __str__(self):
        return "Model: %s" % self.label


BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform'])
BuiltNeurons = collections.namedtuple('BuiltNeurons', ['gain', 'bias'])
BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble',
    ['eval_points', 'encoders', 'intercepts', 'max_rates', 'scaled_encoders'])


def register(callback):
    """A decorator that registers the wrapped function with callback(func)."""
    def wrapper(func):
        callback(func)
        return func
    return wrapper


class Builder(object):
    """Takes a Network object and returns a Model.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

      1. Ensembles, Nodes, Neurons, Probes
      2. Subnetworks (recursively)
      3. Connections
    """

    # A decorator that registers the given Nengo object class with the function
    builds = lambda cls: register(
        lambda f: _builder_func_dict.__setitem__(cls, f))

    def __call__(self, network, dt):
        # Artifacts of the build process.
        self.built = {}

        # Resources used by the build process.
        seed = (random_maxint(np.random) if network.seed is None
                else network.seed)
        self.rng = np.random.RandomState(seed)

        # Build the entire Network into Model output attribute.
        self.model = Model(dt, "%s, dt=%f" % (network.label, dt), seed)
        self.build(network)
        self.model.params = self.built
        return self.model

    def has_built(self, obj):
        """Returns true iff obj has been processed by build."""
        return obj in self.built

    def next_seed(self):
        """Yields a seed to use for RNG during build computations."""
        return random_maxint(self.rng)

    def build(self, obj, *args, **kwargs):
        """Builds the given object with the associated builder method."""
        if isinstance(obj, nengo.objects.Network):
            # Use build_network for arbitrary subclasses of Network
            cls = nengo.objects.Network
        else:
            cls = obj.__class__

        if cls not in _builder_func_dict:
            raise ValueError("Cannot build object of type '%s'." %
                             cls.__name__)

        if not self.has_built(obj):
            self.built[obj] = _builder_func_dict[cls](
                self, obj, *args, **kwargs)
        else:
            # This means the Network object contained two objects with the same
            # id, which gives undefined behaviour. This is most likely the
            # result of Neurons being used in two different Ensembles, in which
            # case the same neuron would need two different tuning curves.
            # TODO: Prevent this at pre-build validation time.
            logger.warning("Object (%s) with id=%d has been referenced twice "
                           "within the Network.",
                           obj.__class__.__name__, id(obj))
        return self.built[obj]

    @builds(nengo.objects.Network)
    def build_network(self, network):
        # 1. Build ensembles and nodes
        logger.info("Building ensembles and nodes")
        for obj in network.ensembles + network.nodes:
            self.build(obj)

        # 2. Then subnetworks
        logger.info("Building subnetworks")
        for subnetwork in network.networks:
            self.build(subnetwork)

        # 3. Then connections
        logger.info("Building connections")
        for conn in network.connections:
            self.build(conn)

    @builds(nengo.objects.Ensemble)  # noqa
    def build_ensemble(self, ens):
        # Create random number generator
        seed = self.next_seed() if ens.seed is None else ens.seed
        rng = np.random.RandomState(seed)

        # Generate eval points
        if ens.eval_points is None or is_integer(ens.eval_points):
            eval_points = self.generate_eval_points(
                ens=ens, n_points=ens.eval_points, rng=rng)
        else:
            eval_points = npext.array(
                ens.eval_points, dtype=np.float64, min_dims=2)

        # Set up signal
        self.model.sig_in[ens] = Signal(np.zeros(ens.dimensions),
                                        name="%s.signal" % ens.label)
        self.model.operators.append(Reset(self.model.sig_in[ens]))

        # Set up encoders
        if ens.encoders is None:
            if isinstance(ens.neurons, nengo.Direct):
                encoders = np.identity(ens.dimensions)
            else:
                sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
                encoders = sphere.sample(ens.neurons.n_neurons, rng=rng)
        else:
            encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions); "
                    "in this case %s." % (encoders.shape, enc_shape))
            encoders /= npext.norm(encoders, axis=1, keepdims=True)

        # Determine max_rates and intercepts
        if isinstance(ens.max_rates, dists.Distribution):
            max_rates = ens.max_rates.sample(
                ens.neurons.n_neurons, rng=rng)
        else:
            max_rates = np.array(ens.max_rates)
        if isinstance(ens.intercepts, dists.Distribution):
            intercepts = ens.intercepts.sample(
                ens.neurons.n_neurons, rng=rng)
        else:
            intercepts = np.array(ens.intercepts)

        # Build the neurons
        if isinstance(ens.neurons, nengo.Direct):
            bn = self.build(ens.neurons, ens.dimensions)
        else:
            bn = self.build(ens.neurons, max_rates, intercepts)

        # Scale the encoders
        if isinstance(ens.neurons, nengo.Direct):
            scaled_encoders = encoders
        else:
            scaled_encoders = encoders * (bn.gain / ens.radius)[:, np.newaxis]

        # Create output signal, using built Neurons
        self.model.operators.append(DotInc(
            Signal(scaled_encoders, name="%s.scaled_encoders" % ens.label),
            self.model.sig_in[ens],
            self.model.sig_in[ens.neurons],
            tag="%s encoding" % ens.label))

        # Output is neural output
        self.model.sig_out[ens] = self.model.sig_out[ens.neurons]

        for probe in ens.probes["decoded_output"]:
            self.build(probe, dimensions=ens.dimensions)
        for probe in ens.probes["spikes"] + ens.probes["voltages"]:
            self.build(probe, dimensions=ens.neurons.n_neurons)

        return BuiltEnsemble(eval_points=eval_points,
                             encoders=encoders,
                             intercepts=intercepts,
                             max_rates=max_rates,
                             scaled_encoders=scaled_encoders)

    @builds(nengo.objects.Node)
    def build_node(self, node):
        # Get input
        if node.output is None or is_callable(node.output):
            if node.size_in > 0:
                self.model.sig_in[node] = Signal(
                    np.zeros(node.size_in), name="%s.signal" % node.label)
                # Reset input signal to 0 each timestep
                self.model.operators.append(Reset(self.model.sig_in[node]))

        # Provide output
        if node.output is None:
            self.model.sig_out[node] = self.model.sig_in[node]
        elif not is_callable(node.output):
            self.model.sig_out[node] = Signal(node.output, name=node.label)
        else:
            sig_in, sig_out = self.build_pyfunc(fn=node.output,
                                                t_in=True,
                                                n_in=node.size_in,
                                                n_out=node.size_out,
                                                label="%s.pyfn" % node.label)
            if node.size_in > 0:
                self.model.operators.append(DotInc(
                    self.model.sig_in[node],
                    Signal(1.0, name="1"),
                    sig_in,
                    tag="%s input" % node.label))
            self.model.sig_out[node] = sig_out

        for probe in node.probes["output"]:
            self.build(probe, dimensions=self.model.sig_out[node].shape)

    @builds(nengo.objects.Probe)
    def build_probe(self, probe, dimensions):
        self.model.sig_in[probe] = Signal(np.zeros(dimensions),
                                          name=probe.label)
        # Reset input signal to 0 each timestep
        self.model.operators.append(Reset(self.model.sig_in[probe]))
        self.model.probes.append(probe)

        # We return a list here because the simulator initializes
        # its probedict with the values returned from the built functions.
        return []

    @staticmethod
    def filter_coefs(pstc, dt):
        pstc = max(pstc, dt)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)

    def filtered_signal(self, signal, pstc):
        name = "%s.filtered(%f)" % (signal.name, pstc)
        filtered = Signal(np.zeros(signal.size), name=name)
        o_coef, n_coef = self.filter_coefs(pstc=pstc, dt=self.model.dt)
        self.model.operators.append(ProdUpdate(
            Signal(n_coef, name="n_coef"),
            signal,
            Signal(o_coef, name="o_coef"),
            filtered,
            tag="%s filtering" % name))
        return filtered

    @builds(nengo.Connection)  # noqa
    def build_connection(self, conn):
        dt = self.model.dt
        rng = np.random.RandomState(self.next_seed())

        self.model.sig_in[conn] = self.model.sig_out[conn.pre]
        self.model.sig_out[conn] = self.model.sig_in[conn.post]

        decoders = None
        eval_points = None
        transform = np.array(conn.transform_full, dtype=np.float64)

        # Figure out the signal going across this connection
        if (isinstance(conn.pre, nengo.Ensemble)
                and isinstance(conn.pre.neurons, nengo.Direct)):
            # Decoded connection in directmode
            if conn.function is None:
                signal = self.model.sig_in[conn]
            else:
                sig_in, signal = self.build_pyfunc(
                    fn=conn.function,
                    t_in=False,
                    n_in=self.model.sig_in[conn].size,
                    n_out=conn.dimensions,
                    label=conn.label)
                self.model.operators.append(DotInc(
                    self.model.sig_in[conn],
                    Signal(1.0, name="1"),
                    sig_in,
                    tag="%s input" % conn.label))
        elif isinstance(conn.pre, nengo.Ensemble):
            # Normal decoded connection
            encoders = self.built[conn.pre].encoders
            gain = self.built[conn.pre.neurons].gain
            bias = self.built[conn.pre.neurons].bias

            eval_points = conn.eval_points
            if eval_points is None:
                eval_points = npext.array(
                    self.built[conn.pre].eval_points, min_dims=2)
            elif is_integer(eval_points):
                eval_points = self.generate_eval_points(
                    ens=conn.pre, n_points=eval_points, rng=rng)
            else:
                eval_points = npext.array(eval_points, min_dims=2)

            x = np.dot(eval_points, encoders.T / conn.pre.radius)
            activities = dt * conn.pre.neurons.rates(x, gain, bias)
            if conn.function is None:
                targets = eval_points
            else:
                targets = npext.array(
                    [conn.function(ep) for ep in eval_points], min_dims=2)

            if conn.weight_solver is not None:
                if conn.decoder_solver is not None:
                    raise ValueError("Cannot specify both 'weight_solver' "
                                     "and 'decoder_solver'.")

                # account for transform
                targets = np.dot(targets, transform.T)
                transform = np.array(1., dtype=np.float64)

                decoders = conn.weight_solver(
                    activities, targets, rng=rng,
                    E=self.built[conn.post].scaled_encoders.T)
                self.model.sig_out[conn] = self.model.sig_in[
                    conn.post.neurons]
                signal_size = self.model.sig_out[conn].size
            else:
                solver = (conn.decoder_solver if conn.decoder_solver is
                          not None else nengo.decoders.lstsq_L2nz)
                decoders = solver(activities, targets, rng=rng)
                signal_size = conn.dimensions

            # Add operator for decoders and filtering
            decoders = decoders.T
            if conn.filter is not None and conn.filter > dt:
                o_coef, n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
                decoder_signal = Signal(
                    decoders * n_coef,
                    name="%s.decoders * n_coef" % conn.label)
            else:
                decoder_signal = Signal(decoders,
                                        name="%s.decoders" % conn.label)
                o_coef = 0

            signal = Signal(np.zeros(signal_size), name=conn.label)
            self.model.operators.append(ProdUpdate(
                decoder_signal,
                self.model.sig_in[conn],
                Signal(o_coef, name="o_coef"),
                signal,
                tag="%s decoding" % conn.label))
        else:
            # Direct connection
            signal = self.model.sig_in[conn]

        # Add operator for filtering
        if decoders is None and conn.filter is not None and conn.filter > dt:
            signal = self.filtered_signal(signal, conn.filter)

        if conn.modulatory:
            # Make a new signal, effectively detaching from post
            self.model.sig_out[conn] = Signal(
                np.zeros(self.model.sig_out[conn].size),
                name="%s.mod_output" % conn.label)
            # Add reset operator?
            # TODO: add unit test

        # Add operator for transform
        if isinstance(conn.post, nengo.objects.Neurons):
            if not self.has_built(conn.post):
                # Since it hasn't been built, it wasn't added to the Network,
                # which is most likely because the Neurons weren't associated
                # with an Ensemble.
                raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                                   "that are not a part of any Ensemble." % (
                                       conn, conn.post))
            transform *= self.built[conn.post].gain[:, np.newaxis]

        self.model.operators.append(
            DotInc(Signal(transform, name="%s.transform" % conn.label),
                   signal,
                   self.model.sig_out[conn],
                   tag=conn.label))

        # Set up probes
        for probe in conn.probes["signal"]:
            self.build(probe, dimensions=self.model.sig_out[conn].size)

        return BuiltConnection(decoders=decoders,
                               eval_points=eval_points,
                               transform=transform)

    def build_pyfunc(self, fn, t_in, n_in, n_out, label):
        if n_in:
            sig_in = Signal(np.zeros(n_in), name="%s.input" % label)
            self.model.operators.append(Reset(sig_in))
        else:
            sig_in = None
        sig_out = Signal(np.zeros(n_out), name="%s.output" % label)
        self.model.operators.append(
            SimPyFunc(output=sig_out, fn=fn, t_in=t_in, x=sig_in))
        return sig_in, sig_out

    @builds(nengo.neurons.Direct)
    def build_direct(self, direct, dimensions):
        self.model.sig_in[direct] = Signal(np.zeros(dimensions),
                                           name=direct.label)
        self.model.sig_out[direct] = self.model.sig_in[direct]
        self.model.operators.append(Reset(self.model.sig_in[direct]))
        return BuiltNeurons(gain=None, bias=None)

    def build_neurons(self, neurons, max_rates, intercepts):
        if neurons.n_neurons <= 0:
            raise ValueError(
                "Number of neurons (%d) must be positive." % neurons.n_neurons)
        gain, bias = neurons.gain_bias(max_rates, intercepts)
        self.model.sig_in[neurons] = Signal(
            np.zeros(neurons.n_neurons), name="%s.input" % neurons.label)
        self.model.sig_out[neurons] = Signal(
            np.zeros(neurons.n_neurons), name="%s.output" % neurons.label)

        self.model.operators.append(Copy(
            src=Signal(bias, name="%s.bias" % neurons.label),
            dst=self.model.sig_in[neurons]))

        for probe in neurons.probes["output"]:
            self.build(probe, dimensions=neurons.n_neurons)

        return BuiltNeurons(gain=gain, bias=bias)

    @builds(nengo.neurons.LIFRate)
    def build_lifrate(self, lif, max_rates, intercepts):
        rval = self.build_neurons(lif, max_rates, intercepts)
        self.model.operators.append(SimLIFRate(
            output=self.model.sig_out[lif],
            J=self.model.sig_in[lif],
            nl=lif))
        return rval

    @builds(nengo.neurons.LIF)
    def build_lif(self, lif, max_rates, intercepts):
        rval = self.build_neurons(lif, max_rates, intercepts)
        voltage = Signal(np.zeros(lif.n_neurons),
                         name="%s.voltage" % lif.label)
        refractory_time = Signal(np.zeros(lif.n_neurons),
                                 name="%s.refractory_time" % lif.label)
        self.model.operators.append(SimLIF(output=self.model.sig_out[lif],
                                           J=self.model.sig_in[lif],
                                           nl=lif,
                                           voltage=voltage,
                                           refractory_time=refractory_time))
        return rval

    def generate_eval_points(self, ens, n_points, rng):
        if n_points is None:
            # use a heuristic to pick the number of points
            dims, neurons = ens.dimensions, ens.neurons.n_neurons
            n_points = max(np.clip(500 * dims, 750, 2500), 2 * neurons)

        return dists.UniformHypersphere(ens.dimensions).sample(
            n_points, rng=rng) * ens.radius
