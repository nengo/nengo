import copy
import logging

import numpy as np

from . import decoders
from . import nonlinearities
from . import objects
from . import templates

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
        return (
            self.shape,
            self.elemstrides,
            self.offset)

    def same_view_as(self, other):
        return self.structure == other.structure \
           and self.base == other.base

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
                return '<anon%d>' % id(self)
            else:
                return 'View(%s[%d])' % (self.base.name, self.offset)

    @name.setter
    def name(self, value):
        self._name = value

    def is_contiguous(self, return_range=False):
        def ret_false():
            if return_range:
                return False, None, None
            else:
                return False
        shape, strides, offset = self.structure
        if not shape:
            if return_range:
                return True, offset, offset + 1
            else:
                return True
        if len(shape) == 1:
            if strides[0] == 1:
                if return_range:
                    return True, offset, offset + shape[0]
                else:
                    return True
            else:
                return ret_false()
        if len(shape) == 2:
            if strides == (1, shape[0]) or strides == (shape[1], 1):
                if return_range:
                    return True, offset, offset + shape[0] * shape[1]
                else:
                    return True
            else:
                return ret_false()

        raise NotImplementedError()
        #if self.ndim == 1 and self.elemstrides[0] == 1:
            #return self.offset, self.offset + self.size

    def shares_memory_with(self, other):
        # XXX: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        if self.base is not other.base:
            return False
        if self is other or self.same_view_as(other):
            return True
        if self.ndim < other.ndim:
            return other.shares_memory_with(self)
        if self.size == 0 or other.size == 0:
            return False

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
            if amin <= amax <= bmin <= bmax:
                return False
            elif bmin <= bmax <= amin <= amax:
                return False
            if ae0 == be0 == 1:
                # -- strides are equal, and we've already checked for
                #    non-overlap. They do overlap, so they are aliased.
                return True
            # TODO: look for common divisor of ae0 and be0
            raise NotImplementedError('1d',
                (self.structure, other.structure))
        elif self.ndim == 2:
            # -- self is a matrix view
            #    and other is either a scalar, vector or matrix view
            a_contig, amin, amax = self.is_contiguous(return_range=True)
            if a_contig:
                # -- self has a contiguous memory layout,
                #    from amin up to but not including amax
                b_contig, bmin, bmax = other.is_contiguous(return_range=True)
                if b_contig:
                    # -- other is also contiguous
                    if amin <= amax <= bmin <= bmax:
                        return False
                    elif bmin <= bmax <= amin <= amax:
                        return False
                    else:
                        return True
                raise NotImplementedError('2d self:contig, other:discontig',
                    (self.structure, other.structure))
            raise NotImplementedError('2d',
                (self.structure, other.structure))
        else:
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
        return tuple(map(int, s / self.dtype.itemsize))

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return "Probing " + str(self.sig)

    def __repr__(self):
        return str(self)


class collect_operators_into(object):
    """
    Within this context, operators that are constructed
    are, by default, appended to an `operators` list.

    For example:

    >>> operators = []
    >>> with collect_operators_into(operators):
    >>>    Reset(foo)
    >>>    Copy(foo, bar)
    >>> assert len(operators) == 2

    After the context exits, `operators` contains the Reset
    and the Copy instances.

    """
    # -- the list of `operators` lists to which we need to append
    #    new operators when creating them.
    lists = []

    def __init__(self, operators):
        if operators is None:
            operators = []
        self.operators = operators

    def __enter__(self):
        self.lists.append(self.operators)

    def __exit__(self, exc_type, exc_value, tb):
        self.lists.remove(self.operators)

    @staticmethod
    def collect_operator(op):
        for lst in collect_operators_into.lists:
            lst.append(op)


class Operator(object):
    """
    Base class for operator instances understood by the reference simulator.
    """

    # -- N.B. automatically an @staticmethod
    def __new__(cls, *args, **kwargs):
        rval = super(Operator, cls).__new__(cls, *args, **kwargs)
        collect_operators_into.collect_operator(rval)
        return rval

    #
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

    # -- Signals that are read and not modified by this operator
    reads = []
    # -- Signals that are only assigned by this operator
    sets = []
    # -- Signals that are incremented by this operator
    incs = []
    # -- Signals that are updated to their [t + 1] value.
    #    After this operator runs, these signals cannot be
    #    used for reads until the next time step.
    updates = []

    @property
    def all_signals(self):
        # -- Sanity check that no one has accidentally modified
        #    these class variables, they should be empty
        assert not Operator.reads
        assert not Operator.sets
        assert not Operator.incs
        assert not Operator.updates

        return self.reads + self.sets + self.incs + self.updates


    def init_sigdict(self, sigdict, dt):
        """
        Install any buffers into the signals view that
        this operator will need. Classes for nonlinearities
        that use extra buffers should create them here.
        """
        for sig in self.all_signals:
            if sig.base not in sigdict:
                sigdict[sig.base] = np.zeros(
                    sig.base.shape,
                    dtype=sig.base.dtype,
                    ) + getattr(sig.base, 'value', 0)


class Reset(Operator):
    """
    Assign a constant value to a Signal.
    """
    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

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

        self.reads = [src]
        self.sets = [] if as_update else [dst]
        self.updates = [dst] if as_update else []

    def __str__(self):
        return 'Copy(%s -> %s)' % (str(self.src), str(self.dst))

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
    def __init__(self, A, X, Y, xT=False, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.xT = xT
        self.tag = tag

        self.reads = [self.A, self.X]
        self.incs = [self.Y]

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
                str(self.A), str(self.X), str(self.Y), self.tag)

    def make_step(self, dct, dt):
        X = dct[self.X]
        A = dct[self.A]
        Y = dct[self.Y]
        X = X.T if self.xT else X
        reshape = reshape_dot(A, X, Y, self.tag)
        def step():
            inc =  np.dot(A, X)
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
            val = np.dot(A,X)
            if reshape:
                val = np.asarray(val).reshape(Y.shape)
            Y[...] *= B
            Y[...] += val
        return step


class SimDirect(Operator):
    """
    Set signal `output` by some non-linear function of J (and possibly other
    things too.)
    """
    def __init__(self, output, J, nl):
        self.output = output
        self.J = J
        self.fn = nl.fn

        self.reads = [J]
        self.updates = [output]

    def make_step(self, dct, dt):
        J = dct[self.J]
        output = dct[self.output]
        fn = self.fn
        def step():
            output[...] = fn(J)
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

    def init_sigdict(self, sigdict, dt):
        Operator.init_sigdict(self, sigdict, dt)
        sigdict[self.voltage] = np.zeros(
            self.nl.n_in,
            dtype=self.voltage.dtype)
        sigdict[self.refractory_time] = np.zeros(
            self.nl.n_in,
            dtype=self.refractory_time.dtype)

    def make_step(self, dct, dt):
        J = dct[self.J]
        output = dct[self.output]
        v = dct[self.voltage]
        rt = dct[self.refractory_time]
        fn = self.nl.step_math0
        def step():
            fn(dt, J, v, rt, output)
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

    def make_step(self, dct, dt):
        J = dct[self.J]
        output = dct[self.output]
        rates_fn = self.nl.math
        def step():
            output[...] = rates_fn(dt, J)
        return step


def builds(cls):
    """A decorator that adds a _builds attribute to a function,
    denoting that that function is used to build
    a certain high-level Nengo object.

    This is used by the Builder class to associate its own methods
    with the objects that those methods build.

    """
    def wrapper(func):
        func._builds = cls
        return func
    return wrapper


class Builder(object):
    """A callable class that copies a model and determines the signals
    and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

      1. Ensembles and Nodes
      2. Probes
      3. Connections

    """

    def __init__(self, copy=True):
        # Whether or not we make a deep-copy of the model we're building
        self.copy = copy

        # Build up a diction mapping from high-level object -> builder method,
        # so that we don't have to use a lame if/elif chain to call the
        # right method.
        self._builders = {}
        for methodname in dir(self):
            method = getattr(self, methodname)
            if hasattr(method, '_builds'):
                self._builders.update({method._builds: method})

    def __call__(self, model, dt):
        if self.copy:
            # Make a copy of the model so that we can reuse the non-built model
            logger.info("Copying model")
            memo = {}
            self.model = copy.deepcopy(model, memo)
            self.model.memo = memo
        else:
            self.model = model

        self.model.name = self.model.name + ", dt=%f" % dt
        self.model.dt = dt
        if self.model.seed is None:
            self.model.seed = np.random.randint(np.iinfo(np.int32).max)

        # The purpose of the build process is to fill up these lists
        self.model.probes = []
        self.model.operators = []

        # 1. Build objects
        logger.info("Building objects")
        for obj in self.model.objs.values():
            self._builders[obj.__class__](obj)

        # 2. Then probes
        logger.info("Building probes")
        for target in self.model.probed:
            if not isinstance(self.model.probed[target], Probe):
                self._builders[objects.Probe](self.model.probed[target])
                self.model.probed[target] = self.model.probed[target].probe

        # 3. Then connections
        logger.info("Building connections")
        for o in self.model.objs.values():
            for c in o.connections_out:
                self._builders[c.__class__](c)
        for c in self.model.connections:
            self._builders[c.__class__](c)

        # Set up t and timesteps
        self.model.operators.append(ProdUpdate(
            Signal(1), Signal(self.model.dt), Signal(1), self.model.t.signal))
        self.model.operators.append(ProdUpdate(
            Signal(1), Signal(1), Signal(1), self.model.steps.signal))

        return self.model

    @builds(objects.Ensemble)
    def build_ensemble(self, ens, signal=None):
        if ens.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be positive' % ens.n_neurons)

        if ens.dimensions <= 0:
            raise ValueError(
                'Number of dimensions (%d) must be positive' % ens.dimensions)

        # Create random number generator
        if ens.seed is None:
            ens.seed = self.model._get_new_seed()
        rng = np.random.RandomState(ens.seed)

        # Generate eval points
        if ens.eval_points is None:
            ens.eval_points = decoders.sample_hypersphere(
                ens.dimensions, ens.EVAL_POINTS, rng) * ens.radius
        else:
            ens.eval_points = np.array(ens.eval_points, dtype=np.float64)
            if ens.eval_points.ndim == 1:
                ens.eval_points.shape = (-1, 1)

        # Set up signal
        if signal is None:
            ens.signal = Signal(np.zeros(ens.dimensions),
                                name=ens.name + ".signal")
        else:
            # Assume that a provided signal is already in the model
            ens.signal = signal
            ens.dimensions = ens.signal.size

        #reset input signal to 0 each timestep (unless this ensemble has
        #a view of a larger signal -- generally meaning it is an ensemble
        #in an ensemble array -- in which case something else will be
        #responsible for resetting)
        if ens.signal.base == ens.signal:
            self.model.operators.append(Reset(ens.signal))

        # Set up neurons
        if ens.neurons.gain is None or ens.neurons.bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.neurons.n_neurons, rng=rng)
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=rng)
            ens.neurons.set_gain_bias(ens.max_rates, ens.intercepts)

        self._builders[ens.neurons.__class__](ens.neurons)

        # Set up encoders
        if ens.encoders is None:
            ens.encoders = decoders.sample_hypersphere(
                ens.dimensions, ens.neurons.n_neurons, rng, surface=True)
        else:
            ens.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if ens.encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (ens.encoders.shape, enc_shape))

            norm = np.sum(ens.encoders * ens.encoders, axis=1)[:, np.newaxis]
            ens.encoders /= np.sqrt(norm)

        ens._scaled_encoders = ens.encoders * (
            ens.neurons.gain / ens.radius)[:, np.newaxis]
        self.model.operators.append(DotInc(Signal(ens._scaled_encoders),
                                           ens.signal,
                                           ens.neurons.input_signal))

        # Set up probes, but don't build them (done explicitly later)
        # Note: Have to set it up here because we only know these things
        #       (dimensions, n_neurons) at build time.
        for probe in ens.probes['decoded_output']:
            probe.dimensions = ens.dimensions
        for probe in ens.probes['spikes']:
            probe.dimensions = ens.n_neurons
        for probe in ens.probes['voltages']:
            probe.dimensions = ens.n_neurons

    @builds(objects.PassthroughNode)
    def build_passthrough(self, ptn):
        ptn.signal = Signal(np.zeros(ptn.dimensions), name=ptn.name + ".signal")

        #reset input signal to 0 each timestep
        self.model.operators.append(Reset(ptn.signal))

        # Set up probes
        for probe in ptn.probes['output']:
            probe.dimensions = ptn.dimensions
            self.model.add(probe)

    @builds(objects.Node)
    def build_node(self, node):
        if not callable(node.output):
            if isinstance(node.output, (int, float, long, complex)):
                node.signal = Signal([node.output], name=node.name)
            else:
                node.signal = Signal(node.output, name=node.name)
            node.output_signal = node.signal
        else:
            node.signal = Signal(np.zeros(node.dimensions),
                                 name=node.name + ".signal")

            #reset input signal to 0 each timestep
            self.model.operators.append(Reset(node.signal))

            # Set up non-linearity
            n_out = np.array(node.output(np.ones(node.dimensions))).size
            node.nonlinear = nonlinearities.Direct(n_in=node.dimensions,
                                                   n_out=n_out,
                                                   fn=node.output,
                                                   name=node.name + ".Direct")
            self.build_direct(node.nonlinear)
            node.output_signal = node.nonlinear.output_signal

            self.model.operators.append(DotInc(node.signal,
                                               Signal([[1.0]]),
                                               node.nonlinear.input_signal))

        # Set up probes
        for probe in node.probes['output']:
            probe.dimensions = node.output_signal.shape

    @builds(objects.Probe)
    def build_probe(self, probe):
        # Set up signal
        probe.signal = Signal(np.zeros(probe.dimensions), name=probe.name)

        #reset input signal to 0 each timestep
        self.model.operators.append(Reset(probe.signal))

        # Set up probe
        probe.probe = Probe(probe.signal, probe.sample_every)
        self.model.probes.append(probe.probe)

    @staticmethod
    def filter_coefs(pstc, dt):
        pstc = max(pstc, dt)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)

    def _build_connection_filter(self, conn):
        if conn.filter is not None and conn.filter > self.model.dt:
            # Set up signal
            name = conn.pre.name + ".filtered(%f)" % conn.filter
            conn.signal = Signal(np.zeros(conn.pre.size), name=name)

            # Set up filters and transforms
            o_coef, n_coef = self.filter_coefs(pstc=conn.filter, dt=self.model.dt)
            self.model.operators.append(ProdUpdate(Signal(n_coef),
                                                   conn.pre,
                                                   Signal(o_coef),
                                                   conn.signal))
        else:
            # Signal should already be in the model
            conn.signal = conn.pre

    def _build_connection_transform(self, conn):
        self.model.operators.append(DotInc(Signal(conn.transform),
                                           conn.signal,
                                           conn.post))

    def _build_connection_probes(self, conn):
        for probe in conn.probes['signal']:
            probe.dimensions = conn.signal.size
            self.model.add(probe)

    @builds(objects.SignalConnection)
    def build_signalconnection(self, conn):
        # Pre / post may be high level objects (ensemble, node) or signals
        if not isinstance(conn.pre, SignalView):
            conn.pre = conn.pre.signal

        if not isinstance(conn.post, SignalView):
            conn.post = conn.post.signal

        # Set up filters and transform
        self._build_connection_filter(conn)
        self._build_connection_transform(conn)

        # Set up probes
        self._build_connection_probes(conn)

    @builds(objects.NonlinearityConnection)
    def build_nonlinearityconnection(self, conn):
        if not isinstance(conn.pre, SignalView):
            if hasattr(conn.pre, 'nonlinear'):
                conn.pre = conn.pre.nonlinear.output_signal
            else:
                conn.pre = conn.pre.output_signal

        # Post could be a node / ensemble, etc
        if isinstance(conn.post, nonlinearities.Nonlinearity):
            if isinstance(conn.post, nonlinearities.NeuralNonlinearity):
                conn.transform = conn.transform * conn.post.gain[:,None]
            conn.post = conn.post.input_signal
        elif not isinstance(conn.post, SignalView):
            conn.post = conn.post.signal

        # Set up filters and transform
        self._build_connection_filter(conn)
        self._build_connection_transform(conn)

        # Set up probes
        self._build_connection_probes(conn)

    @builds(objects.DecodedConnection)
    def build_decodedconnection(self, conn):
        # Pre must be an ensemble -- but, don't want to import objects
        assert isinstance(conn.pre, objects.Ensemble)

        # Post could be a node / ensemble, etc
        if isinstance(conn.post, nonlinearities.Nonlinearity):
            if isinstance(conn.post, nonlinearities.NeuralNonlinearity):
                conn.transform = conn.transform * conn.post.gain[:,None]
            conn.post = conn.post.input_signal
        elif not isinstance(conn.post, SignalView):
            conn.post = conn.post.signal
        dt = self.model.dt

        # Set up signal
        dims = conn.dimensions
        conn.signal = Signal(np.zeros(dims), name=conn.name)

        # Set up decoders
        if conn._decoders is None:
            activities = conn.pre.activities(conn.eval_points) * dt
            if conn.function is None:
                targets = conn.eval_points
            else:
                targets = np.array(
                    [conn.function(ep) for ep in conn.eval_points])
                if len(targets.shape) < 2:
                    targets.shape = targets.shape[0], 1
            conn._decoders = conn.decoder_solver(activities, targets)

        # Set up filters and transform
        conn.pre = conn.pre.neurons.output_signal

        # DecodedConnection._add_filter
        if conn.filter is not None and conn.filter > dt:
            o_coef, n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)

            self.model.operators.append(
                ProdUpdate(Signal(conn._decoders*n_coef),
                           conn.pre,
                           Signal(o_coef),
                           conn.signal))
        else:
            self.model.operators.append(
                ProdUpdate(Signal(conn._decoders),
                           conn.pre,
                           Signal(0),
                           conn.signal))

        self._build_connection_transform(conn)

        # Set up probes
        self._build_connection_probes(conn)

    @builds(objects.ConnectionList)
    def build_connectionlist(self, conn):
        conn.transform = np.asarray(conn.transform)

        i = 0
        for connection in conn.connections:
            pre_dim = connection.dimensions

            if conn.transform.ndim == 0:
                trans = np.zeros((connection.post.dimensions, pre_dim))
                np.fill_diagonal(trans[i:i+pre_dim,:], conn.transform)
            elif conn.transform.ndim == 2:
                trans = conn.transform[:,i:i+pre_dim]
            else:
                raise NotImplementedError(
                    "Only transforms with 0 or 2 ndims are accepted")

            i += pre_dim

            connection.transform = trans
            self._builders[connection.__class__](connection)

    @builds(templates.EnsembleArray)
    def build_ensemblearray(self, ea):
        ea.signal = Signal(np.zeros(ea.dimensions), name=ea.name+".signal")
        self.model.operators.append(Reset(ea.signal))
        dims = ea.dimensions_per_ensemble

        for i, ens in enumerate(ea.ensembles):
            self.build_ensemble(ens, signal=ea.signal[i*dims:(i+1)*dims])

        for probe in ea.probes['decoded_output']:
            probe.dimensions = ea.dimensions

    @builds(nonlinearities.Direct)
    def build_direct(self, nl):
        nl.input_signal = Signal(np.zeros(nl.n_in), name=nl.name + '.input')
        nl.output_signal = Signal(np.zeros(nl.n_out), name=nl.name + '.output')
        self.model.operators.append(Reset(nl.input_signal))
        self.model.operators.append(SimDirect(output=nl.output_signal,
                                              J=nl.input_signal,
                                              nl=nl))

    def build_neural_nonlinearity(self, nl):
        nl.input_signal = Signal(np.zeros(nl.n_neurons),
                                 name=nl.name + '.input')
        nl.output_signal = Signal(np.zeros(nl.n_neurons),
                                  name=nl.name + '.output')
        nl.bias_signal = Signal(nl.bias, name=nl.name + '.bias')
        self.model.operators.append(
            Copy(src=nl.bias_signal, dst=nl.input_signal))

    @builds(nonlinearities.LIFRate)
    def build_lifrate(self, lif):
        self.build_neural_nonlinearity(lif)
        self.model.operators.append(SimLIFRate(output=lif.output_signal,
                                               J=lif.input_signal,
                                               nl=lif))

    @builds(nonlinearities.LIF)
    def build_lif(self, lif):
        self.build_neural_nonlinearity(lif)
        lif.voltage = Signal(np.zeros(lif.n_neurons))
        lif.refractory_time = Signal(np.zeros(lif.n_neurons))
        self.model.operators.append(SimLIF(output=lif.output_signal,
                                           J=lif.input_signal,
                                           nl=lif,
                                           voltage=lif.voltage,
                                           refractory_time=lif.refractory_time))
