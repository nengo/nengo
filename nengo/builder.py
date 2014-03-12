import collections
import logging

import numpy as np

import nengo
import nengo.decoders
import nengo.objects
import nengo.utils.distributions as dists
import nengo.utils.numpy as npext

logger = logging.getLogger(__name__)

"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


def _array2d(x, **kwargs):
    """Ensure an array is two-dimensional"""
    x = np.array(x, **kwargs)
    if x.ndim < 2:
        x.shape = x.size, 1
    return x


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
        # XXX: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
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

    def init_sigdict(self, sigdict, dt):
        """
        Install any buffers into the signals view that
        this operator will need. Classes for neurons
        that use extra buffers should create them here.
        """
        for sig in self.all_signals:
            if sig.base not in sigdict:
                sigdict[sig.base] = np.asarray(
                    np.zeros(
                        sig.base.shape,
                        dtype=sig.base.dtype,
                    ) + getattr(sig.base, 'value', 0))


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
        return 'SimPyFunc(%s -> %s "%s")' % (
            str(self.x), str(self.output), str(self.fn))

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

    def __init__(self):
        # Build up a dictionary mapping from high-level object
        # to builder method, so that we don't have to use a lame
        # if/elif chain to call the right method.
        self.builders = {}
        for methodname in dir(self):
            method = getattr(self, methodname)
            if hasattr(method, '_builds'):
                self.builders[method._builds] = method
        self._data = {}
        self.probes = []
        self.operators = []
        self.sig_in = {}
        self.sig_out = {}

    def __call__(self, model, dt):
        self.model = model  # This model does not get changed!!

        self.dt = dt
        self._rng = None
        self.seed = (np.random.randint(np.iinfo(np.int32).max)
                     if self.model.seed is None else self.model.seed)
        # XXX temporary, until better probes
        self._data = {'gain': {},
                      'bias': {},
                      'encoders': {},
                      'scaled_encoders': {},
                      'eval_points': {},
                      'decoders': {},
                      'transform': {}}
        self.probes = []
        self.operators = []
        self.sig_in = {}
        self.sig_out = {}

        # 1. Build objects
        logger.info("Building objects")
        for obj in self.model.objs:
            self.builders[obj.__class__](obj)

        # 2. Then probes
        logger.info("Building probes")
        for target in model.probed:
            probe = self.model.probed[target]
            self.builders[nengo.Probe](probe)
            self.sig_in[probe] = probe.sig

        # 3. Then connections
        logger.info("Building connections")
        for c in self.model.connections:
            self.builders[c.__class__](c)

        return {"dt": self.dt,
                "_data": self._data,
                "label": self.model.label + ", dt=%f" % self.dt,
                "seed": self.seed,
                "probes": self.probes,
                "operators": self.operators}

    def _get_new_seed(self):
        if self._rng is None:
            # never create rng without knowing the seed
            assert self.seed is not None
            self._rng = np.random.RandomState(self.seed)
        return self._rng.randint(np.iinfo(np.int32).max)

    @builds(nengo.Ensemble)  # noqa
    def build_ensemble(self, ens):
        # Create random number generator
        seed = ens.seed
        if seed is None:
            seed = self._get_new_seed()
        rng = np.random.RandomState(seed)

        # Generate eval points
        if ens.eval_points is None:
            eval_points = dists.UniformHypersphere(ens.dimensions).sample(
                ens.EVAL_POINTS, rng=rng) * ens.radius
        else:
            eval_points = _array2d(ens.eval_points, dtype=np.float64)
        self._data['eval_points'][ens] = eval_points

        # Set up signal
        self.sig_in[ens] = Signal(np.zeros(ens.dimensions),
                                  name=ens.label + ".signal")
        self.operators.append(Reset(self.sig_in[ens]))

        # Set up neurons

        ens.neurons.dimensions = ens.dimensions  # XXX ens modified
        self._data['gain'][ens.neurons] = ens.neurons.gain
        self._data['bias'][ens.neurons] = ens.neurons.bias
        if (self._data['gain'][ens.neurons] is None
                or self._data['bias'][ens.neurons] is None):
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            max_rates, intercepts = ens.max_rates, ens.intercepts
            if hasattr(max_rates, 'sample'):
                max_rates = max_rates.sample(ens.neurons.n_neurons, rng=rng)
            if hasattr(intercepts, 'sample'):
                intercepts = intercepts.sample(ens.neurons.n_neurons, rng=rng)
            gain, bias = ens.neurons.gain_bias(max_rates, intercepts)
            self._data['gain'][ens.neurons] = gain
            self._data['bias'][ens.neurons] = bias

        self.builders[ens.neurons.__class__](ens.neurons)

        # Set up encoders
        encoders = ens.encoders
        if encoders is None:
            if isinstance(ens.neurons, nengo.Direct):
                encoders = np.identity(ens.dimensions)
            else:
                sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
                encoders = sphere.sample(ens.neurons.n_neurons, rng=rng)
        else:
            encoders = np.array(encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (encoders.shape, enc_shape))

            encoders /= npext.norm(encoders, axis=1, keepdims=True)
        self._data['encoders'][ens] = encoders

        if isinstance(ens.neurons, nengo.Direct):
            scaled_encoders = encoders
        else:
            scaled_encoders = encoders * (self._data['gain'][ens.neurons]
                                          / ens.radius)[:, np.newaxis]
        self._data['scaled_encoders'][ens] = scaled_encoders
        self.operators.append(DotInc(
            Signal(scaled_encoders, name=ens.label + ".scaled_encoders"),
            self.sig_in[ens],
            self.sig_in[ens.neurons],
            tag=ens.label + ' encoding'))

        # Output is neural output
        self.sig_out[ens] = self.sig_out[ens.neurons]

        # Set up probes, but don't build them (done explicitly later)
        # Note: Have to set it up here because we only know these things
        #       (dimensions, n_neurons) at build time.
        for probe in ens.probes['decoded_output']:
            probe.dimensions = ens.dimensions
        for probe in ens.probes['spikes']:
            probe.dimensions = ens.n_neurons
        for probe in ens.probes['voltages']:
            probe.dimensions = ens.n_neurons

    @builds(nengo.Node)
    def build_node(self, node):
        # Get input
        if (node.output is None
                or isinstance(node.output, collections.Callable)):
            if node.size_in > 0:
                self.sig_in[node] = Signal(np.zeros(node.size_in),
                                           name=node.label + ".signal")
                # reset input signal to 0 each timestep
                self.operators.append(Reset(self.sig_in[node]))

        # Provide output
        if node.output is None:
            self.sig_out[node] = self.sig_in[node]
        elif not isinstance(node.output, collections.Callable):
            self.sig_out[node] = Signal(node.output, name=node.label)
        else:
            sig_in, sig_out = self.build_pyfunc(
                fn=node.output,
                t_in=True,
                n_in=node.size_in,
                n_out=node.size_out,
                label="%s.pyfn" % node.label)
            if node.size_in > 0:
                self.operators.append(DotInc(
                    self.sig_in[node],
                    Signal(1.0, name="1"),
                    sig_in,
                    tag=node.label + " input"))
            self.sig_out[node] = sig_out

        # Set up probes
        for probe in node.probes['output']:
            probe.dimensions = self.sig_out[node].shape

    @builds(nengo.Probe)
    def build_probe(self, probe):
        # Set up probe
        if probe.sample_every is None:
            probe.sample_every = self.dt
        probe.sig = Signal(np.zeros(probe.dimensions),
                           name=probe.label)
        # reset input signal to 0 each timestep
        self.operators.append(Reset(probe.sig))
        self.probes.append(probe)

    @staticmethod
    def filter_coefs(pstc, dt):
        pstc = max(pstc, dt)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)

    def _filtered_signal(self, signal, filter):
        name = signal.name + ".filtered(%f)" % filter
        filtered = Signal(np.zeros(signal.size), name=name)
        o_coef, n_coef = self.filter_coefs(pstc=filter, dt=self.dt)
        self.operators.append(ProdUpdate(
            Signal(n_coef, name="n_coef"),
            signal,
            Signal(o_coef, name="o_coef"),
            filtered,
            tag=name + " filtering"))
        return filtered

    @builds(nengo.Connection)  # noqa
    def build_connection(self, conn):
        dt = self.dt
        rng = np.random.RandomState(self._get_new_seed())

        self.sig_in[conn] = self.sig_out[conn.pre]
        self.sig_out[conn] = self.sig_in[conn.post]

        decoders = None
        transform = np.array(conn.transform_full, dtype=np.float64)

        # Figure out the signal going across this connection
        if (isinstance(conn.pre, nengo.Ensemble)
                and isinstance(conn.pre.neurons, nengo.Direct)):
            # Decoded connection in directmode
            if conn.function is None:
                signal = self.sig_in[conn]
            else:
                sig_in, signal = self.build_pyfunc(
                    fn=conn.function,
                    t_in=False,
                    n_in=self.sig_in[conn].size,
                    n_out=conn.dimensions,
                    label=conn.label)
                self.operators.append(DotInc(
                    self.sig_in[conn],
                    Signal(1.0, name="1"),
                    sig_in,
                    tag="%s input" % conn.label))
        elif isinstance(conn.pre, nengo.Ensemble):
            # Normal decoded connection
            eval_points = conn.eval_points
            if eval_points is None:
                eval_points = self._data['eval_points'][conn.pre]

            x = np.dot(eval_points,
                       self._data['encoders'][conn.pre].T / conn.pre.radius)
            activities = dt * conn.pre.neurons.rates(
                x, self._data['gain'][conn.pre.neurons],
                self._data['bias'][conn.pre.neurons])
            if conn.function is None:
                targets = eval_points
            else:
                targets = _array2d([conn.function(ep) for ep in eval_points])

            if conn.weight_solver is not None:
                if conn.decoder_solver is not None:
                    raise ValueError("Cannot specify both 'weight_solver' "
                                     "and 'decoder_solver'.")

                # account for transform
                targets = np.dot(targets, transform.T)
                transform = np.array(1., dtype=np.float64)

                decoders = conn.weight_solver(
                    activities, targets, rng=rng,
                    E=self._data['scaled_encoders'][conn.post].T)
                self.sig_out[conn] = self.sig_in[conn.post.neurons]
                signal_size = self.sig_out[conn].size
            else:
                solver = (conn.decoder_solver if conn.decoder_solver is
                          not None else nengo.decoders.lstsq_L2nz)
                decoders = solver(activities, targets, rng=rng)
                signal_size = conn.dimensions

            self._data['decoders'][conn] = decoders

            # Add operator for decoders and filtering
            decoders = decoders.T
            if conn.filter is not None and conn.filter > dt:
                o_coef, n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
                decoder_signal = Signal(
                    decoders * n_coef,
                    name=conn.label + ".decoders * n_coef")
            else:
                decoder_signal = Signal(decoders,
                                        name=conn.label + '.decoders')
                o_coef = 0

            signal = Signal(np.zeros(signal_size), name=conn.label)
            self.operators.append(ProdUpdate(
                decoder_signal,
                self.sig_in[conn],
                Signal(o_coef, name="o_coef"),
                signal,
                tag=conn.label + " filtering"))
        else:
            # Direct connection
            signal = self.sig_in[conn]

        # Add operator for filtering
        if decoders is None and conn.filter is not None and conn.filter > dt:
            signal = self._filtered_signal(signal, conn.filter)

        if conn.modulatory:
            # Make a new signal, effectively detaching from post
            self.sig_out[conn] = Signal(np.zeros(self.sig_out[conn].size),
                                        name=conn.label + ".mod_output")
            # Add reset operator?
            # XXX add unit test

        # Add operator for transform
        if isinstance(conn.post, nengo.objects.Neurons):
            transform *= self._data['gain'][conn.post][:, np.newaxis]
        self.operators.append(
            DotInc(Signal(transform, name=conn.label + ".transform"),
                   signal,
                   self.sig_out[conn],
                   tag=conn.label))
        self._data['transform'][conn] = transform

        # Set up probes
        for probe in conn.probes['signal']:
            probe.dimensions = self.sig_out[conn].size
            # XXX this should be done in model?
            self.model.add(probe)

    def build_pyfunc(self, fn, t_in, n_in, n_out, label):
        if n_in:
            sig_in = Signal(np.zeros(n_in), name=label + '.input')
            self.operators.append(Reset(sig_in))
        else:
            sig_in = None
        sig_out = Signal(np.zeros(n_out), name=label + '.output')
        self.operators.append(
            SimPyFunc(output=sig_out, fn=fn, t_in=t_in, x=sig_in))
        return sig_in, sig_out

    def build_neurons(self, neurons):
        self.sig_in[neurons] = Signal(np.zeros(neurons.n_in),
                                      name=neurons.label + '.input')
        self.sig_out[neurons] = Signal(np.zeros(neurons.n_out),
                                       name=neurons.label + '.output')
        bias_signal = Signal(self._data['bias'][neurons],
                             name=neurons.label + '.bias')
        self.operators.append(Copy(src=bias_signal, dst=self.sig_in[neurons]))

        # Set up probes
        for probe in neurons.probes['output']:
            probe.dimensions = neurons.n_neurons

    @builds(nengo.Direct)
    def build_direct(self, direct):
        self.sig_in[direct] = Signal(np.zeros(direct.dimensions),
                                     name=direct.label)
        self.sig_out[direct] = self.sig_in[direct]
        self.operators.append(Reset(self.sig_in[direct]))

    @builds(nengo.LIFRate)
    def build_lifrate(self, lif):
        if lif.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % lif.n_neurons)
        self.build_neurons(lif)
        self.operators.append(SimLIFRate(output=self.sig_out[lif],
                                         J=self.sig_in[lif],
                                         nl=lif))

    @builds(nengo.LIF)
    def build_lif(self, lif):
        if lif.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % lif.n_neurons)
        self.build_neurons(lif)
        voltage = Signal(np.zeros(lif.n_neurons), name=lif.label + ".voltage")
        refractory_time = Signal(np.zeros(lif.n_neurons),
                                 name=lif.label + ".refractory_time")
        self.operators.append(SimLIF(output=self.sig_out[lif],
                                     J=self.sig_in[lif],
                                     nl=lif,
                                     voltage=voltage,
                                     refractory_time=refractory_time))
