"""
Low-level objects
=================

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

"""
import copy
import logging

import numpy as np
import simulator as sim


logger = logging.getLogger(__name__)


"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


def filter_coefs(pstc, dt):
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
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
        return np.dtype(self.base._dtype)

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
            raise TODO('reshape of strided view')

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

    def add_to_model(self, model):
        if self.base not in model.signals:
            raise TypeError("Cannot add signal views. Add the signal instead.")

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'base': self.base.name,
            'shape': list(self.shape),
            'elemstrides': list(self.elemstrides),
            'offset': self.offset,
        }

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
    def __init__(self, n=1, dtype=np.float64, name=None):
        self.n = n
        self._dtype = dtype
        if name is not None:
            self._name = name
        if assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", " + str(self.n) + "D)"
        except AttributeError:
            return "Signal (id " + str(id(self)) + ", " + str(self.n) + "D)"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return (self.n,)

    @property
    def size(self):
        return self.n

    @property
    def elemstrides(self):
        return (1,)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

    def add_to_model(self, model):
        model.signals.append(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'n': self.n,
            'dtype': str(self.dtype),
        }


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return "Probing " + str(self.sig)

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.probes.append(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'dt': self.dt,
        }


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, value, name=None):
        self.value = np.asarray(value)

        Signal.__init__(self, self.value.size, name=name)

    def __str__(self):
        if self.name is not None:
            return "Constant(" + self.name + ")"
        return "Constant(id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'value': self.value.tolist(),
        }


def is_signal(sig):
    return isinstance(sig, SignalView)


def is_constant(sig):
    """
    Return True iff `sig` is (or is a view of) a Constant signal.
    """
    return isinstance(sig.base, Constant)


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
        def step():
            # -- we check for size mismatch,
            #    because incrementing scalar to len-1 arrays is ok
            #    if the shapes are not compatible, we'll get a
            #    problem in Y[...] += inc
            try:
                inc =  np.dot(A, X)
            except Exception, e:
                e.args = e.args + (A.shape, X.shape)
                raise
            if inc.shape != Y.shape:
                if inc.size == Y.size == 1:
                    inc = np.asarray(inc).reshape(Y.shape)
                else:
                    raise ValueError('shape mismatch in %s %s x %s -> %s' % (
                        self.tag, self.A, self.X, self.Y), (
                        A.shape, X.shape, inc.shape, Y.shape))
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

        def step():
            val = np.dot(A,X)
            if val.shape != Y.shape:
                if val.size == Y.size == 1:
                    val = np.asarray(val).reshape(Y.shape)
                else:
                    raise ValueError('shape mismatch in %s (%s vs %s)' %
                                     (self.tag, val, Y))
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
