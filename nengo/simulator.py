"""
Simulator.py

Reference simulator for nengo models.
"""
import logging
import itertools
from collections import defaultdict
import time

import networkx
import networkx as nx
import numpy as np

import core

logger = logging.getLogger(__name__)


def is_base(sig):
    return sig.base == sig


def is_view(sig):
    return not is_base(sig)


class SignalDict(dict):
    """
    Map from Signal -> ndarray

    SignalDict overrides __getitem__ for two reasons:
    1. so that scalars are returned as 0-d ndarrays
    2. so that a SignalView lookup returns a views of its base

    """
    def __getitem__(self, obj):
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
                              strides=bytestrides,
                             )
            return view
        else:
            raise KeyError(obj)


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
        rates_fn = self.nl.rates
        bias = self.nl.bias
        def step():
            output[...] = dt * rates_fn(J - bias)
        return step


class Simulator(object):
    """Reference simulator for models.
    """
    def __init__(self, model):
        if not hasattr(model, 'dt'):
            raise ValueError("Model does not appear to be built. "
                             "See Model.prep_for_simulation.")

        # -- map from Signal.base -> ndarray
        self._sigdict = SignalDict()
        self.model = model
        for op in model._operators:
            op.init_sigdict(self._sigdict, model.dt)

        self.dg = self._init_dg()
        self._step_order = [node
            for node in networkx.topological_sort(self.dg)
            if hasattr(node, 'make_step')]
        self._steps = [node.make_step(self._sigdict, model.dt)
            for node in self._step_order]

        self.n_steps = 0
        self.probe_outputs = dict((probe, []) for probe in model.probes)

    def _init_dg(self, verbose=False):
        operators = self.model._operators
        dg = networkx.DiGraph()

        for op in operators:
            dg.add_edges_from(itertools.product(op.reads + op.updates, [op]))
            dg.add_edges_from(itertools.product([op], op.sets + op.incs))

        # -- all views of a base object in a particular dictionary
        by_base_writes = defaultdict(list)
        by_base_reads = defaultdict(list)
        reads = defaultdict(list)
        sets = defaultdict(list)
        incs = defaultdict(list)
        ups = defaultdict(list)

        for op in operators:
            for node in op.sets + op.incs:
                by_base_writes[node.base].append(node)

            for node in op.reads:
                by_base_reads[node.base].append(node)

            for node in op.reads:
                reads[node].append(op)

            for node in op.sets:
                sets[node].append(op)

            for node in op.incs:
                incs[node].append(op)

            for node in op.updates:
                ups[node].append(op)

        # -- assert that only one op sets any particular view
        for node in sets:
            assert len(sets[node]) == 1, (node, sets[node])

        # -- assert that only one op updates any particular view
        for node in ups:
            assert len(ups[node]) == 1, (node, ups[node])

        # --- assert that any node that is incremented is also set/updated
        for node in incs:
            assert len(sets[node]+ups[node]) > 0, (node)

        # -- assert that no two views are both set and aliased
        if len(sets) >= 2:
            for node, other in itertools.combinations(sets, 2):
                assert not node.shares_memory_with(other), \
                    ("%s shares memory with %s" % (node, other))

        # -- assert that no two views are both updated and aliased
        if len(ups) >= 2:
            for node, other in itertools.combinations(ups, 2):
                assert not node.shares_memory_with(other), (
                        node, other)

        # -- Scheduling algorithm for serial evaluation:
        #    1) All sets on a given base signal
        #    2) All incs on a given base signal
        #    3) All reads on a given base signal
        #    4) All updates on a given base signal

        # -- incs depend on sets
        for node, post_ops in incs.items():
            pre_ops = list(sets[node])
            for other in by_base_writes[node.base]:
                pre_ops += sets[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        # -- reads depend on writes (sets and incs)
        for node, post_ops in reads.items():
            pre_ops = sets[node] + incs[node]
            for other in by_base_writes[node.base]:
                pre_ops += sets[other] + incs[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        # -- updates depend on reads, sets, and incs.
        for node, post_ops in ups.items():
            pre_ops = sets[node] + incs[node] + reads[node]
            for other in by_base_writes[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            for other in by_base_reads[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        return dg

    @property
    def signals(self):
        """Support access to current ndarrays via `self.signals[sig]`.

        Here `sig` can be a signal within the model used to generate this
        simulator, even though that model was deepcopied in the process of
        generating the simulator.

        This property is also used to implement a pretty-printing algorithm so
        that `print sim.signals` returns a multiline string.
        """
        class Accessor(object):
            def __getitem__(_, item):
                try:
                    return self._sigdict[item]
                except KeyError, e:
                    try:
                        return self._sigdict[self.copied(item)]
                    except KeyError:
                        raise e  # -- re-raise the original KeyError

            def __setitem__(_, item, val):
                try:
                    self._sigdict[item][...] = val
                except KeyError, e:
                    try:
                        self._sigdict[self.copied(item)][...] = val
                    except KeyError:
                        raise e  # -- re-raise the original KeyError

            def __iter__(_):
                return self._sigdict.__iter__()

            def __len__(_):
                return self._sigdict.__len__()

            def __str__(_):
                import StringIO
                sio = StringIO.StringIO()
                for k in self._sigdict:
                    print >> sio, k, self._sigdict[k]
                return sio.getvalue()

        return Accessor()

    def step(self):
        """Advance the simulator by `self.model.dt` seconds.
        """
        for step_fn in self._steps:
            step_fn()

        # -- probes signals -> probe buffers
        for probe in self.model.probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                tmp = self._sigdict[probe.sig].copy()
                self.probe_outputs[probe].append(tmp)

        self.n_steps += 1

    def copied(self, obj):
        """Get the simulator's copy of a model object.

        Parameters
        ----------
        obj : Nengo object
            A model from the original model

        Returns
        -------
        sim_obj : Nengo object
            The simulator's copy of `obj`.

        Examples
        --------
        Manually set a raw signal value to ``5`` in the simulator
        (advanced usage). [TODO: better example]

        >>> model = nengo.Model()
        >>> foo = m.add(Signal(n=1))
        >>> sim = model.simulator()
        >>> sim.signals[sim.copied(foo)] = np.asarray([5])
        """
        return self.model.memo[id(obj)]

    def reset(self):
        """TODO"""
        raise NotImplementedError

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.model.dt))
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.model.name, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        for i in xrange(steps):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()

    def data(self, probe):
        """Get data from signals that have been probed.

        Parameters
        ----------
        probe : Probe
            TODO

        Returns
        -------
        data : ndarray
            TODO: what are the dimensions?
        """
        if not isinstance(probe, core.Probe):
            if isinstance(probe, str):
                probe = self.model.probed[probe]
            else:
                probe = self.model.probed[self.model.memo[id(probe)]]
        return np.asarray(self.probe_outputs[probe])

    def probe_data(self, probe):
        """TODO
        """
        return np.asarray(self.probe_outputs[probe])
