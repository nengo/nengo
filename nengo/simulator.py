"""
Simulator.py

Reference simulator for nengo models.
"""

from __future__ import print_function

from collections import defaultdict, Mapping
import itertools
import logging

import networkx as nx
import numpy as np

import nengo.builder

logger = logging.getLogger(__name__)


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
                              strides=bytestrides)
            return view
        else:
            raise KeyError(obj)


class ProbeDict(Mapping):
    """
    Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.

    """

    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, key):
        rval = self.raw[key]
        if isinstance(rval, dict):
            return rval
        rval = np.asarray(self.raw[key])
        rval.flags.writeable = False
        return rval

    def __str__(self):
        return str(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)


class Simulator(object):
    """Reference simulator for models."""

    def __init__(self, model, dt=0.001, seed=None, builder=None):
        if builder is None:
            # By default, we'll use builder.Builder and copy the model.
            builder = nengo.builder.Builder()

        # Call the builder to build the model
        self.model = model
        self.__dict__.update(builder(model, dt))

        # Note: seed is not used right now, but one day...
        if seed is not None:
            self.seed = seed  # Use simulator seed instead of model seed

        # -- map from Signal.base -> ndarray
        self._sigdict = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        for op in self.operators:
            op.init_sigdict(self._sigdict, self.dt)

        self.dg = self._init_dg()
        self._step_order = [node
                            for node in nx.topological_sort(self.dg)
                            if hasattr(node, 'make_step')]
        self._steps = [node.make_step(self._sigdict, self.dt)
                       for node in self._step_order]

        self.n_steps = 0
        self._data.update(dict((probe, []) for probe in self.probes))
        self.data = ProbeDict(self._data)

    def _init_dg(self, verbose=False):  # noqa
        operators = self.operators
        dg = nx.DiGraph()

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
            assert len(sets[node] + ups[node]) > 0, (node)

        # -- assert that no two views are both set and aliased
        if len(sets) >= 2:
            for node, other in itertools.combinations(sets, 2):
                assert not node.shares_memory_with(other), \
                    ("%s shares memory with %s" % (node, other))

        # -- assert that no two views are both updated and aliased
        if len(ups) >= 2:
            for node, other in itertools.combinations(ups, 2):
                assert not node.shares_memory_with(other), (node, other)

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
                return self._sigdict[item]

            def __setitem__(_, item, val):
                self._sigdict[item][...] = val

            def __iter__(_):
                return self._sigdict.__iter__()

            def __len__(_):
                return self._sigdict.__len__()

            def __str__(_):
                import io
                sio = io.StringIO()
                for k in self._sigdict:
                    print_function(k, self._sigdict[k], file=sio)
                return sio.getvalue()

        return Accessor()

    def step(self):
        """Advance the simulator by `self.dt` seconds.
        """
        old_err = np.seterr(invalid='raise', divide='ignore')
        try:
            for step_fn in self._steps:
                step_fn()
        finally:
            np.seterr(**old_err)

        # -- probes signals -> probe buffers
        for probe in self.probes:
            period = int(probe.dt / self.dt)
            if self.n_steps % period == 0:
                tmp = self._sigdict[probe.sig].copy()
                self._data[probe].append(tmp)

        self._sigdict['__time__'] += self.dt
        self.n_steps += 1

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.label, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        for i in range(steps):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        last_t = self._sigdict['__time__'] - self.dt
        n_steps = self.n_steps if dt is None else int(
            self.n_steps / (dt / self.dt))
        return np.linspace(0, last_t, n_steps)
