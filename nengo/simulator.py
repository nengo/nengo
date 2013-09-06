import logging
import itertools
from collections import defaultdict
import time

import networkx
import numpy as np

from objects import LIF, LIFRate, Direct

def are_aliased(a, b):
    # Terminology: two arrays *overlap* if the lowermost memory addressed
    # touched by upper one is higher than the uppermost memory address touched
    # by the lower one.
    # 
    # np.may_share_memory returns True iff there is overlap.
    # Overlap is a necessary but insufficient condition for *aliasing*.
    #
    # Aliasing is when two ndarrays refer a common memory location.
    #

    # -- least and greatest addresses don't even overlap
    if not np.may_share_memory(a, b):
        return False

    a_info = a.__array_interface__
    b_info = b.__array_interface__
    if a.dtype != b.dtype:
        raise NotImplementedError()
    #a_data = a_info['data']
    #b_data = b_info['data']
    #a_shape = a_info['shape']
    #b_shape = b_info['shape']
    a_strides = a_info['strides']
    b_strides = b_info['strides']

    if a_strides == b_strides == None:
        # -- a and b are both contiguous blocks
        #    if they didn't overlap then
        #    np.may_share_memory would have returned False
        #    It didn't -> they overlap -> they are aliased.
        return  True
    else:
        raise NotImplementedError('are_aliased?', (a_info, b_info))


def foo(sim, go_on_init):
    nxdg = networkx.DiGraph()

    operators = []

    class SigBuf(object):
        def __init__(self, dct, sig):
            self.dct = dct
            self.sig = sig

            if dct is sim._output_signals:
                if sig.base not in dct:
                    dct[sig.base] = sim.signals[sig.base].copy()

        def __hash__(self):
            return hash( (id(self.dct), self.sig))

        def __eq__(self, other):
            # XXX signal equality should be implemented to match
            #     base objects, strides, etc.
            return self.dct is other.dct and self.sig == other.sig

        def is_aliased_with(self, other):
            if self.dct is other.dct:
                if self is other:
                    return True
                elif self.sig.same_view_as(other.sig):
                    return True
                else:
                    return are_aliased(self.get(), other.get())
            else:
                return False

        def get(self):
            return get_signal(self.dct, self.sig)

        def __str__(self):
            dctname = {
                id(sim.signals): 'signals',
                id(sim.signals_tmp): 'signals_tmp',
                id(sim.signals_copy): 'signals_copy',
                id(sim._output_signals): 'output_signals',
                    }[id(self.dct)]
            return 'SigBuf(%s, %s)' % (dctname, self.sig)


    class Reset(object):
        def __init__(self, dst):
            self.dst = dst

            nxdg.add_edge(self, dst)

            operators.append(self)

            self.reads = []
            self.sets = [dst]
            self.incs = []

            if go_on_init:
                self.go()

        def __str__(self):
            return 'Reset(%s)' % str(self.dst)

        def go(self):
            self.dst.get()[...] = 0

    class Copy(object):
        def __init__(self, dst, src):
            self.dst = dst
            self.src = src

            nxdg.add_edge(src, self)
            nxdg.add_edge(self, dst)

            operators.append(self)

            self.reads = [src]
            self.sets = [dst]
            self.incs = []

            if go_on_init:
                self.go()

        def __str__(self):
            return 'Copy(%s -> %s)' % (
                    str(self.src), str(self.dst))

        def go(self):
            self.dst.get()[...] = self.src.get()

    class DotInc(object):
        def __init__(self, A, X, Y, xT=False):
            self.A = A
            self.X = X
            self.Y = Y
            self.xT = xT

            nxdg.add_edge(self.A, self)
            nxdg.add_edge(self.X, self)
            nxdg.add_edge(self, self.Y)

            operators.append(self)

            self.reads = [self.A, self.X]
            self.sets = []
            self.incs = [self.Y]

            if go_on_init:
                self.go()

        def __str__(self):
            return 'DotInc(%s, %s -> %s)' % (
                    str(self.A), str(self.X), str(self.Y))

        def go(self):
            dot_inc(self.A.get(),
                    self.X.get().T if self.xT else self.X.get(),
                    self.Y.get())

    class NonLin(object):
        def __init__(self, output, J, nl, dt):
            self.output = output
            self.J = J
            self.nl = nl
            self.dt = dt

            nxdg.add_edge(J, self)
            nxdg.add_edge(self, output)

            operators.append(self)

            self.reads = [J]
            self.sets = [output]
            self.incs = []

            if go_on_init:
                self.go()

        def go(self):
            self.nl.step(
                dt=self.dt,
                J=self.J.get(),
                output=self.output.get())

    class DGCLS(object):
        @property
        def operators(self):
            return operators

        def add_constraints(self, verbose=False):

            # -- all views of a base object in a particular dictionary
            t0 = time.time()
            if verbose: print 'building alias table ...'
            by_base = defaultdict(list)
            reads = defaultdict(list)
            sets = defaultdict(list)
            incs = defaultdict(list)

            for op in self.operators:
                for node in op.reads + op.sets + op.incs:
                    by_base[SigBuf(node.dct, node.sig.base)].append(node)

                for node in op.reads:
                    reads[node].append(op)

                for node in op.sets:
                    sets[node].append(op)

                for node in op.incs:
                    incs[node].append(op)

            for node in sets:
                assert len(sets[node]) == 1, (node, sets[node])

            aliased = defaultdict(lambda : False)
            aliased_with = defaultdict(set)
            if verbose: print 'bases done ...'
            for ii, (sb, views) in enumerate(by_base.items()):
                if len(views) > 10:
                    if verbose: print '%i / %i ' % (ii, len(views))
                for v1, v2 in itertools.combinations(views, 2):
                    is_aliased = v1.is_aliased_with(v2)
                    if is_aliased:
                        aliased[(v1, v2)] = True
                        aliased[(v2, v1)] = True
                        aliased_with[v1].add(v2)
                        aliased_with[v2].add(v1)
                if len(views) > 10:
                    if verbose: print '%i / %i ' % (ii, len(views)), 'done'
            t1 = time.time()
            if verbose: print 'building alias table took', (t1 - t0)
            #self.sets_before_incs()
            #self.add_aliasing_restrictions()

            # assert that for every node (a) that is set
            # there are no other signals (b) that are
            # aliased to (a) and also set.
            for node, other in itertools.combinations(sets, 2):
                assert not aliased[(node, other)]

            # reads depend on sets and incs
            for node, post_ops in reads.items():
                pre_ops = sets[node] + incs[node]
                for other in aliased_with[node]:
                    pre_ops += sets[other] + incs[other]
                for pre_op, post_op in itertools.product(pre_ops, post_ops):
                    nxdg.add_edge(pre_op, post_op)

            # incs depend on sets
            for node, post_ops in incs.items():
                pre_ops = sets[node]
                for other in aliased_with[node]:
                    pre_ops += sets[other]
                for pre_op, post_op in itertools.product(pre_ops, post_ops):
                    nxdg.add_edge(pre_op, post_op)

        def eval_order(self):
            return [node for node in networkx.topological_sort(nxdg)
                    if not isinstance(node, SigBuf)]
    DG = DGCLS()

    return DG, SigBuf, Copy, DotInc, NonLin, Reset



logger = logging.getLogger(__name__)

class SimDirect(object):
    def __init__(self, nl):
        self.nl = nl

    def step(self, dt, J, output):
        output[...] = self.nl.fn(J)


class SimLIF(object):
    def __init__(self, nl):
        self.nl = nl
        self.voltage = np.zeros(nl.n_in)
        self.refractory_time = np.zeros(nl.n_in)

    def step(self, dt, J, output):
        self.nl.step_math0(dt, J, self.voltage, self.refractory_time, output)


class SimLIFRate(object):
    def __init__(self, nl):
        self.nl = nl

    def step(self, dt, J, output):
        output[:] = dt * self.nl.rates(J - self.nl.bias)


registry = {
    LIF: SimLIF,
    LIFRate: SimLIFRate,
    Direct: SimDirect,
}

def get_signal(signals_dct, obj):
    # look up a Signal or SignalView
    # in a `signals_dct` such as self.signals
    if obj in signals_dct:
        return signals_dct[obj]
    elif obj.base in signals_dct:
        base_array = signals_dct[obj.base]
        try:
            # wtf?
            itemsize = int(obj.dtype.itemsize)
        except TypeError:
            itemsize = int(obj.dtype().itemsize)
        byteoffset = itemsize * obj.offset
        bytestrides = [itemsize * s for s in obj.elemstrides]
        view = np.ndarray(shape=obj.shape,
                          dtype=obj.dtype,
                          buffer=base_array.data,
                          offset=byteoffset,
                          strides=bytestrides,
                         )
        view[...]
        return view
    else:
        raise TypeError()


def dot_inc(a, b, targ):
    # -- we check for size mismatch,
    #    because incrementing scalar to len-1 arrays is ok
    #    if the shapes are not compatible, we'll get a
    #    problem in targ[...] += inc
    try:
        inc =  np.dot(a, b)
    except Exception, e:
        e.args = e.args + (a.shape, b.shape)
        raise
    if inc.shape != targ.shape:
        if inc.size == targ.size == 1:
            inc = np.asarray(inc).reshape(targ.shape)
        else:
            raise ValueError('shape mismatch', (inc.shape, targ.shape))
    targ[...] += inc


class Simulator(object):
    def __init__(self, model):
        self.model = model

        self.n_steps = 0
        self.signals = {}
        self.signals_tmp = {}
        self.signals_copy = {}
        self.nonlinearities = {}
        self.probe_outputs = {}
        self.constant_signals = []
        self.dynamic_signals = []

        for sig in self.model.signals:
            if hasattr(sig, 'value'):
                self.signals[sig] = np.asarray(sig.value)
                self.signals_tmp[sig] = np.asarray(sig.value)
                self.signals_copy[sig] = np.asarray(sig.value)
                self.constant_signals.append(sig)
            else:
                self.signals[sig] = np.zeros(sig.n)
                self.signals_tmp[sig] = np.zeros(sig.n)
                self.signals_copy[sig] = np.zeros(sig.n)
                self.dynamic_signals.append(sig)

        for pop in self.model.nonlinearities:
            self.nonlinearities[pop] = registry[pop.__class__](pop)

        for probe in self.model.probes:
            self.probe_outputs[probe] = []

        dg, SigBuf, Copy, DotInc, NonLin, Reset = foo(self,
                                                      go_on_init=False)
        output_signals = {}
        self._output_signals = output_signals

        # -- reset: 0 -> signals_tmp
        nl_outputs = set(nl.output_signal for nl in self.model.nonlinearities)
        nl_inputs = set(nl.input_signal for nl in self.model.nonlinearities)
        for sig in self.dynamic_signals:
            if sig not in nl_outputs and sig not in nl_inputs:
                Reset(SigBuf(self.signals_tmp, sig))

        # -- reset nonlinearities: bias -> input_current
        for nl in self.model.nonlinearities:
            Copy(SigBuf(self.signals_tmp, nl.input_signal,),
                 SigBuf(self.signals, nl.bias_signal))

        # -- encoders: signals -> input current
        #    (N.B. this includes neuron -> neuron connections)
        for enc in self.model.encoders:
            DotInc(SigBuf(self.signals, enc.sig),
                   SigBuf(self.signals, enc.weights_signal),
                   SigBuf(self.signals_tmp, enc.pop.input_signal),
                   xT=True)

        # -- population dynamics
        for nl in self.model.nonlinearities:
            pop = self.nonlinearities[nl]
            NonLin(output=SigBuf(self.signals_tmp, nl.output_signal),
                   J=SigBuf(self.signals_tmp, nl.input_signal),
                   nl=pop,
                   dt=self.model.dt)

        # -- decoders: population output -> signals_tmp
        for dec in self.model.decoders:
            DotInc(SigBuf(self.signals_tmp, dec.pop.output_signal),
                   SigBuf(self.signals, dec.weights_signal),
                   SigBuf(self.signals_tmp, dec.sig),
                   xT=True)

        # -- copy: signals -> signals_copy
        for sig in self.dynamic_signals:
            Copy(SigBuf(self.signals_copy, sig),
                 SigBuf(self.signals, sig))

        # -- reset: 0 -> signals
        for sig in self.dynamic_signals:
            Reset(SigBuf(output_signals, sig))

        # -- filters: signals_copy -> signals
        for filt in self.model.filters:
            try:
                DotInc(SigBuf(self.signals, filt.alpha_signal),
                       SigBuf(self.signals_copy, filt.oldsig),
                       SigBuf(output_signals, filt.newsig))
            except Exception, e:
                e.args = e.args + (filt.oldsig, filt.newsig)
                raise

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            DotInc(SigBuf(self.signals, tf.alpha_signal),
                   SigBuf(self.signals_tmp, tf.insig),
                   SigBuf(output_signals, tf.outsig))

        self.dg = dg
        # -- XXX: speed up to some reasonable level
        self.dg.add_constraints()


    def step(self):
        for op in self.dg.eval_order():
            op.go()

        for k, v in self._output_signals.items():
            self.signals[k][...] = v

        # -- probes signals -> probe buffers
        for probe in self.model.probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                tmp = get_signal(self.signals, probe.sig).copy()
                self.probe_outputs[probe].append(tmp)

        self.n_steps += 1

    def run_steps(self, N):
        for i in xrange(N):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()

    def probe_data(self, probe):
        return np.asarray(self.probe_outputs[probe])
