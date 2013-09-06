import logging

import networkx
import numpy as np

from objects import LIF, LIFRate, Direct


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

        def may_share_memory(self, other):
            return np.may_share_memory(
                get_signal(self.dct, self.sig),
                get_signal(other.dct, other.sig))

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

        def add_constraints(self):
            self.sets_before_incs()
            self.add_aliasing_restrictions()

        def sets_before_incs(self):
            for op1 in operators:
                for op2 in operators:
                    if op1 is op2:
                        continue
                    for w1 in op1.sets:
                        for w2 in op2.incs:
                            if w1.may_share_memory(w2):
                                nxdg.add_edge(op1, op2)

        def add_aliasing_restrictions(self):
            # -- all operators that read *from*
            #    get_signal(buf, sig)
            #    depend on all operators that output to
            #    an area with overlap with get_signal(buf, sig)

            # -- add edges for memory alias dependencies
            #    TODO: make this more efficient
            for op1 in operators:
                for op2 in operators:
                    if op1 is op2:
                        continue
                    for w1 in op1.sets + op1.incs:
                        for r2 in op2.reads:
                            if w1.may_share_memory(r2):
                                nxdg.add_edge(op1, op2)
                                try:
                                    networkx.topological_sort(nxdg)
                                except:
                                    print 'sig', id(sim.signals)
                                    print 'outsig', id(sim._output_signals)
                                    print 'sig_copy', id(sim.signals_copy)
                                    print 'w dct', id(op1.X.dct)
                                    print 'op2 dct', id(op1.X.dct)

                                    print 'edge', op1, op2
                                    print w1.sig, r2.sig
                                    print 'a', op1.A.sig
                                    print 'a', op1.X.sig
                                    print 'a', op1.Y.sig
                                    print 'b', op2.src.sig
                                    print 'b', op2.dst.sig
                                    raise

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
        for sig in self.dynamic_signals:
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
        if 0:
            # -- This is still not right
            for op in self.dg.eval_order():
                op.go()
        else:
            for op in self.dg.operators:
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
