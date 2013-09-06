import logging

import numpy as np

from objects import LIF, LIFRate, Direct


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

    def step(self):
        class NonLin(object):
            def __init__(self, *args):
                raise NotImplementedError()
                self.__dict__.update(locals())
                del self.self
                dg.add_node(self)



        import networkx

        dg = networkx.DiGraph()

        operators = []

        class SigBuf(object):
            def __init__(self, sig, buf):
                self.sig = sig
                self.buf = buf

            def may_share_memory(self, other):
                return np.may_share_memory(
                    get_signal(self.buf, self.sig),
                    get_signal(other.buf, other.sig),
                    )

        output_signals = dict()

        class Assign(object):
            def __init__(self, dst_buf, dst_sig, src_buf, src_sig):

                self.dst = SigBuf(dst_buf, dst_sig)
                self.src = SigBuf(src_buf, src_sig)

                dg.add_edge(self.src, self)
                dg.add_edge(self, self.dst)

                operators.append(self)

                self.reads = [self.src]
                self.writes = [self.dst]

        class DotInc(object):
            def __init__(self, Abuf, Asig, Xbuf, Xsig, Ybuf, Ysig):
                self.__dict__.update(locals())
                del self.self

                self.A = SigBuf(Abuf, Asig)
                self.X = SigBuf(Xbuf, Xsig)
                self.Y = SigBuf(Ybuf, Ysig)

                self.reads = [self.A, self.X]
                self.writes = [self.Y]

                dg.add_edge(self.A, self)
                dg.add_edge(self.X, self)
                dg.add_edge(self, self.Y)

                operators.append(self)


        # -- add edges for memory alias dependencies
        #    TODO: make this more efficient
        for node in dg.nodes():
            if isinstance(node, tuple):
                buf, sig = node
                # -- all operators that read *from*
                #    get_signal(buf, sig)
                #    depend on all operators that output to
                #    an area with overlap with get_signal(buf, sig)


        # -- reset: 0 -> signals_tmp
        for sig in self.dynamic_signals:
            self.signals_tmp[sig][...] = 0

        # -- reset nonlinearities: bias -> input_current
        for nl in self.model.nonlinearities:
            self.signals_tmp[nl.input_signal][...] = self.signals[nl.bias_signal]
            #Assign(self.signals, nl.input_signal, self.signals, nl.bias_signal)

        # -- encoders: signals -> input current
        #    (N.B. this includes neuron -> neuron connections)
        for enc in self.model.encoders:
            dot_inc(get_signal(self.signals, enc.sig),
                    enc.weights.T,
                    self.signals_tmp[enc.pop.input_signal])

            #DotInc(Abuf=self.signals, Asig=enc.sig,
            #       Xbuf=self.signals, Xsig=enc.weights_signal, #.T?
            #       Ybuf=self.signals, Ysig=enc.pop.input_signal)

        # -- population dynamics
        for nl in self.model.nonlinearities:
            pop = self.nonlinearities[nl]
            pop.step(dt=self.model.dt,
                     J=self.signals_tmp[nl.input_signal],
                     output=self.signals_tmp[nl.output_signal])

        # -- decoders: population output -> signals_tmp
        for dec in self.model.decoders:
            dot_inc(self.signals_tmp[dec.pop.output_signal],
                    dec.weights.T,
                    get_signal(self.signals_tmp, dec.sig))

        # -- copy: signals -> signals_copy
        for sig in self.dynamic_signals:
            self.signals_copy[sig][...] = self.signals[sig]

        # -- reset: 0 -> signals
        for sig in self.dynamic_signals:
            self.signals[sig][...] = 0

        # -- hack to keep signals up to date with signals_tmp for tests
        for nl in self.model.nonlinearities:
            self.signals[nl.input_signal][...] = self.signals_tmp[nl.input_signal]
            Assign(SigBuf(output_signals, nl.input_signal),
                   SigBuf(self.signals_tmp, nl.input_signal))

            self.signals[nl.output_signal][...] = self.signals_tmp[nl.output_signal]
            Assign(SigBuf(output_signals, nl.output_signal),
                   SigBuf(self.signals_tmp, nl.output_signal))

        for enc in self.model.encoders:
            self.signals[enc.pop.input_signal][...] = self.signals_tmp[enc.pop.input_signal]
            Assign(SigBuf(output_signals, enc.pop.input_signal),
                   SigBuf(self.signals_tmp, enc.pop.input_signal))


        # -- filters: signals_copy -> signals
        for filt in self.model.filters:
            try:
                dot_inc(filt.alpha,
                        get_signal(self.signals_copy, filt.oldsig),
                        get_signal(self.signals, filt.newsig))
                DotInc(filt.alpha,
                       SigBuf(self.signals_copy, filt.oldsig),
                       SigBuf(output_signals, filt.newsig))
            except Exception, e:
                e.args = e.args + (filt.oldsig, filt.newsig)
                raise

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            dot_inc(tf.alpha,
                    get_signal(self.signals_tmp, tf.insig),
                    get_signal(self.signals, tf.outsig))

            DotInc(tf.alpha, #Abuf=self.signals, Asig=enc.sig,
                   self.signals_tmp, tf.insig, #.T?
                   SigBuf(output_signals, tf.outsig))

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
