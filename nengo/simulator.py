import logging

import numpy as np

from .core import LIF, LIFRate, Direct


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
        # -- reset nonlinearities: bias -> input_current
        for nl in self.model.nonlinearities:
            self.signals[nl.input_signal][...] = self.signals[nl.bias_signal]

        # -- encoders: signals -> input current
        #    (N.B. this includes neuron -> neuron connections)
        for enc in self.model.encoders:
            dot_inc(get_signal(self.signals, enc.sig),
                    enc.weights.T,
                    self.signals[enc.pop.input_signal])

        # -- reset: 0 -> signals_tmp
        for sig in self.dynamic_signals:
            self.signals_tmp[sig][...] = 0

        # -- population dynamics
        for nl in self.model.nonlinearities:
            pop = self.nonlinearities[nl]
            pop.step(dt=self.model.dt,
                     J=self.signals[nl.input_signal],
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

        # -- filters: signals_copy -> signals
        for filt in self.model.filters:
            try:
                dot_inc(filt.alpha,
                        get_signal(self.signals_copy, filt.oldsig),
                        get_signal(self.signals, filt.newsig))
            except Exception, e:
                e.args = e.args + (filt.oldsig, filt.newsig)
                raise

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            dot_inc(tf.alpha,
                    get_signal(self.signals_tmp, tf.insig),
                    get_signal(self.signals, tf.outsig))

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
