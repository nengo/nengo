import numpy as np

from neuron import lif_step

class Simulator(object):
    def __init__(self, model):
        self.model = model

        self.n_steps = 0
        self.signals = {}
        self.signals_tmp = {}
        self.signals_copy = {}
        self.populations = {}
        self.probe_outputs = {}

        for sig in self.model.signals:
            self.signals[sig] = np.zeros(sig.n)
            self.signals_tmp[sig] = np.zeros(sig.n)
            self.signals_copy[sig] = np.zeros(sig.n)

        for pop in self.model.populations:
            self.populations[pop] = {
                'ic': np.zeros(pop.n),
                'v': np.zeros(pop.n),
                'rt': np.zeros(pop.n),
                'out': np.zeros(pop.n),
                'jbias': np.zeros(pop.n),
            }

        for probe in self.model.signal_probes:
            self.probe_outputs[probe] = []

    def step(self):
        for pop in self.model.populations:
            self.populations[pop]['ic'] = 1.0 * self.populations[pop]['jbias']

        for enc in self.model.encoders:
            self.populations[enc.pop]['ic'] += np.dot(self.signals[enc.sig],
                                                      enc.weights.T)

        for pop in self.model.populations:
            lif_step(
                self.populations[pop]['ic'],
                self.populations[pop]['v'],
                self.populations[pop]['rt'],
                self.populations[pop]['out'],
                dt=self.model.dt,
                #tau_rc=pop.tau_rc,
                #tau_ref=pop.tau_ref,
                tau_rc=0.02,
                tau_ref=0.002,
                upsample=1,
                )

        for dec in self.model.decoders:
            self.signals_tmp[dec.sig] = np.dot(self.populations[dec.pop]['out'],
                                               dec.weights.T)

        for sig in self.model.signals:
            self.signals_copy[sig] = self.signals[sig]

        for filt in self.model.filters:
            self.signals[filt.newsig] = \
                    filt.alpha * self.signals_copy[filt.oldsig]

        for tf in self.model.transforms:
            self.signals[tf.outsig] += tf.alpha * self.signals_tmp[tf.insig]

        for ct in self.model.custom_transforms:
            self.signals[ct.outsig] = ct.func(self.signals[ct.insig])

        for probe in self.model.signal_probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                self.probe_outputs[probe].append(self.signals[probe.sig])

        self.n_steps += 1

    def run_steps(self, N):
        for i in xrange(N):
            self.step()

    def signal_probe_output(self, probe):
        return self.probe_outputs[probe]

