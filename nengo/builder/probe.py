import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Operator, Reset
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import Connection, LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.node import Node
from nengo.probe import Probe, ProbeBuffer
from nengo.utils.compat import iteritems


class SimProbeOutput(Operator):
    def __init__(self, signal, probe_dt=None):
        self.signal = signal
        self.probe_dt = probe_dt

        self.sets = []
        self.incs = []
        self.reads = [signal]
        self.updates = []


class SimProbeBuffer(SimProbeOutput):

    def __init__(self, signal, probe_dt=None):
        super(SimProbeBuffer, self).__init__(signal, probe_dt=probe_dt)
        self.buffer = []

    def make_step(self, signals, dt, rng):
        # clear buffer, in case this has been run before
        del self.buffer[:]

        period = 1 if self.probe_dt is None else self.probe_dt / dt
        sim_step = signals['__step__']
        signal = signals[self.signal]
        buf = self.buffer

        def step():
            if sim_step % period < 1:
                buf.append(signal.copy())

        return step


@Builder.register(ProbeBuffer)
def build_probe_buffer(model, probe_buffer, probe):
    op = SimProbeBuffer(model.sig[probe]['in'], probe_dt=probe.sample_every)
    model.add_op(op, probe=True)

    # Add a reference so that Simulator can get this data for the user
    model.params[probe] = op.buffer


def conn_probe(model, probe):
    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeds[conn] = model.seeds[probe]

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    model.build(conn)


def synapse_probe(model, key, probe):
    try:
        sig = model.sig[probe.obj][key]
    except IndexError:
        raise ValueError("Attribute '%s' is not probable on %s."
                         % (key, probe.obj))

    if isinstance(probe.slice, slice):
        sig = sig[probe.slice]
    else:
        raise NotImplementedError("Indexing slices not implemented")

    if probe.synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['in'] = filtered_signal(
            model, probe, sig, probe.synapse)

probemap = {
    Ensemble: {'decoded_output': None,
               'input': 'in'},
    Neurons: {'output': None,
              'spikes': None,
              'rates': None,
              'input': 'in'},
    Node: {'output': None},
    Connection: {'output': 'out',
                 'input': 'in'},
    LearningRule: {},  # make LR signals probable, but no mapping required
}


@Builder.register(Probe)
def build_probe(model, probe):
    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in iteritems(probemap):
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise ValueError("Type '%s' is not probeable" % type(probe.obj))

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(model, probe)
    else:
        synapse_probe(model, key, probe)

    model.build(probe.output, probe)
