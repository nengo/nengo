import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Reset, Copy
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import Connection, LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.node import Node
from nengo.probe import Probe
from nengo.utils.compat import iteritems


def conn_probe(model, probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeds[conn] = model.seeds[probe]

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    model.build(conn)


def signal_probe(model, key, probe):
    # Signal probes directly probe a target signal

    try:
        sig = model.sig[probe.obj][key]
    except IndexError:
        raise ValueError("Attribute '%s' is not probeable on %s."
                         % (key, probe.obj))

    if probe.slice is not None:
        model.sig[probe]['slice_in'] = (
            Signal(np.zeros_like(sig.value[probe.slice])))
        model.add_op(Copy(model.sig[probe]['slice_in'], sig,
                          as_update=True, pre_slice=probe.slice))
        sig = model.sig[probe]['slice_in']

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
    LearningRule: {},  # make LR signals probeable, but no mapping required
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
        signal_probe(model, key, probe)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
