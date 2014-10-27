import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import Connection, LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.node import Node
from nengo.probe import Probe
from nengo.utils.compat import iteritems


def conn_probe(probe, model, config):
    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeds[conn] = model.seeds[probe]

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    Builder.build(conn, model=model, config=config)


def synapse_probe(key, probe, model, config):
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
            probe, sig, probe.synapse, model=model, config=config)

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
def build_probe(probe, model, config):
    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in iteritems(probemap):
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise ValueError("Type '%s' is not probeable" % type(probe.obj))

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(probe, model, config)
    else:
        synapse_probe(key, probe, model, config)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
