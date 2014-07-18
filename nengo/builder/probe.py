import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.probe import Probe


def conn_probe(pre, probe, **conn_args):
    # TODO: make this connection in the network config context
    return Connection(pre, probe, **conn_args)


def synapse_probe(model, sig, probe):
    # We can use probe.conn_args here because we don't modify synapse
    synapse = probe.conn_args.get('synapse', None)

    if synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['synapse_in'] = sig
        model.build(probe, synapse, 'synapse')
        model.sig[probe]['in'] = model.sig[probe]['synapse_out']


def probe_ensemble(model, probe, conn_args):
    ens = probe.target
    if probe.attr == 'decoded_output':
        return conn_probe(ens, probe, **conn_args)
    elif probe.attr in ('neuron_output', 'spikes'):
        return conn_probe(ens.neurons, probe, transform=1.0, **conn_args)
    elif probe.attr == 'voltage':
        return synapse_probe(model, model.sig[ens]['voltage'], probe)
    elif probe.attr == 'input':
        return synapse_probe(model, model.sig[ens]['in'], probe)


def probe_node(model, probe, conn_args):
    if probe.attr == 'output':
        return conn_probe(probe.target, probe,  **conn_args)


def probe_connection(model, probe, conn_args):
    if probe.attr == 'signal':
        sig_out = model.sig[probe.target]['out']
        return synapse_probe(model, sig_out, probe)


@Builder.register(Probe)
def build_probe(model, probe):
    # Make a copy so as not to modify the probe
    conn_args = probe.conn_args.copy()
    # If we make a connection, we won't add it to a network
    conn_args['add_to_container'] = False

    if isinstance(probe.target, Ensemble):
        conn = probe_ensemble(model, probe, conn_args)
    elif isinstance(probe.target, Node):
        conn = probe_node(model, probe, conn_args)
    elif isinstance(probe.target, Connection):
        conn = probe_connection(model, probe, conn_args)

    # Most probes are implemented as connections
    if conn is not None:
        # Make a sink signal for the connection
        model.sig[probe]['in'] = Signal(np.zeros(conn.size_out),
                                        name=probe.label)
        model.add_op(Reset(model.sig[probe]['in']))
        # Set connection's seed to probe's (which isn't used elsewhere)
        model.seeds[conn] = model.seeds[probe]
        # Build the connection
        model.build(conn)

    # Let the model know
    model.probes.append(probe)

    # We put a list here so that the simulator can fill it
    # as it simulates the model
    model.params[probe] = []
