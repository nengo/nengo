import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.probe import Probe


def conn_probe(pre, probe, **conn_args):
    # TODO: make this connection in the network config context
    return Connection(pre, probe, **conn_args)


def synapse_probe(sig, probe, model, config):
    # We can use probe.conn_args here because we don't modify synapse
    synapse = probe.conn_args.get('synapse', None)

    if synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['in'] = filtered_signal(
            probe, sig, synapse, model=model, config=config)


def probe_ensemble(probe, conn_args, model, config):
    ens = probe.target
    if probe.attr == 'decoded_output':
        return conn_probe(ens, probe, **conn_args)
    elif probe.attr in ('neuron_output', 'spikes'):
        return conn_probe(ens.neurons, probe, transform=1.0, **conn_args)
    elif probe.attr == 'voltage':
        return synapse_probe(model.sig[ens]['voltage'], probe, model, config)
    elif probe.attr == 'input':
        return synapse_probe(model.sig[ens]['in'], probe, model, config)


def probe_node(probe, conn_args, model, config):
    if probe.attr == 'output':
        return conn_probe(probe.target, probe,  **conn_args)


def probe_connection(probe, conn_args, model, config):
    if probe.attr == 'signal':
        sig_out = model.sig[probe.target]['out']
        return synapse_probe(sig_out, probe, model, config)


def build_probe(probe, model, config):
    # Make a copy so as not to modify the probe
    conn_args = probe.conn_args.copy()
    # If we make a connection, we won't add it to a network
    conn_args['add_to_container'] = False

    if isinstance(probe.target, Ensemble):
        conn = probe_ensemble(probe, conn_args, model, config)
    elif isinstance(probe.target, Node):
        conn = probe_node(probe, conn_args, model, config)
    elif isinstance(probe.target, Connection):
        conn = probe_connection(probe, conn_args, model, config)

    # Most probes are implemented as connections
    if conn is not None:
        # Make a sink signal for the connection
        model.sig[probe]['in'] = Signal(np.zeros(conn.size_out),
                                        name=probe.label)
        model.add_op(Reset(model.sig[probe]['in']))
        # Set connection's seed to probe's (which isn't used elsewhere)
        model.seeds[conn] = model.seeds[probe]
        # Build the connection
        Builder.build(conn, model=model, config=config)

    # Let the model know
    model.probes.append(probe)

    # We put a list here so that the simulator can fill it
    # as it simulates the model
    model.params[probe] = []

Builder.register_builder(build_probe, Probe)
