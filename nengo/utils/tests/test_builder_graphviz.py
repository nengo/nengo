import numpy as np

import nengo
from nengo.utils.builder import (
    generate_graphviz, remove_passthrough_nodes, objs_and_connections)


def test_create_dot():
    """Constructing a .dot file for a model"""

    model = nengo.Network()
    with model:
        D = 3
        input = nengo.Node([1] * D, label='input')
        a = nengo.networks.EnsembleArray(50, D, label='a')
        b = nengo.networks.EnsembleArray(50, D, label='b')
        output = nengo.Node(None, size_in=D, label='output')

        nengo.Connection(input, a.input, synapse=0.01)
        nengo.Connection(a.output, b.input, synapse=0.01)
        nengo.Connection(b.output, b.input, synapse=0.01, transform=0.9)
        nengo.Connection(a.output, a.input, synapse=0.01,
                         transform=np.ones((D, D)))
        nengo.Connection(b.output, output, synapse=0.01)

    objs, conns = objs_and_connections(model)
    dot = generate_graphviz(objs, conns)
    assert len(dot.splitlines()) == 31
    # not sure what else to check here

    dot = generate_graphviz(
        *remove_passthrough_nodes(objs, conns))
    assert len(dot.splitlines()) == 27
    # not sure what else to check here
