import numpy as np
import pytest

import nengo
from nengo.utils.builder import generate_graphviz, remove_passthrough_nodes


def test_create_dot():
    """Constructing a .dot file for a model"""

    model = nengo.Model()
    with model:
        D = 3
        input = nengo.Node([1]*D, label='input')
        a = nengo.networks.EnsembleArray(50, D, label='a')
        b = nengo.networks.EnsembleArray(50, D, label='b')
        output = nengo.Node(None, size_in=D, label='output')

        nengo.Connection(input, a.input, filter=0.01)
        nengo.Connection(a.output, b.input, filter=0.01)
        nengo.Connection(b.output, b.input, filter=0.01, transform=0.9)
        nengo.Connection(a.output, a.input, filter=0.01,
                         transform=np.ones((D, D)))
        nengo.Connection(b.output, output, filter=0.01)

    dot = generate_graphviz(model.objs, model.connections)
    assert len(dot.splitlines()) == 31
    # not sure what else to check here

    objs, conns = remove_passthrough_nodes(model.objs, model.connections)

    dot = generate_graphviz(objs, conns)
    assert len(dot.splitlines()) == 27
    # not sure what else to check here

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
