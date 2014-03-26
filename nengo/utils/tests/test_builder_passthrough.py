import numpy as np
import pytest

import nengo
from nengo.utils.builder import remove_passthrough_nodes


def test_remove_passthrough():
    """Test scanning through a model and removing Nodes with output=None"""

    model = nengo.Model()
    with model:
        D = 3
        input = nengo.Node([1]*D, label='input')
        a = nengo.networks.EnsembleArray(50, D, label='a')
        b = nengo.networks.EnsembleArray(50, D, label='b')

        def printout(t, x):
            print(t, x)
        output = nengo.Node(printout, size_in=D, label='output')

        nengo.Connection(input, a.input, filter=0.01)
        nengo.Connection(a.output, b.input, filter=0.01)
        nengo.Connection(b.output, b.input, filter=0.01, transform=0.9)
        nengo.Connection(a.output, a.input, filter=0.01,
                         transform=np.ones((D, D)))
        nengo.Connection(b.output, output, filter=0.01)

    objs, conns = remove_passthrough_nodes(model.objs.values(),
                                           model.connections.values())

    assert len(objs) == 8
    assert len(conns) == 21


def test_remove_passthrough_bg():
    """Test scanning through a model and removing Nodes with output=None"""

    model = nengo.Model()
    with model:
        D = 3
        input = nengo.Node([1]*D, label='input')

        def printout(t, x):
            print(t, x)
        output = nengo.Node(printout, size_in=D, label='output')
        bg = nengo.networks.BasalGanglia(D, 20, label='BG')
        nengo.Connection(input, bg.input, filter=0.01)
        nengo.Connection(bg.output, output, filter=0.01)

    objs, conns = remove_passthrough_nodes(model.objs.values(),
                                           model.connections.values())

    assert len(objs) == 17
    assert len(conns) == 42


def test_passthrough_errors():
    """Test errors removing Nodes with output=None"""

    model = nengo.Model()
    with model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        node = nengo.Node(None, size_in=1)
        nengo.Connection(a, node, filter=0.01)
        nengo.Connection(node, b, filter=0.01)
    with pytest.raises(NotImplementedError):
        remove_passthrough_nodes(model.objs.values(),
                                 model.connections.values())

    model = nengo.Model()
    with model:
        node = nengo.Node(None, size_in=1)
        nengo.Connection(node, node, filter=0.01)
    with pytest.raises(Exception):
        remove_passthrough_nodes(model.objs.values(),
                                 model.connections.values())


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
