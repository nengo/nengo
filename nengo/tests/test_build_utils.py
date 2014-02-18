import pytest

import nengo
import nengo.build_utils


def test_remove_passthrough():
    """Test scanning through a model and removing Nodes with output=None"""

    model = nengo.Model('test_remove_passthrough')
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
        nengo.Connection(b.output, output, filter=0.01)

    objs, conns = nengo.build_utils.remove_passthrough_nodes(
        model.objs, model.connections)

    sim = nengo.Simulator(model)
    sim.run(0.01)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
