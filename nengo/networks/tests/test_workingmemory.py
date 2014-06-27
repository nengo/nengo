import numpy as np
import pytest

import nengo
from nengo.utils.functions import piecewise
from nengo.utils.testing import Plotter


def test_inputgatedmemory(Simulator):
    with nengo.Network(seed=123) as net:
        test_input = nengo.Node(piecewise({0.0: 0, 0.3: 0.5, 1.0: 0}))

        gate_input = nengo.Node(piecewise({0.0: 0, 0.8: 1}))

        reset_input = nengo.Node(0)

        mem = nengo.networks.InputGatedMemory(100, 1, difference_gain=5.0)
        nengo.Connection(test_input, mem.input)

        nengo.Connection(gate_input, mem.gate)

        nengo.Connection(reset_input, mem.reset_node)

        mem_p = nengo.Probe(mem.output, synapse=0.01)

    sim = Simulator(net)
    sim.run(1.2)

    data = sim.data[mem_p]
    trange = sim.trange()

    with Plotter(Simulator) as plt:
        plt.plot(trange, data)

        plt.savefig('test_workingmemory.test_inputgatedmemory.pdf')
        plt.close()

    assert abs(np.mean(data[trange < 0.3])) < 0.01
    assert abs(np.mean(data[(trange > 0.8) & (trange < 1.0)]) - 0.5) < 0.02
    assert abs(np.mean(data[trange > 1.0]) - 0.5) < 0.02


def test_feedbackgatedmemory(Simulator):
    with nengo.Network(seed=123) as net:
        test_input = nengo.Node(piecewise({0.0: 0, 0.3: 0.5, 1.0: 0}))

        gate_input = nengo.Node(piecewise({0.0: 0, 0.8: 1}))

        reset_input = nengo.Node(0)

        mem = nengo.networks.FeedbackGatedMemory(100, 1)
        nengo.Connection(test_input, mem.input)

        nengo.Connection(gate_input, mem.gate)

        nengo.Connection(reset_input, mem.reset)

        mem_p = nengo.Probe(mem.output, synapse=0.01)

    sim = Simulator(net)
    sim.run(1.2)

    data = sim.data[mem_p]
    trange = sim.trange()

    with Plotter(Simulator) as plt:
        plt.plot(trange, data)

        plt.savefig('test_workingmemory.test_feedbackgatedmemory.pdf')
        plt.close()

    assert abs(np.mean(data[trange < 0.3])) < 0.01
    assert abs(np.mean(data[(trange > 0.5) & (trange < 0.7)]) - 0.50) < 0.02
    assert abs(np.mean(data[trange > 0.9]) - 0.50) < 0.025
    # Note: This type of memory is not the most stable of memory. Output tends
    # to drift quite a lot w.r.t. the input gated memory.


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
