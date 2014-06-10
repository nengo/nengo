import numpy as np
import pytest

import nengo
from nengo.utils.functions import piecewise
from nengo.utils.testing import Plotter


def test_inputgatedmemory(Simulator):
    with nengo.Network(seed=123) as net:
        test_input = nengo.Node(piecewise({0.0: 0, 0.3: 0.5, 1.0: 0}))

        gate_input = nengo.Node(piecewise({0.0: 0, 0.8: 1}))

        mem = nengo.networks.InputGatedMemory(100, 1, input_gain=2.0)
        nengo.Connection(test_input, mem.input)

        nengo.Connection(gate_input, mem.gate)

        mem_p = nengo.Probe(mem.output, synapse=0.01)

    sim = Simulator(net)
    sim.run(1.2)

    data = sim.data[mem_p]
    trange = sim.trange()

    with Plotter(Simulator) as plt:
        plt.plot(trange, data)

        plt.savefig('test_workingmemory.test_basic.pdf')
        plt.close()

    assert abs(np.mean(data[trange < 0.3])) < 0.01
    assert abs(np.mean(data[(trange > 0.8) & (trange < 1.0)]) - 0.5) < 0.01
    assert abs(np.mean(data[trange > 1.0]) - 0.5) < 0.01

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
