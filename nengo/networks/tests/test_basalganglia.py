import numpy as np
import pytest

import nengo
from nengo.utils.testing import Plotter


def test_basic(Simulator):
    bg = nengo.networks.BasalGanglia(
        dimensions=5, label="BG", solver=nengo.Default)
    with bg:
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    sim = Simulator(bg, seed=123)
    sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    with Plotter(Simulator) as plt:
        plt.plot(t, sim.data[p])
        plt.ylabel("Output")
        plt.savefig('test_basalganglia.test_basic.pdf')
        plt.close()

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
