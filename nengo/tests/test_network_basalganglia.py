import numpy as np
import pytest

import nengo


def test_basic(Simulator):
    model = nengo.Model('test_basalganglia_basic')

    bg = nengo.networks.BasalGanglia(dimensions=5, label='BG')
    input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label='input')
    nengo.Connection(input, bg.input)
    p = nengo.Probe(bg.output, 'output')

    sim = Simulator(model)
    sim.run(0.2)

    output = np.mean(sim.data(p)[50:], axis=0)

    assert output[0] > -0.15
    assert np.all(output[1:] < -0.8)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
