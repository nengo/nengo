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


def test_labels():
    nengo.Model('test_basalganglia_labels')
    bg = nengo.networks.BasalGanglia(dimensions=2)
    assert bg.label == 'Basal Ganglia'
    assert bg.strD1.label == 'Basal Ganglia.Striatal D1 neurons'
    assert bg.strD2.label == 'Basal Ganglia.Striatal D2 neurons'
    assert bg.stn.label == 'Basal Ganglia.Subthalamic nucleus'
    assert bg.gpi.label == 'Basal Ganglia.Globus pallidus internus'
    assert bg.gpe.label == 'Basal Ganglia.Globus pallidus externus'
    assert bg.input.label == 'Basal Ganglia.input'
    assert bg.output.label == 'Basal Ganglia.output'
    bg_short = nengo.networks.BasalGanglia(dimensions=2, label='BG')
    assert bg_short.label == 'BG'
    assert bg_short.strD1.label == 'BG.Striatal D1 neurons'
    assert bg_short.strD2.label == 'BG.Striatal D2 neurons'
    assert bg_short.stn.label == 'BG.Subthalamic nucleus'
    assert bg_short.gpi.label == 'BG.Globus pallidus internus'
    assert bg_short.gpe.label == 'BG.Globus pallidus externus'
    assert bg_short.input.label == 'BG.input'
    assert bg_short.output.label == 'BG.output'


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
