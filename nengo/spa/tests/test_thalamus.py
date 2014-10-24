import pytest

import nengo
from nengo import spa

import numpy as np


@pytest.mark.optional  # Too slow
def test_thalamus(Simulator):
    model = spa.SPA(seed=30)

    with model:
        model.vision = spa.Buffer(dimensions=16, neurons_per_dimension=80)
        model.vision2 = spa.Buffer(dimensions=16, neurons_per_dimension=80)
        model.motor = spa.Buffer(dimensions=16, neurons_per_dimension=80)
        model.motor2 = spa.Buffer(dimensions=32, neurons_per_dimension=80)

        actions = spa.Actions(
            'dot(vision, A) --> motor=A, motor2=vision*vision2',
            'dot(vision, B) --> motor=vision, motor2=vision*A*~B',
            'dot(vision, ~A) --> motor=~vision, motor2=~vision*vision2'
        )
        model.bg = spa.BasalGanglia(actions)
        model.thalamus = spa.Thalamus(model.bg)

        def input_f(t):
            if t < 0.1:
                return 'A'
            elif t < 0.3:
                return 'B'
            elif t < 0.5:
                return '~A'
            else:
                return '0'
        model.input = spa.Input(vision=input_f, vision2='B*~A')

        input, vocab = model.get_module_input('motor')
        input2, vocab2 = model.get_module_input('motor2')
        p = nengo.Probe(input, 'output', synapse=0.03)
        p2 = nengo.Probe(input2, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)

    data = vocab.dot(sim.data[p].T)
    data2 = vocab2.dot(sim.data[p2].T)

    assert 0.8 < data[0, 100] < 1.1  # Action 1
    assert -0.2 < data2[0, 100] < 0.35
    assert -0.25 < data[1, 100] < 0.2
    assert 0.4 < data2[1, 100] < 0.6
    assert -0.2 < data[0, 299] < 0.2  # Action 2
    assert 0.6 < data2[0, 299] < 0.95
    assert 0.8 < data[1, 299] < 1.1
    assert -0.2 < data2[1, 299] < 0.2
    assert 0.8 < data[0, 499] < 1.0  # Action 3
    assert 0.0 < data2[0, 499] < 0.5
    assert -0.2 < data[1, 499] < 0.2
    assert 0.4 < data2[1, 499] < 0.7


def test_routing(Simulator):
    D = 2
    model = spa.SPA(seed=123)
    with model:
        model.ctrl = spa.Buffer(D, label='ctrl')

        def input_func(t):
            if t < 0.25:
                return 'A'
            else:
                return 'B'
        model.input = spa.Input(ctrl=input_func)

        model.buff1 = spa.Buffer(D, label='buff1')
        model.buff2 = spa.Buffer(D, label='buff2')
        model.buff3 = spa.Buffer(D, label='buff3')

        node1 = nengo.Node([1, 0])
        node2 = nengo.Node([0, 1])

        nengo.Connection(node1, model.buff1.state.input)
        nengo.Connection(node2, model.buff2.state.input)

        actions = spa.Actions('dot(ctrl, A) --> buff3=buff1',
                              'dot(ctrl, B) --> buff3=buff2')
        model.bg = spa.BasalGanglia(actions)
        model.thal = spa.Thalamus(model.bg)

        buff1_probe = nengo.Probe(model.buff1.state.output, synapse=0.03)
        buff2_probe = nengo.Probe(model.buff2.state.output, synapse=0.03)
        buff3_probe = nengo.Probe(model.buff3.state.output, synapse=0.03)
        thal_probe = nengo.Probe(model.thal.output, synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)

    data1 = sim.data[buff1_probe]
    data2 = sim.data[buff2_probe]
    data3 = sim.data[buff3_probe]
    data_thal = sim.data[thal_probe]
    trange = sim.trange()

    assert abs(np.mean(data1[trange < 0.25] - data3[trange < 0.25])) < 0.15
    assert abs(np.mean(data2[trange > 0.25] - data3[trange > 0.25])) < 0.15
    assert data_thal[249][1] < 0.01
    assert data_thal[249][0] > 0.8
    assert data_thal[499][1] > 0.8
    assert data_thal[499][0] < 0.01


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            self.vision = spa.Buffer(dimensions=16)
            actions = spa.Actions('0.5 --> motor=A')
            self.bg = spa.BasalGanglia(actions)

    with pytest.raises(NameError):
        SPA()  # motor does not exist

    class SPA(spa.SPA):
        def __init__(self):
            self.scalar = spa.Buffer(dimensions=16, subdimensions=1)
            actions = spa.Actions('0.5 --> scalar=dot(scalar, FOO)')
            self.bg = spa.BasalGanglia(actions)
            self.thalamus = spa.Thalamus(self.bg)

    with pytest.raises(NotImplementedError):
        SPA()  # dot products not implemented


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
