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
    D = 3
    model = spa.SPA(seed=123)
    with model:
        model.ctrl = spa.Buffer(16, label='ctrl')

        def input_func(t):
            if t < 0.2:
                return 'A'
            elif t < 0.4:
                return 'B'
            else:
                return 'C'
        model.input = spa.Input(ctrl=input_func)

        model.buff1 = spa.Buffer(D, label='buff1')
        model.buff2 = spa.Buffer(D, label='buff2')
        model.buff3 = spa.Buffer(D, label='buff3')

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.state.input)
        nengo.Connection(node2, model.buff2.state.input)

        actions = spa.Actions('dot(ctrl, A) --> buff3=buff1',
                              'dot(ctrl, B) --> buff3=buff2',
                              'dot(ctrl, C) --> buff3=buff1*buff2',
                              )
        model.bg = spa.BasalGanglia(actions)
        model.thal = spa.Thalamus(model.bg)

        buff3_probe = nengo.Probe(model.buff3.state.output, synapse=0.03)

    sim = Simulator(model)
    sim.run(0.6)

    data = sim.data[buff3_probe]

    valueA = np.mean(data[100:200], axis=0)  # should be [0, 1, 0]
    valueB = np.mean(data[300:400], axis=0)  # should be [0, 0, 1]
    valueC = np.mean(data[500:600], axis=0)  # should be [1, 0, 0]

    assert valueA[0] < 0.2
    assert valueA[1] > 0.8
    assert valueA[2] < 0.2

    assert valueB[0] < 0.2
    assert valueB[1] < 0.2
    assert valueB[2] > 0.8

    assert valueC[0] > 0.8
    assert valueC[1] < 0.2
    assert valueC[2] < 0.2


def test_errors():
    # motor does not exist
    with pytest.raises(NameError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            actions = spa.Actions('0.5 --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # dot products not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.scalar = spa.Buffer(dimensions=16, subdimensions=1)
            actions = spa.Actions('0.5 --> scalar=dot(scalar, FOO)')
            model.bg = spa.BasalGanglia(actions)
            model.thalamus = spa.Thalamus(model.bg)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
