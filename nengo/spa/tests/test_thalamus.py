import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaParseError

import numpy as np


@pytest.mark.slow
def test_thalamus(Simulator, plt, seed):
    model = spa.Module(seed=seed)

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

    with Simulator(model) as sim:
        sim.run(0.5)

    t = sim.trange()
    data = vocab.dot(sim.data[p].T)
    data2 = vocab2.dot(sim.data[p2].T)

    plt.subplot(2, 1, 1)
    plt.plot(t, data.T)
    plt.subplot(2, 1, 2)
    plt.plot(t, data2.T)

    # Action 1
    assert data[0, t == 0.1] > 0.8
    assert data[1, t == 0.1] < 0.2
    assert data2[0, t == 0.1] < 0.35
    assert data2[1, t == 0.1] > 0.4
    # Action 2
    assert data[0, t == 0.3] < 0.2
    assert data[1, t == 0.3] > 0.8
    assert data2[0, t == 0.3] > 0.5
    assert data2[1, t == 0.3] < 0.3
    # Action 3
    assert data[0, t == 0.5] > 0.8
    assert data[1, t == 0.5] < 0.2
    assert data2[0, t == 0.5] < 0.5
    assert data2[1, t == 0.5] > 0.4


def test_routing(Simulator, seed, plt):
    D = 3
    model = spa.Module(seed=seed)
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

    with Simulator(model) as sim:
        sim.run(0.6)

    data = sim.data[buff3_probe]

    plt.plot(sim.trange(), data)

    valueA = np.mean(data[150:200], axis=0)  # should be [0, 1, 0]
    valueB = np.mean(data[350:400], axis=0)  # should be [0, 0, 1]
    valueC = np.mean(data[550:600], axis=0)  # should be [1, 0, 0]

    assert valueA[0] < 0.2
    assert valueA[1] > 0.8
    assert valueA[2] < 0.2

    assert valueB[0] < 0.2
    assert valueB[1] < 0.2
    assert valueB[2] > 0.8

    assert valueC[0] > 0.8
    assert valueC[1] < 0.2
    assert valueC[2] < 0.2


def test_nondefault_routing(Simulator, seed, plt):
    D = 3
    model = spa.Module(seed=seed)
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
        model.cmp = spa.Compare(D)

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.state.input)
        nengo.Connection(node2, model.buff2.state.input)

        actions = spa.Actions('dot(ctrl, A) --> cmp.A=buff1, cmp.B=buff1',
                              'dot(ctrl, B) --> cmp.A=buff1, cmp.B=buff2',
                              'dot(ctrl, C) --> cmp.A=buff2, cmp.B=buff2',
                              )
        model.bg = spa.BasalGanglia(actions)
        model.thal = spa.Thalamus(model.bg)

        compare_probe = nengo.Probe(model.cmp.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    similarity = sim.data[compare_probe]

    valueA = np.mean(similarity[150:200], axis=0)  # should be [1]
    valueB = np.mean(similarity[350:400], axis=0)  # should be [0]
    valueC = np.mean(similarity[550:600], axis=0)  # should be [1]

    assert valueA > 0.6
    assert valueB < 0.3
    assert valueC > 0.6


def test_errors():
    # motor does not exist
    with pytest.raises(SpaParseError):
        with spa.Module() as model:
            model.vision = spa.Buffer(dimensions=16)
            actions = spa.Actions('0.5 --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # dot products not implemented
    with pytest.raises(NotImplementedError):
        with spa.Module() as model:
            model.scalar = spa.Buffer(dimensions=16, subdimensions=1)
            actions = spa.Actions('0.5 --> scalar=dot(scalar, FOO)')
            model.bg = spa.BasalGanglia(actions)
            model.thalamus = spa.Thalamus(model.bg)
