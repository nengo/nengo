import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaModuleError

import numpy as np


@pytest.mark.slow
def test_thalamus(Simulator, plt, seed):
    model = spa.Module(seed=seed)

    with model:
        model.vision = spa.State(vocab=16, neurons_per_dimension=80)
        model.vision2 = spa.State(vocab=16, neurons_per_dimension=80)
        model.motor = spa.State(vocab=16, neurons_per_dimension=80)
        model.motor2 = spa.State(vocab=32, neurons_per_dimension=80)

        spa.Actions(
            'dot(vision, A) --> motor=A, motor2=translate(vision*vision2)',
            'dot(vision, B) --> motor=vision, motor2=translate(vision*A*~B)',
            'dot(vision, ~A) --> motor=~vision, '
            'motor2=translate(~vision*vision2)'
        ).build(model)

        def input_f(t):
            if t < 0.1:
                return 'A'
            elif t < 0.3:
                return 'B'
            elif t < 0.5:
                return '~A'
            else:
                return '0'
        model.input = spa.Input()
        model.input.vision = input_f
        model.input.vision2 = 'B*~A'

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
    model = spa.Module(seed=seed)
    model.config[spa.State].vocab = 3
    model.config[spa.State].subdimensions = 3
    with model:
        model.ctrl = spa.State(16, subdimensions=16, label='ctrl')

        def input_func(t):
            if t < 0.2:
                return 'A'
            elif t < 0.4:
                return 'B'
            else:
                return 'C'
        model.input = spa.Input()
        model.input.ctrl = input_func

        model.buff1 = spa.State(label='buff1')
        model.buff2 = spa.State(label='buff2')
        model.buff3 = spa.State(label='buff3')

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        spa.Actions(
            'dot(ctrl, A) --> buff3=buff1',
            'dot(ctrl, B) --> buff3=buff2',
            'dot(ctrl, C) --> buff3=buff1*buff2',
        ).build(model)

        buff3_probe = nengo.Probe(model.buff3.output, synapse=0.03)

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


def test_routing_recurrency_compilation(Simulator, seed, plt):
    model = spa.Module(seed=seed)
    model.config[spa.State].vocab = 2
    model.config[spa.State].subdimensions = 2
    with model:
        model.buff1 = spa.State(label='buff1')
        model.buff2 = spa.State(label='buff2')
        spa.Actions('0.5 --> buff2=buff1, buff1=buff2').build(model)

    with Simulator(model) as sim:
        assert sim


def test_nondefault_routing(Simulator, seed):
    model = spa.Module(seed=seed)
    model.config[spa.State].vocab = 3
    model.config[spa.State].subdimensions = 3
    with model:
        model.ctrl = spa.State(16, subdimensions=16, label='ctrl')

        def input_func(t):
            if t < 0.2:
                return 'A'
            elif t < 0.4:
                return 'B'
            else:
                return 'C'
        model.input = spa.Input()
        model.input.ctrl = input_func

        model.buff1 = spa.State(label='buff1')
        model.buff2 = spa.State(label='buff2')
        model.cmp = spa.Compare(3)

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        spa.Actions(
            'dot(ctrl, A) --> cmp.input_a=buff1, cmp.input_b=buff1',
            'dot(ctrl, B) --> cmp.input_a=buff1, cmp.input_b=buff2',
            'dot(ctrl, C) --> cmp.input_a=buff2, cmp.input_b=buff2',
        ).build(model)

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
    with pytest.raises(SpaModuleError):
        with spa.Module() as model:
            model.vision = spa.State(vocab=16)
            spa.Actions('0.5 --> motor=A').build(model)
