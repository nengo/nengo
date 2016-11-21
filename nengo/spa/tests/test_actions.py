import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaTypeError


def test_new_action_syntax(Simulator, seed, plt, rng):
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

        model.state = spa.State(label='state')
        model.buff1 = spa.State(label='buff1')
        model.buff2 = spa.State(label='buff2')
        model.buff3 = spa.State(label='buff3')

        node1 = nengo.Node([0, 1, 0])
        node2 = nengo.Node([0, 0, 1])

        nengo.Connection(node1, model.buff1.input)
        nengo.Connection(node2, model.buff2.input)

        actions = spa.Actions(
            'state = buff1',
            'dot(ctrl, A) --> buff3=buff1',
            'dot(ctrl, B) --> buff3=buff2',
            'dot(ctrl, C) --> buff3=buff1*buff2')
        actions.build()

        state_probe = nengo.Probe(model.state.output, synapse=0.03)
        buff3_probe = nengo.Probe(model.buff3.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    data = sim.data[buff3_probe]

    plt.plot(sim.trange(), data)

    state_val = np.mean(sim.data[state_probe][200:], axis=0)
    assert state_val[0] < 0.2
    assert state_val[1] > 0.8
    assert state_val[2] < 0.2

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


def test_dot_product(Simulator, seed, plt):
    d = 16

    with spa.Module(seed=seed) as model:
        model.state_a = spa.State(d)
        model.state_b = spa.State(d)
        model.result = spa.Scalar()

        model.stimulus = spa.Input()
        model.stimulus.state_a = 'A'
        model.stimulus.state_b = lambda t: 'A' if t <= 0.3 else 'B'

        actions = spa.Actions('result = dot(state_a, state_b)')
        actions.build()

        p = nengo.Probe(model.result.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.6)

    plt.plot(sim.trange(), sim.data[p])

    t = sim.trange()
    assert np.mean(sim.data[p][(t > 0.1) & (t <= 0.3)]) > 0.8
    assert np.mean(sim.data[p][t > 0.4]) < 0.1


class TestExceptions():
    @pytest.fixture
    def model(self):
        with spa.Module() as model:
            model.state_a = spa.State(16)
            model.state_b = spa.State(32)
            model.state_c = spa.State(32)
            model.scalar = spa.Scalar()
        return model

    def test_invalid_types_in_binary_operation(self, model):
        with model:
            actions = spa.Actions('state_c = state_a + state_b')
            with pytest.raises(SpaTypeError):
                actions.build()

    def test_approx_inverse_of_scalar(self, model):
        with model:
            actions = spa.Actions('state_c = ~scalar')
            with pytest.raises(SpaTypeError):
                actions.build()

    def test_dot_product_incompatiple_vocabs(self, model):
        with model:
            actions = spa.Actions('scalar = dot(state_a, state_b)')
            with pytest.raises(SpaTypeError):
                actions.build()

    def test_dot_product_first_argument_invalid(self, model):
        with model:
            actions = spa.Actions('scalar = dot(scalar, state_b)')
            with pytest.raises(SpaTypeError):
                actions.build()

    def test_dot_product_second_argument_invalid(self, model):
        with model:
            actions = spa.Actions('scalar = dot(state_a, scalar)')
            with pytest.raises(SpaTypeError):
                actions.build()

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_type_inference_not_possible(self, model, method):
        with model:
            actions = spa.Actions('scalar = dot({}(state_a), A)'.format(
                method))
            with pytest.raises(SpaTypeError):
                actions.build()

    @pytest.mark.parametrize('method', ['reinterpret', 'translate'])
    def test_cast_scalar(self, model, method):
        with model:
            actions = spa.Actions('state_a = {}(scalar)'.format(
                method))
            with pytest.raises(SpaTypeError):
                actions.build()

    def test_reinterpret_non_matching_dimensions(self, model):
        with model:
            actions = spa.Actions('state_a = reinterpret(state_b)')
            with pytest.raises(SpaTypeError):
                actions.build()
