import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.networks.unitvectorarray import HeuristicRadius
from nengo.utils.optimization import RadiusForUnitVector


@pytest.fixture(params=[HeuristicRadius, RadiusForUnitVector])
def Module(request, seed):
    def create_module():
        module = spa.Module(seed=seed)
        module.config[spa.State].radius_method = request.param
        return module
    return create_module


def test_basic(Module):
    with Module() as model:
        model.state = spa.State(dimensions=16)

    input = model.get_module_input('state')
    output = model.get_module_output('state')
    assert input[0] is model.state.input
    assert output[0] is model.state.output
    assert input[1] is output[1]
    assert input[1].dimensions == 16


def test_neurons(Module):
    with Module() as model:
        model.state = spa.State(dimensions=16, neurons_per_dimension=2)

    assert len(model.state.state_ensembles.ensembles) == 1
    assert model.state.state_ensembles.ensembles[0].n_neurons == 16 * 2

    with Module() as model:
        model.state = spa.State(dimensions=16, subdimensions=1,
                                neurons_per_dimension=2)

    assert len(model.state.state_ensembles.ensembles) == 16
    assert model.state.state_ensembles.ensembles[0].n_neurons == 2


def test_dimension_exception(Module):
    with pytest.raises(Exception):
        with Module() as model:
            vocab = spa.Vocabulary(16)
            model.state = spa.State(dimensions=12, vocab=vocab)

    with Module() as model:
        model.state = spa.State(dimensions=12, subdimensions=3)


def test_no_feedback_run(Module, Simulator):
    with Module() as model:
        model.state = spa.State(dimensions=32, feedback=0.0)

        def state_input(t):
            if 0 <= t < 0.2:
                return 'A'
            elif 0.2 <= t < 0.4:
                return 'B'
            else:
                return '0'
        model.state_input = spa.Input(state=state_input)

    state, vocab = model.get_module_output('state')

    with model:
        p = nengo.Probe(state, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    data = np.dot(sim.data[p], vocab.vectors.T)
    assert data[200, 0] > 0.9
    assert data[200, 1] < 0.2
    assert data[400, 0] < 0.2
    assert data[400, 1] > 0.9
    assert data[499, 0] < 0.2
    assert data[499, 1] < 0.2


def test_memory_run(Module, Simulator, plt):
    with Module() as model:
        model.memory = spa.State(dimensions=32, feedback=1.0,
                                 feedback_synapse=0.01)

        def state_input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.state_input = spa.Input(memory=state_input)

    memory, vocab = model.get_module_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()

    similarity = np.dot(sim.data[p], vocab.vectors.T)
    plt.plot(t, similarity)
    plt.ylabel("Similarity to 'A'")
    plt.xlabel("Time (s)")

    # value should peak above 1.0, then decay down to near 1.0
    assert np.mean(similarity[(t > 0.05) & (t < 0.1)]) > 1.2
    assert np.mean(similarity[(t > 0.2) & (t < 0.3)]) > 0.8
    assert np.mean(similarity[t > 0.49]) > 0.7


def test_memory_run_decay(Module, Simulator, plt):
    with Module() as model:
        model.memory = spa.State(dimensions=32, feedback=(1.0 - 0.01/0.05),
                                 feedback_synapse=0.01)

        def state_input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.state_input = spa.Input(memory=state_input)

    memory, vocab = model.get_module_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)
    data = np.dot(sim.data[p], vocab.vectors.T)

    t = sim.trange()
    plt.plot(t, data)

    assert data[t == 0.05, 0] > 1.0
    assert data[t == 0.299, 0] < 0.4
