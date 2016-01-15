import numpy as np
import pytest

import nengo
from nengo import spa


def test_basic():
    with spa.SPA() as model:
        model.memory = spa.Memory(dimensions=16)

    input = model.get_module_input('memory')
    output = model.get_module_output('memory')
    assert input[0] is model.memory.state.input
    assert output[0] is model.memory.state.output
    assert input[1] is output[1]
    assert input[1].dimensions == 16


def test_neurons():
    with spa.SPA() as model:
        model.memory = spa.Memory(dimensions=16, neurons_per_dimension=2)

    assert len(model.memory.state.ensembles) == 1
    assert model.memory.state.ensembles[0].n_neurons == 16 * 2

    with spa.SPA() as model:
        model.memory = spa.Memory(dimensions=16, subdimensions=1,
                                  neurons_per_dimension=2)

    assert len(model.memory.state.ensembles) == 16
    assert model.memory.state.ensembles[0].n_neurons == 2


def test_exception():
    with pytest.raises(Exception):
        with spa.SPA() as model:
            vocab = spa.Vocabulary(16)
            model.buffer = spa.Memory(dimensions=12, vocab=vocab)

    with spa.SPA() as model:
        model.memory = spa.Memory(dimensions=12, subdimensions=3)


def test_run(Simulator, seed, plt):
    with spa.SPA(seed=seed) as model:
        model.memory = spa.Memory(dimensions=32)

        def input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.input = spa.Input(memory=input)

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


def test_run_decay(Simulator, plt, seed):
    with spa.SPA(seed=seed) as model:
        model.memory = spa.Memory(dimensions=32, tau=0.05)

        def input(t):
            if 0 <= t < 0.05:
                return 'A'
            else:
                return '0'

        model.input = spa.Input(memory=input)

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
