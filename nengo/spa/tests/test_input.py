import numpy as np
import pytest

import nengo
from nengo import spa


def test_fixed():
    with spa.Module() as model:
        model.buffer1 = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=8, subdimensions=2)
        model.input = spa.Input(buffer1='A', buffer2='B')

    input1, vocab1 = model.get_module_input('buffer1')
    input2, vocab2 = model.get_module_input('buffer2')

    assert np.allclose(model.input.input_nodes['buffer1'].output,
                       vocab1.parse('A').v)
    assert np.allclose(model.input.input_nodes['buffer2'].output,
                       vocab2.parse('B').v)


def test_deprecated_underscore_notation(Simulator, plt):
    d = 16
    with pytest.warns(DeprecationWarning):
        with spa.Module() as model:
            model.cmp = spa.Compare(dimensions=d)
            model.input = spa.Input(cmp_A='A', cmp_B='A')
            p = nengo.Probe(model.cmp.output, synapse=0.03)

        with Simulator(model) as sim:
            sim.run(0.5)

    plt.plot(sim.trange(), sim.data[p])
    plt.xlabel("t [s]")
    plt.ylabel("Similarity")

    t = sim.trange() > 0.2
    assert np.mean(sim.data[p][t]) > 0.8


def test_time_varying():
    with spa.Module() as model:
        model.buffer = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=16)

        def input(t):
            if t < 0.1:
                return 'A'
            elif t < 0.2:
                return 'B'
            else:
                return '0'

        model.input = spa.Input(buffer=input, buffer2='B')

    input, vocab = model.get_module_input('buffer')

    assert np.allclose(model.input.input_nodes['buffer'].output(t=0),
                       vocab.parse('A').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.1),
                       vocab.parse('B').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.2),
                       np.zeros(16))


def test_predefined_vocabs():
    D = 64

    with spa.Module() as model:
        model.vocab1 = spa.Vocabulary(D)
        model.vocab1.parse('A+B+C')

        model.vocab2 = spa.Vocabulary(D)
        model.vocab1.parse('A+B+C')

        model.buffer1 = spa.Buffer(dimensions=D, vocab=model.vocab1)
        model.buffer2 = spa.Buffer(dimensions=D, vocab=model.vocab2)

        def input(t):
            if t < 0.1:
                return 'A'
            elif t < 0.2:
                return 'B'
            else:
                return 'C'

        model.input = spa.Input(buffer1=input, buffer2=input)

    a1 = model.input.input_nodes['buffer1'].output(t=0.0)
    b1 = model.input.input_nodes['buffer1'].output(t=0.1)
    c1 = model.input.input_nodes['buffer1'].output(t=0.2)

    a2 = model.input.input_nodes['buffer2'].output(t=0.0)
    b2 = model.input.input_nodes['buffer2'].output(t=0.1)
    c2 = model.input.input_nodes['buffer2'].output(t=0.2)

    assert np.allclose(a1, model.vocab1.parse('A').v)
    assert np.allclose(b1, model.vocab1.parse('B').v)
    assert np.allclose(c1, model.vocab1.parse('C').v)

    assert np.allclose(a2, model.vocab2.parse('A').v)
    assert np.allclose(b2, model.vocab2.parse('B').v)
    assert np.allclose(c2, model.vocab2.parse('C').v)

    assert np.dot(a1, a2) < 0.95
    assert np.dot(b1, b2) < 0.95
    assert np.dot(c1, c2) < 0.95
