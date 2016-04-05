import numpy as np

import nengo
from nengo import spa
from nengo.utils.numpy import rmse


def test_basic():
    with spa.SPA() as model:
        model.bind = spa.Bind(dimensions=16)

    inputA = model.get_module_input('bind.A')
    inputB = model.get_module_input('bind.B')
    output = model.get_module_output('bind')
    # all nodes should be acquired correctly
    assert inputA[0] is model.bind.A
    assert inputB[0] is model.bind.B
    assert output[0] is model.bind.output
    # all inputs and outputs should share the same vocab
    assert inputA[1] is inputB[1]
    assert inputA[1].dimensions == 16
    assert output[1].dimensions == 16


def test_run(Simulator, seed):
    rng = np.random.RandomState(seed)
    vocab = spa.Vocabulary(16, rng=rng)

    with spa.SPA(seed=seed, vocabs=[vocab]) as model:
        model.bind = spa.Bind(dimensions=16)

        def inputA(t):
            if 0 <= t < 0.1:
                return 'A'
            else:
                return 'B'

        model.input = spa.Input(**{'bind.A': inputA, 'bind.B': 'A'})

    bind, vocab = model.get_module_output('bind')

    with model:
        p = nengo.Probe(bind, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    error = rmse(vocab.parse("B*A").v, sim.data[p][-1])
    assert error < 0.1

    error = rmse(vocab.parse("A*A").v, sim.data[p][100])
    assert error < 0.1
