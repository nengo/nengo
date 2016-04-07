import nengo
from nengo import spa


def test_basic():
    with spa.SPA() as model:
        model.compare = spa.Compare(dimensions=16)

    inputA = model.get_module_input('compare.A')
    inputB = model.get_module_input('compare.B')
    output = model.get_module_output('compare')
    # all nodes should be acquired correctly
    assert inputA[0] is model.compare.inputA
    assert inputB[0] is model.compare.inputB
    assert output[0] is model.compare.output
    # all inputs should share the same vocab
    assert inputA[1] is inputB[1]
    assert inputA[1].dimensions == 16
    # output should have no vocab
    assert output[1] is None


def test_run(Simulator, seed):
    with spa.SPA(seed=seed) as model:
        model.compare = spa.Compare(dimensions=16)

        def inputA(t):
            if 0 <= t < 0.1:
                return 'A'
            else:
                return 'B'

        model.input = spa.Input(**{'compare.A': inputA, 'compare.B': 'A'})

    compare, vocab = model.get_module_output('compare')

    with model:
        p = nengo.Probe(compare, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.2)

    assert sim.data[p][100] > 0.8
    assert sim.data[p][199] < 0.2
