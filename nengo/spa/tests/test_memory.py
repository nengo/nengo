import numpy as np
import pytest

import nengo
from nengo import spa


def test_basic():
    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=16)

    model = Basic()

    input = model.get_module_input('memory')
    output = model.get_module_output('memory')
    assert input[0] is model.memory.state.input
    assert output[0] is model.memory.state.output
    assert input[1] is output[1]
    assert input[1].dimensions == 16


def test_neurons():
    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=16, neurons_per_dimension=2)

    model = Basic()
    assert len(model.memory.state.ensembles) == 1
    assert model.memory.state.ensembles[0].n_neurons == 16 * 2

    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=16, subdimensions=1,
                                     neurons_per_dimension=2)

    model = Basic()
    assert len(model.memory.state.ensembles) == 16
    assert model.memory.state.ensembles[0].n_neurons == 2


def test_exception():
    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=12)

    with pytest.raises(Exception):
        Basic()

    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=12, subdimensions=3)
    Basic()


def test_run(Simulator):
    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=32)

            def input(t):
                if 0 <= t < 0.05:
                    return 'A'
                else:
                    return '0'

            self.input = spa.Input(memory=input)
    model = Basic(seed=123)

    memory, vocab = model.get_module_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)

    data = np.dot(sim.data[p], vocab.vectors.T)
    assert data[100, 0] > 1.5
    assert data[300, 0] > 1.0
    assert data[499, 0] > 0.8


def test_run_decay(Simulator):
    class Basic(spa.SPA):
        def __init__(self):
            self.memory = spa.Memory(dimensions=32, tau=0.05)

            def input(t):
                if 0 <= t < 0.05:
                    return 'A'
                else:
                    return '0'

            self.input = spa.Input(memory=input)
    model = Basic(seed=123)

    memory, vocab = model.get_module_output('memory')

    with model:
        p = nengo.Probe(memory, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.3)

    data = np.dot(sim.data[p], vocab.vectors.T)
    assert data[50, 0] > 1.3
    assert data[299, 0] < 0.5


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
