import numpy as np
import pytest

import nengo
from nengo import spa


def test_basic():
    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=16)

    model = Basic()

    input = model.get_module_input('buffer')
    output = model.get_module_output('buffer')
    assert input[0] is model.buffer.state.input
    assert output[0] is model.buffer.state.output
    assert input[1] is output[1]
    assert input[1].dimensions == 16


def test_neurons():
    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=16, neurons_per_dimension=2)

    model = Basic()
    assert len(model.buffer.state.ensembles) == 1
    assert model.buffer.state.ensembles[0].n_neurons == 16 * 2

    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=16, subdimensions=1,
                                     neurons_per_dimension=2)

    model = Basic()
    assert len(model.buffer.state.ensembles) == 16
    assert model.buffer.state.ensembles[0].n_neurons == 2


def test_exception():
    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=12)

    with pytest.raises(Exception):
        Basic()

    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=12, subdimensions=3)
    Basic()


def test_run(Simulator):
    class Basic(spa.SPA):
        def __init__(self):
            super(Basic, self).__init__()
            self.buffer = spa.Buffer(dimensions=32)

            def input(t):
                if 0 <= t < 0.2:
                    return 'A'
                elif 0.2 <= t < 0.4:
                    return 'B'
                else:
                    return '0'
            self.input = spa.Input(buffer=input)
    model = Basic(seed=123)

    buffer, vocab = model.get_module_output('buffer')

    with model:
        p = nengo.Probe(buffer, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)

    data = np.dot(sim.data[p], vocab.vectors.T)
    assert data[200, 0] > 0.9
    assert data[200, 1] < 0.2
    assert data[400, 0] < 0.2
    assert data[400, 1] > 0.9
    assert data[499, 0] < 0.2
    assert data[499, 1] < 0.2


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
