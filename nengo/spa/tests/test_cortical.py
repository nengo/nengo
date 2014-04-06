import numpy as np
import pytest

import nengo
from nengo import spa


def test_connect(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__(rng=np.random.RandomState(1))
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1'))
            self.input = spa.Input(buffer1='A')

    model = SPA(seed=124)

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A').v)
    assert match[199] > 0.94


def test_transform(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__(rng=np.random.RandomState(3))
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1*B'))
            self.input = spa.Input(buffer1='A')

    model = SPA(seed=123)

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A*B').v)
    assert match[199] > 0.7


def test_translate(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__(rng=np.random.RandomState(1))
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=32)
            self.input = spa.Input(buffer1='A')
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1'))

    model = SPA(seed=123)

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A').v)
    assert match[199] > 0.8


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__()
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions(
                'dot(buffer,A) --> buffer=buffer'))

    with pytest.raises(NotImplementedError):
        SPA()

    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__()
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer'))

    with pytest.raises(NameError):
        SPA()


def test_direct(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__(rng=np.random.RandomState(2))
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=32)
            self.cortical = spa.Cortical(spa.Actions(
                'buffer1=A', 'buffer2=B',
                'buffer1=C, buffer2=C'))

    model = SPA(seed=123)

    output1, vocab1 = model.get_module_output('buffer1')
    output2, vocab2 = model.get_module_output('buffer2')

    with model:
        p1 = nengo.Probe(output1, 'output', synapse=0.03)
        p2 = nengo.Probe(output2, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match1 = np.dot(sim.data[p1], vocab1.parse('A+C').v)
    match2 = np.dot(sim.data[p2], vocab2.parse('A+C').v)
    assert match1[199] > 0.5
    assert match2[199] > 0.5

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
