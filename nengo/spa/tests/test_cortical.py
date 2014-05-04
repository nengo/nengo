import numpy as np
import pytest

import nengo
from nengo import spa


def test_connect(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)
            self.buffer3 = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer1',
                                                     'buffer3=~buffer1'))
            self.input = spa.Input(buffer1='A')

    model = SPA(seed=122)

    output2, vocab = model.get_module_output('buffer2')
    output3, vocab = model.get_module_output('buffer3')

    with model:
        p2 = nengo.Probe(output2, 'output', synapse=0.03)
        p3 = nengo.Probe(output3, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p2], vocab.parse('A').v)
    assert match[199] > 0.95
    match = np.dot(sim.data[p3], vocab.parse('~A').v)
    assert match[199] > 0.95


def test_transform(Simulator):

    class SPA(spa.SPA):
        def __init__(self):
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
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions('buffer2=buffer'))

    with pytest.raises(NameError):
        SPA()  # buffer2 does not exist

    class SPA(spa.SPA):
        def __init__(self):
            self.buffer = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions(
                'dot(buffer,A) --> buffer=buffer'))

    with pytest.raises(NotImplementedError):
        SPA()  # conditional expressions not implemented

    class SPA(spa.SPA):
        def __init__(self):
            self.scalar = spa.Buffer(dimensions=1, subdimensions=1)
            self.cortical = spa.Cortical(spa.Actions(
                'scalar=dot(scalar, FOO)'))

    with pytest.raises(NotImplementedError):
        SPA()  # dot products not implemented

    class SPA(spa.SPA):
        def __init__(self):
            self.unitary = spa.Buffer(dimensions=16)
            self.cortical = spa.Cortical(spa.Actions(
                'unitary=unitary*unitary'))

    with pytest.raises(NotImplementedError):
        SPA()  # convolution not implemented


def test_direct(Simulator):

    class SPA(spa.SPA):
        def __init__(self):
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
    assert match1[199] > 0.45
    assert match2[199] > 0.45

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
