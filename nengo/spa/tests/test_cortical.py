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
    assert match[199] > 0.9
    match = np.dot(sim.data[p3], vocab.parse('~A').v)
    assert match[199] > 0.9


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
    match2 = np.dot(sim.data[p2], vocab2.parse('B+C').v)
    # both values should be near 1.0 since buffer1 is driven to both A and C
    # and buffer2 is driven to both B and C.
    assert match1[199] > 0.75
    assert match2[199] > 0.75


def test_convolution(Simulator, plt, seed):
    model = spa.SPA(seed=seed)
    D = 5
    with model:
        model.inA = spa.Buffer(dimensions=D)
        model.inB = spa.Buffer(dimensions=D)
        model.outAB = spa.Buffer(dimensions=D)
        model.outABinv = spa.Buffer(dimensions=D)
        model.outAinvB = spa.Buffer(dimensions=D)
        model.outAinvBinv = spa.Buffer(dimensions=D)

        model.cortical = spa.Cortical(spa.Actions(
            'outAB = inA * inB',
            'outABinv = inA * ~inB',
            'outAinvB = ~inA * inB',
            'outAinvBinv = ~inA * ~inB',
            ))
        nengo.Connection(nengo.Node([0, 1, 0, 0, 0]), model.inA.state.input)
        nengo.Connection(nengo.Node([0, 0, 1, 0, 0]), model.inB.state.input)

        pAB = nengo.Probe(model.outAB.state.output, synapse=0.03)
        pABinv = nengo.Probe(model.outABinv.state.output, synapse=0.03)
        pAinvB = nengo.Probe(model.outAinvB.state.output, synapse=0.03)
        pAinvBinv = nengo.Probe(model.outAinvBinv.state.output, synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    t = sim.trange()
    plt.subplot(4, 1, 1)
    plt.ylabel('A*B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAB])
    plt.subplot(4, 1, 2)
    plt.ylabel('A*~B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pABinv])
    plt.subplot(4, 1, 3)
    plt.ylabel('~A*B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAinvB])
    plt.subplot(4, 1, 4)
    plt.ylabel('~A*~B')
    plt.axhline(0.85, c='k')
    plt.plot(t, sim.data[pAinvBinv])

    # Check results.  Since A is [0,1,0,0,0] and B is [0,0,1,0,0], this means:
    #    ~A = [0,0,0,0,1]
    #    ~B = [0,0,0,1,0]
    #   A*B = [0,0,0,1,0]
    #  A*~B = [0,0,0,0,1]
    #  ~A*B = [0,1,0,0,0]
    # ~A*~B = [0,0,1,0,0]
    # (Remember that X*[1,0,0,0,0]=X (identity transform) and X*[0,1,0,0,0]
    #  is X rotated to the right once)

    # Ideal answer: A*B = [0,0,0,1,0]
    assert np.allclose(np.mean(sim.data[pAB][-10:], axis=0),
                       np.array([0, 0, 0, 1, 0]), atol=0.15)

    # Ideal answer: A*~B = [0,0,0,0,1]
    assert np.allclose(np.mean(sim.data[pABinv][-10:], axis=0),
                       np.array([0, 0, 0, 0, 1]), atol=0.15)

    # Ideal answer: ~A*B = [0,1,0,0,0]
    assert np.allclose(np.mean(sim.data[pAinvB][-10:], axis=0),
                       np.array([0, 1, 0, 0, 0]), atol=0.15)

    # Ideal answer: ~A*~B = [0,0,1,0,0]
    assert np.allclose(np.mean(sim.data[pAinvBinv][-10:], axis=0),
                       np.array([0, 0, 1, 0, 0]), atol=0.15)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
