import numpy as np
import pytest

import nengo
from nengo import spa


def test_fixed():
    class SPA(spa.SPA):
        def __init__(self):
            self.buffer1 = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=8, subdimensions=2)
            self.input = spa.Input(buffer1='A', buffer2='B')

    model = SPA()

    input1, vocab1 = model.get_module_input('buffer1')
    input2, vocab2 = model.get_module_input('buffer2')

    assert np.allclose(model.input.input_nodes['buffer1'].output,
                       vocab1.parse('A').v)
    assert np.allclose(model.input.input_nodes['buffer2'].output,
                       vocab2.parse('B').v)


def test_time_varying():
    class SPA(spa.SPA):
        def __init__(self):
            self.buffer = spa.Buffer(dimensions=16)
            self.buffer2 = spa.Buffer(dimensions=16)

            def input(t):
                if t < 0.1:
                    return 'A'
                elif t < 0.2:
                    return 'B'
                else:
                    return '0'

            self.input = spa.Input(buffer=input, buffer2='B')

    model = SPA()

    input, vocab = model.get_module_input('buffer')

    assert np.allclose(model.input.input_nodes['buffer'].output(t=0),
                       vocab.parse('A').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.1),
                       vocab.parse('B').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.2),
                       np.zeros(16))


def test_predefined_vocabs():
    class SPA(spa.SPA):
        def __init__(self):
            D = 64
            self.vocab1 = spa.Vocabulary(D)
            self.vocab1.parse('A+B+C')

            self.vocab2 = spa.Vocabulary(D)
            self.vocab1.parse('A+B+C')

            self.buffer1 = spa.Buffer(dimensions=D, vocab=self.vocab1)
            self.buffer2 = spa.Buffer(dimensions=D, vocab=self.vocab2)

            def input(t):
                if t < 0.1:
                    return 'A'
                elif t < 0.2:
                    return 'B'
                else:
                    return 'C'

            self.input = spa.Input(buffer1=input, buffer2=input)

    model = SPA()

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

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
