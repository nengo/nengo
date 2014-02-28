import numpy as np
import pytest

import nengo
from nengo import spa


def test_fixed():
    class SPA(spa.SPA):
        def __init__(self):
            super(SPA, self).__init__()
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
            super(SPA, self).__init__()
            self.buffer = spa.Buffer(dimensions=16)

            def input(t):
                if t < 0.1:
                    return 'A'
                elif t < 0.2:
                    return 'B'
                else:
                    return '0'

            self.input = spa.Input(buffer=input)

    model = SPA()

    input, vocab = model.get_module_input('buffer')

    assert np.allclose(model.input.input_nodes['buffer'].output(t=0),
                       vocab.parse('A').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.1),
                       vocab.parse('B').v)
    assert np.allclose(model.input.input_nodes['buffer'].output(t=0.2),
                       np.zeros(16))


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
