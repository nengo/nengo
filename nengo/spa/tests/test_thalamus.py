import pytest

import nengo
from nengo import spa


def test_thalamus(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            self.vision = spa.Buffer(dimensions=16, neurons_per_dimension=80)
            self.vision2 = spa.Buffer(dimensions=16, neurons_per_dimension=80)
            self.motor = spa.Buffer(dimensions=16, neurons_per_dimension=80)
            self.motor2 = spa.Buffer(dimensions=32, neurons_per_dimension=80)

            actions = spa.Actions(
                'dot(vision, A) --> motor=A, motor2=vision*vision2',
                'dot(vision, B) --> motor=vision, motor2=vision*A*~B',
                'dot(vision, ~A) --> motor=~vision, motor2=~vision*vision2'
            )
            self.bg = spa.BasalGanglia(actions)
            self.thalamus = spa.Thalamus(self.bg)

            def input_f(t):
                if t < 0.1:
                    return 'A'
                elif t < 0.3:
                    return 'B'
                elif t < 0.5:
                    return '~A'
                else:
                    return '0'
            self.input = spa.Input(vision=input_f, vision2='B*~A')

    model = SPA(seed=30)

    with model:
        input, vocab = model.get_module_input('motor')
        input2, vocab2 = model.get_module_input('motor2')
        p = nengo.Probe(input, 'output', synapse=0.03)
        p2 = nengo.Probe(input2, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.5)

    data = vocab.dot(sim.data[p].T)
    data2 = vocab2.dot(sim.data[p2].T)

    assert 0.9 < data[0, 100] < 1.1  # Action 1
    assert -0.2 < data2[0, 100] < 0.35
    assert -0.2 < data[1, 100] < 0.2
    assert 0.4 < data2[1, 100] < 0.6
    assert -0.2 < data[0, 299] < 0.2  # Action 2
    assert 0.6 < data2[0, 299] < 0.8
    assert 0.8 < data[1, 299] < 1.1
    assert -0.2 < data2[1, 299] < 0.2
    assert 0.8 < data[0, 499] < 1.0  # Action 3
    assert 0.0 < data2[0, 499] < 0.4
    assert -0.2 < data[1, 499] < 0.2
    assert 0.45 < data2[1, 499] < 0.7


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            self.vision = spa.Buffer(dimensions=16)

            actions = spa.Actions(
                '0.5 --> motor=A'
                )
            self.bg = spa.BasalGanglia(actions)
    with pytest.raises(NameError):
        SPA()


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
