import numpy as np
import pytest

import nengo
from nengo import spa


def test_basal_ganglia(Simulator):
    class SPA(spa.SPA):
        def __init__(self):
            self.vision = spa.Buffer(dimensions=16)
            self.motor = spa.Buffer(dimensions=16)

            actions = spa.Actions(
                '0.5 --> motor=A',
                'dot(vision,CAT) --> motor=B',
                'dot(vision*CAT,DOG) --> motor=C',
                '2*dot(vision,CAT*0.5) --> motor=D',
                'dot(vision,CAT)+0.5-dot(vision,CAT) --> motor=E',
                )
            self.bg = spa.BasalGanglia(actions)

            def input(t):
                if t < 0.1:
                    return '0'
                elif t < 0.2:
                    return 'CAT'
                elif t < 0.3:
                    return 'DOG*~CAT'
                else:
                    return '0'
            self.input = spa.Input(vision=input)

    model = SPA(seed=128)

    with model:
        p = nengo.Probe(model.bg.input, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.3)

    assert 0.55 > sim.data[p][100, 0] > 0.45
    assert 1.1 > sim.data[p][200, 1] > 0.85
    assert 0.8 > sim.data[p][299, 2] > 0.6

    assert np.allclose(sim.data[p][:, 1], sim.data[p][:, 3])
    assert np.allclose(sim.data[p][:, 0], sim.data[p][:, 4])


def test_errors():
    class SPA(spa.SPA):
        def __init__(self):
            self.vision = spa.Buffer(dimensions=16)
            self.motor = spa.Buffer(dimensions=16)

            actions = spa.Actions(
                'dot(vision, motor) --> motor=A'
                )
            self.bg = spa.BasalGanglia(actions)
    with pytest.raises(NotImplementedError):
        SPA()


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
