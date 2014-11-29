import numpy as np
import pytest

import nengo
from nengo import spa


def test_basal_ganglia(Simulator, seed, plt):
    model = spa.SPA(seed=seed)

    with model:
        model.vision = spa.Buffer(dimensions=16)
        model.motor = spa.Buffer(dimensions=16)

        actions = spa.Actions(
            '0.5 --> motor=A',
            'dot(vision,CAT) --> motor=B',
            'dot(vision*CAT,DOG) --> motor=C',
            '2*dot(vision,CAT*0.5) --> motor=D',
            'dot(vision,CAT)+0.5-dot(vision,CAT) --> motor=E',
        )
        model.bg = spa.BasalGanglia(actions)

        def input(t):
            if t < 0.1:
                return '0'
            elif t < 0.2:
                return 'CAT'
            elif t < 0.3:
                return 'DOG*~CAT'
            else:
                return '0'
        model.input = spa.Input(vision=input)
        p = nengo.Probe(model.bg.input, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.3)
    t = sim.trange()

    plt.plot(t, sim.data[p])
    plt.title('Basal Ganglia output')

    assert 0.6 > sim.data[p][t == 0.1, 0] > 0.4
    assert sim.data[p][t == 0.2, 1] > 0.8
    assert sim.data[p][-1, 2] > 0.6

    assert np.allclose(sim.data[p][:, 1], sim.data[p][:, 3])
    assert np.allclose(sim.data[p][:, 0], sim.data[p][:, 4])


def test_errors():
    # dot products between two sources not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('dot(vision, motor) --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # inversion of sources not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('dot(~vision, FOO) --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # convolution not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.scalar = spa.Buffer(dimensions=1, subdimensions=1)
            actions = spa.Actions('scalar*scalar --> scalar=1')
            model.bg = spa.BasalGanglia(actions)

    # bias source inputs not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.scalar = spa.Buffer(dimensions=1, subdimensions=1)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('scalar --> motor=A')
            model.bg = spa.BasalGanglia(actions)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
