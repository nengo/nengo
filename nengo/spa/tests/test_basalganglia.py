import pytest

import nengo
from nengo import spa


def test_basal_ganglia(Simulator, seed, plt, allclose):
    model = spa.SPA(seed=seed)

    with model:
        model.vision = spa.Buffer(dimensions=16)
        model.motor = spa.Buffer(dimensions=16)
        model.compare = spa.Compare(dimensions=16)

        # test all acceptable condition formats
        actions = spa.Actions(
            '0.5 --> motor=A',
            'dot(vision, CAT) --> motor=B',
            'dot(vision*CAT, DOG) --> motor=C',
            '2*dot(vision, CAT*0.5) --> motor=D',
            'dot(vision, CAT) + 0.5 - dot(vision,CAT) --> motor=E',
            'dot(vision, PARROT) + compare --> motor=F',
            '0.5*dot(vision, MOUSE) + 0.5*compare --> motor=G',
            '( dot(vision, MOUSE) - compare ) * 0.5 --> motor=H'
        )
        model.bg = spa.BasalGanglia(actions)

        def input(t):
            if t < 0.1:
                return '0'
            elif t < 0.2:
                return 'CAT'
            elif t < 0.3:
                return 'DOG*~CAT'
            elif t < 0.4:
                return 'PARROT'
            elif t < 0.5:
                return 'MOUSE'
            else:
                return '0'
        model.input = spa.Input(vision=input,
                                compare_A='SHOOP', compare_B='SHOOP')
        p = nengo.Probe(model.bg.input, 'output', synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)
    t = sim.trange()

    plt.plot(t, sim.data[p])
    plt.legend(["A", "B", "C", "D", "E", "F", "G", "H"])
    plt.title('Basal Ganglia output')

    # assert the basal ganglia is prioritizing things correctly
    # Motor F
    assert sim.data[p][t == 0.4, 5] > 0.8
    # Motor G
    assert sim.data[p][t == 0.5, 6] > 0.8
    # Motor A
    assert 0.6 > sim.data[p][t == 0.1, 0] > 0.4
    # Motor B
    assert sim.data[p][t == 0.2, 1] > 0.8
    # Motor C
    assert sim.data[p][t == 0.3, 2] > 0.6

    # Motor B should be the same as Motor D
    assert allclose(sim.data[p][:, 1], sim.data[p][:, 3])
    # Motor A should be the same as Motor E
    assert allclose(sim.data[p][:, 0], sim.data[p][:, 4])


def test_errors():
    # dot products between two sources not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('dot(vision, motor) --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # inversion of sources not implemented both ways
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('dot(~vision, FOO) --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.vision = spa.Buffer(dimensions=16)
            model.motor = spa.Buffer(dimensions=16)
            actions = spa.Actions('dot(FOO, ~vision) --> motor=A')
            model.bg = spa.BasalGanglia(actions)

    # convolution not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.scalar = spa.Buffer(dimensions=1, subdimensions=1)
            actions = spa.Actions('scalar*scalar --> scalar=1')
            model.bg = spa.BasalGanglia(actions)
