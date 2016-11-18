import numpy as np

import nengo
from nengo import spa


def test_basic(Simulator, plt, seed):
    with nengo.Network(seed=seed) as model:
        bg = spa.BasalGanglia(action_count=5, seed=seed)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


def test_thalamus(Simulator, plt, seed):

    with nengo.Network(seed=seed) as net:
        bg = spa.BasalGanglia(action_count=5)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)

        thal = spa.Thalamus(action_count=5)
        nengo.Connection(bg.output, thal.input)

        p = nengo.Probe(thal.output, synapse=0.01)

    with Simulator(net) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > 0.8
    assert np.all(output[1:] < 0.01)
