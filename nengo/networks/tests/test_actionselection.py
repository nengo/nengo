import numpy as np

import nengo


def test_basic(Simulator, plt, seed):
    bg = nengo.networks.BasalGanglia(
        dimensions=5, net=nengo.Network(seed=seed))
    with bg:
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    sim = Simulator(bg)
    sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


def test_thalamus(Simulator, plt, seed):

    with nengo.Network(seed=seed) as net:
        bg = nengo.networks.BasalGanglia(dimensions=5)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)

        thal = nengo.networks.Thalamus(dimensions=5)
        nengo.Connection(bg.output, thal.input)

        p = nengo.Probe(thal.output, synapse=0.01)

    sim = Simulator(net)
    sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > 0.8
    assert np.all(output[1:] < 0.01)
