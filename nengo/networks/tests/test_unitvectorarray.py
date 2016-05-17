import numpy as np

import nengo


def test_unit_vector_array(Simulator, plt, seed, rng):
    d = 64
    v = np.squeeze(
        nengo.dists.UniformHypersphere(surface=True).sample(1, d, rng))

    with nengo.Network(seed=seed) as model:
        uva = nengo.networks.UnitVectorArray(50, d)
        nengo.Connection(nengo.Node(v), uva.input)
        p = nengo.Probe(uva.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    similarity = np.dot(sim.data[p], v)
    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), similarity)
    plt.xlabel("Time [s]")
    plt.ylabel("Similarity")

    plt.subplot(2, 1, 2)
    for y in v:
        plt.axhline(y, c='b')
    plt.plot(sim.trange(), sim.data[p], c='r', label="actual")
    plt.xlabel("Time [s]")
    plt.ylabel("Component value")

    assert np.all(similarity[sim.trange() > 0.3] > .9)


def test_unit_vector_array_represent_identity(Simulator, plt, seed, rng):
    d = 64
    identity = np.zeros(d)
    identity[0] = 1.

    with nengo.Network(seed=seed) as model:
        uva = nengo.networks.UnitVectorArray(50, d, represent_identity=True)
        nengo.Connection(nengo.Node(identity), uva.input)
        p = nengo.Probe(uva.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    similarity = np.dot(sim.data[p], identity)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), similarity)
    plt.xlabel("Time [s]")
    plt.ylabel("Similarity")

    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[p][:, 0], c='r', label='Dimensions 0')
    plt.plot(sim.trange(), sim.data[p][:, 1:], c='b')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Component value")

    assert np.all(similarity[sim.trange() > 0.3] > .9)
    assert np.all(sim.data[p][sim.trange() > 0.3, 0] > .9)
    assert np.all(sim.data[p][sim.trange() > 0.3, 1:] < .1)
