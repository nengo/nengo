import numpy as np

import nengo


def test_matrix_mult(Simulator, rng, nl, plt):
    shape_a = (2, 2)
    shape_b = (2, 2)

    Amat = rng.rand(*shape_a)
    Bmat = rng.rand(*shape_b)

    with nengo.Network("Matrix multiplication test") as model:
        node_a = nengo.Node(Amat.ravel())
        node_b = nengo.Node(Bmat.ravel())

        with nengo.Config(nengo.Ensemble) as cfg:
            cfg[nengo.Ensemble].neuron_type = nl()
            mult_net = nengo.networks.MatrixMult(100, shape_a, shape_b)

        p = nengo.Probe(mult_net.output, synapse=0.01)

        nengo.Connection(node_a, mult_net.input_a)
        nengo.Connection(node_b, mult_net.input_b)

    dt = 0.001
    sim = Simulator(model, dt=dt)
    sim.run(1)

    t = sim.trange()
    plt.plot(t, sim.data[p])
    for d in np.dot(Amat, Bmat).flatten():
        plt.axhline(d, color='k')

    atol, rtol = .2, .01
    Dmat = np.dot(Amat, Bmat).ravel()
    assert np.allclose(sim.data[p][-1], Dmat, atol=atol, rtol=rtol)
