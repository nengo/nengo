import numpy as np
import pytest

import nengo
from nengo.utils.numpy import rmse
from nengo.utils.testing import Plotter


def test_matrix_mult(Simulator, nl):
    shapeA = (2, 2)
    shapeB = (2, 2)

    Amat = np.random.random(shapeA)
    Bmat = np.random.random(shapeB)

    model = nengo.Network('Matrix Mult Test')
    with model:
        nodeA = nengo.Node(Amat.ravel())
        nodeB = nengo.Node(Bmat.ravel())

        mult_net = nengo.networks.MatrixMult(100, shapeA, shapeB,
                                             neuron_type=nl())

        with mult_net:
            D_p = nengo.Probe(mult_net.output, synapse=0.01)

        nengo.Connection(nodeA, mult_net.inputA)
        nengo.Connection(nodeB, mult_net.inputB)

    dt = 0.001
    sim = Simulator(model, dt=dt)
    sim.run(1)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data[D_p])
        for d in np.dot(Amat, Bmat).flatten():
            plt.axhline(d, color='k')
        plt.savefig('test_matrix_mul.pdf')
        plt.close()

    atol, rtol = .2, .01
    Dmat = np.dot(Amat, Bmat).ravel()
    assert np.allclose(sim.data[D_p][-1], Dmat, atol=atol, rtol=rtol)

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
