import logging
import os

import numpy as np

import nengo
from nengo.templates import EnsembleArray
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestEnsembleArrayCreation(unittest.TestCase):

    def test_n_ensembles(self):
        ea = EnsembleArray('test_n_ensembles', nengo.LIF(1), 1)
        with self.assertRaises(ValueError):
            ea.n_ensembles = 3

    def test_neuron_parititoning(self):
        ea_even = EnsembleArray('Same size', nengo.LIF(10), 5)
        for ens in ea_even.ensembles:
            self.assertEqual(ens.n_neurons, 2)

        ea_odd = EnsembleArray('Different size', nengo.LIF(19), 4)

        # Order of the sizes shouldn't matter
        sizes = [5, 5, 5, 4]
        for ens in ea_odd.ensembles:
            sizes.remove(ens.n_neurons)
        self.assertEqual(len(sizes), 0)

class TestEnsembleArray(SimulatorTestCase):
    def test_multidim(self):
        """Test an ensemble array with multiple dimensions per ensemble"""
        n_neurons = 100
        radius = 1

        ta = np.zeros((6,3))
        ta[[0, 2, 4], [0, 1, 2]] = 1
        tb = np.zeros((6,3))
        tb[[1, 3, 5], [0, 1, 2]] = 1

        rng = np.random.RandomState(523887)
        a = rng.uniform(low=-0.7, high=0.7, size=3)
        b = rng.uniform(low=-0.7, high=0.7, size=3)
        c = np.dot(ta, a) + np.dot(tb, b)

        model = nengo.Model('Matrix Multiplication', seed=123)
        inputA = model.make_node('input A', output=a)
        inputB = model.make_node('input B', output=b)
        A = model.add(EnsembleArray('A', nengo.LIF(n_neurons), 3,
                                    radius=radius))
        B = model.add(EnsembleArray('B', nengo.LIF(n_neurons), 3,
                                    radius=radius))
        C = model.add(EnsembleArray('C', nengo.LIF(n_neurons), 3,
                                    dimensions_per_ensemble=2, radius=radius))

        model.connect(inputA, A)
        model.connect(inputB, B)
        model.connect(A, C, transform=ta)
        model.connect(B, C, transform=tb)

        model.probe(A, filter=0.03)
        model.probe(B, filter=0.03)
        model.probe(C, filter=0.03)

        sim = model.simulator()
        sim.run(1.0)

        t = sim.data(model.t).flatten()
        a_sim = sim.data(A)[t > 0.5].mean(0)
        b_sim = sim.data(B)[t > 0.5].mean(0)
        c_sim = sim.data(C)[t > 0.5].mean(0)

        atol = 0.01
        rtol = 0.05
        self.assertTrue(np.allclose(a, a_sim, atol=atol, rtol=rtol))
        self.assertTrue(np.allclose(b, b_sim, atol=atol, rtol=rtol))
        self.assertTrue(np.allclose(c, c_sim, atol=atol, rtol=rtol))

    def test_matrix_mul(self):
        N = 100

        Amat = np.asarray([[.5, -.5]])
        Bmat = np.asarray([[0, -1.,], [.7, 0]])

        model = nengo.Model('Matrix Multiplication', seed=123)

        radius = 1

        model.add(EnsembleArray('A', nengo.LIF(N * Amat.size),
                                Amat.size, radius=radius))
        model.add(EnsembleArray('B', nengo.LIF(N * Bmat.size),
                                Bmat.size, radius=radius))

        inputA = model.make_node('input A', output=Amat.ravel())
        inputB = model.make_node('input B', output=Bmat.ravel())
        model.connect('input A', 'A')
        model.connect('input B', 'B')
        model.probe('A', sample_every=0.01, filter=0.01)
        model.probe('B', sample_every=0.01, filter=0.01)

        C = model.add(EnsembleArray(
            'C', nengo.LIF(N * Amat.size * Bmat.shape[1] * 2),
            Amat.size * Bmat.shape[1], dimensions_per_ensemble=2,
            radius=1.5 * radius))

        transformA = np.zeros((C.dimensions, Amat.size))
        transformB = np.zeros((C.dimensions, Bmat.size))

        for i in range(Amat.shape[0]):
            for j in range(Amat.shape[1]):
                for k in range(Bmat.shape[1]):
                    tmp = (j + k * Amat.shape[1] + i * Bmat.size)
                    transformA[tmp * 2][j + i * Amat.shape[1]] = 1
                    transformB[tmp * 2 + 1][k + j * Bmat.shape[1]] = 1

        model.connect('A', 'C', transform=transformA)
        model.connect('B', 'C', transform=transformB)
        model.probe('C', sample_every=0.01, filter=0.01)

        D = model.add(EnsembleArray(
            'D', nengo.LIF(N * Amat.shape[0] * Bmat.shape[1]),
            Amat.shape[0] * Bmat.shape[1], radius=radius))

        def product(x):
            return x[0]*x[1]

        transformC = np.zeros((D.dimensions, Bmat.size))
        for i in range(Bmat.size):
            transformC[i / Bmat.shape[0]][i] = 1

        model.connect('C', 'D', function=product, transform=transformC)
        model.probe('D', sample_every=0.01, filter=0.01)

        sim = model.simulator()
        sim.run(1)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data('D'))
            for d in np.dot(Amat, Bmat).flatten():
                plt.axhline(d, color='k')
            plt.savefig('test_ensemble_array.test_matrix_mul.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[50:, 0], 0.5,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('A')[50:, 1], -0.5,
                                    atol=.1, rtol=.01))

        self.assertTrue(np.allclose(sim.data('B')[50:, 0], 0,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('B')[50:, 1], -1,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('B')[50:, 2], .7,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('B')[50:, 3], 0,
                                    atol=.1, rtol=.01))

        Dmat = np.dot(Amat, Bmat)
        for i in range(Amat.shape[0]):
            for k in range(Bmat.shape[1]):
                self.assertTrue(np.allclose(
                    sim.data('D')[-10:, i * Bmat.shape[1] + k],
                    Dmat[i, k],
                    atol=0.1, rtol=0.1), (
                        sim.data('D')[-10:, i * Bmat.shape[1] + k],
                        Dmat[i, k]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
