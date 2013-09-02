import logging
import os

import numpy as np

import nengo
from nengo.templates.ensemble_array import EnsembleArray
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestEnsembleArrayEncoders(unittest.TestCase):

    @staticmethod
    def _random_encoders(n_neurons, *dims):
        encoders = np.random.randn(*dims)
        return encoders

    @staticmethod
    def _normalize(encoders, n_neurons, *dims):
        normed = np.copy(encoders)
        if normed.shape == ():
            normed.shape = (1,)
        if normed.shape == (dims[0],):
            normed = np.tile(normed, (n_neurons, 1))
        norm = np.sum(normed * normed, axis=1)[:, np.newaxis]
        normed /= np.sqrt(norm)
        return normed

    def _test_encoders(self, n_ensembles, n_dimensions):
        n_neurons = 10
        other_args = {'name': 'A',
                      'neurons_per_ensemble': nengo.LIF(n_neurons),
                      'n_ensembles': n_ensembles,
                      'dimensions_per_ensemble': n_dimensions}

        logger.debug("No encoders")
        self.assertIsNotNone(EnsembleArray(encoders=None, **other_args))

        logger.debug("One set of one encoder for all neurons")
        encoders = self._random_encoders(n_neurons, n_dimensions)
        ea = EnsembleArray(encoders=encoders, **other_args)
        normed = self._normalize(encoders, n_neurons, n_dimensions)
        for ens in ea.ensembles:
            self.assertTrue(np.allclose(normed, ens.encoders),
                            (normed, ens.encoders))

        logger.debug("One set of encoders specific to each neuron")
        encoders = self._random_encoders(n_neurons, n_neurons, n_dimensions)
        ea = EnsembleArray(encoders=encoders, **other_args)
        normed = self._normalize(encoders, n_neurons, n_neurons, n_dimensions)
        for ens in ea.ensembles:
            self.assertTrue(np.allclose(normed, ens.encoders),
                            (normed, ens.encoders))

        logger.debug("All encoders specified")
        encoders = self._random_encoders(
            n_neurons, n_ensembles, n_neurons, n_dimensions)
        ea = EnsembleArray(encoders=encoders, **other_args)
        normed = [self._normalize(enc, n_neurons, n_neurons, n_dimensions)
                  for enc in encoders]
        for enc, ens in zip(normed, ea.ensembles):
            self.assertTrue(np.allclose(enc, ens.encoders), (enc, ens.encoders))

    def test_encoders(self):
        self._test_encoders(3, 4)
        self._test_encoders(4, 4)

    def test_encoders_weird(self):
        self._test_encoders(1, 1)

    def test_encoders_one_ensemble(self):
        self._test_encoders(1, 5)

    def test_encoders_one_dimension(self):
        self._test_encoders(6, 1)

class TestEnsembleArray(SimulatorTestCase):

    def test_multidim_probe(self):
        # Adjust these values to change the matrix dimensions
        #  Matrix A is D1xD2
        #  Matrix B is D2xD3
        #  result is D1xD3
        D1 = 1
        D2 = 2
        D3 = 3
        seed = 123
        N = 200

        Amat = np.asarray([[.4, .8]])
        Bmat = np.asarray([[-1.0, -0.6, -.15], [0.25, .5, .7]])

        net = nef.Network('Multidim array', seed=seed, simulator=self.Simulator)

        # values should stay within the range (-radius,radius)
        radius = 2.0

        # make 2 matrices to store the input
        logging.debug("make_array: input matrices A and B")
        net.make_array('A', neurons=N, array_size=D1 * D2,
            radius=radius, neuron_type='lif')
        net.make_array('B', neurons=N, array_size=D2 * D3,
            radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        inputA = net.make_input('input A', value=Amat.flatten())
        inputB = net.make_input('input B', value=Bmat.flatten())
        logging.debug("connect: input matrices A and B")
        net.connect('input A', 'A')
        net.connect('input B', 'B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        logging.debug("make_array: intermediate C")
        net.make_array('C', N * 2, D1 * D2 * D3,
            dimensions=2,
            radius=1.5 * radius,
            neuron_type='lif')

        transformA = [[0] * (D1 * D2) for i in range(D1 * D2 * D3 * 2)]
        transformB = [[0] * (D2 * D3) for i in range(D1 * D2 * D3 * 2)]
        for i in range(D1):
            for k in range(D3):
                for j in range(D2):
                    tmp = (j + k * D2 + i * D2 * D3)
                    transformA[tmp * 2][j + i * D2] = 1
                    transformB[tmp * 2 + 1][k + j * D3] = 1

        transformA = np.asarray(transformA)
        transformB = np.asarray(transformB)

        logging.debug("A->C trans: %s, shape=%s",
                      str(transformA), transformA.shape)
        logging.debug("B->C trans: %s, shape=%s",
                      str(transformB), transformB.shape)

        logging.debug("connect A->C")
        net.connect('A', 'C', transform=transformA)
        logging.debug("connect B->C")
        net.connect('B', 'C', transform=transformB)

        net.make_probe('C', dt_sample=0.01, pstc=0.01)

        net.run(1)

        logging.debug("Cprobe.shape=%s", str(net.model.data['C'].shape))
        logging.debug("Amat=%s", str(Amat))
        logging.debug("Bmat=%s", str(Bmat))
        data = net.model.data['C']

        with Plotter(self.Simulator) as plt:
            for i in range(D1):
                for k in range(D3):
                    for j in range(D2):
                        tmp = (j + k * D2 + i * D2 * D3)
                        plt.subplot(D1 * D2 * D3, 2, 1 + 2 * tmp);
                        plt.title('A[%i, %i]' % (i, j))
                        plt.axhline(Amat[i, j])
                        plt.ylim(-radius, radius)
                        plt.plot(data[:, 2 * tmp])

                        plt.subplot(D1 * D2 * D3, 2, 2 + 2 * tmp);
                        plt.title('B[%i, %i]' % (j, k))
                        plt.axhline(Bmat[j, k])
                        plt.ylim(-radius, radius)
                        plt.plot(data[:, 2 * tmp + 1])
            plt.savefig('test_old_api.test_multidimprobe.pdf')
            plt.close()

        for i in range(D1):
            for k in range(D3):
                for j in range(D2):
                    tmp = (j + k * D2 + i * D2 * D3)
                    self.assertTrue(np.allclose(
                        data[-10:, 2 * tmp],
                        Amat[i, j],
                        atol=0.1, rtol=0.1), (
                            data[-10:, 2 * tmp],
                            Amat[i, j]))

                    self.assertTrue(np.allclose(
                        data[-10:, 1 + 2 * tmp],
                        Bmat[j, k],
                        atol=0.1, rtol=0.1))

    def test_matrix_mul(self):
        # Adjust these values to change the matrix dimensions
        #  Matrix A is D1xD2
        #  Matrix B is D2xD3
        #  result is D1xD3
        D1 = 1
        D2 = 2
        D3 = 2
        seed = 123
        N = 200

        Amat = np.asarray([[.5, -.5]])
        Bmat = np.asarray([[0, -1.,], [.7, 0]])

        net = nef.Network('Matrix Multiplication', seed=seed,
                          simulator=self.Simulator)

        # values should stay within the range (-radius,radius)
        radius = 1

        # make 2 matrices to store the input
        logging.debug("make_array: input matrices A and B")
        net.make_array('A', neurons=N, array_size=D1 * D2,
            radius=radius, neuron_type='lif')
        net.make_array('B', neurons=N, array_size=D2 * D3,
            radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        inputA = net.make_input('input A', value=Amat.ravel())
        inputB = net.make_input('input B', value=Bmat.ravel())
        logging.debug("connect: input matrices A and B")
        net.connect('input A', 'A')
        net.connect('input B', 'B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        logging.debug("make_array: intermediate C")
        net.make_array('C', 4 * N, D1 * D2 * D3,
            dimensions=2,
            radius=1.5 * radius,
            # encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
            neuron_type='lif')

        #  determine the transformation matrices to get the correct pairwise
        #  products computed.  This looks a bit like black magic but if
        #  you manually try multiplying two matrices together, you can see
        #  the underlying pattern.  Basically, we need to build up D1*D2*D3
        #  pairs of numbers in C to compute the product of.  If i,j,k are the
        #  indexes into the D1*D2*D3 products, we want to compute the product
        #  of element (i,j) in A with the element (j,k) in B.  The index in
        #  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
        #  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
        #  two values per ensemble.  We add 1 to the B index so it goes into
        #  the second value in the ensemble.
        transformA = [[0] * (D1 * D2) for i in range(D1 * D2 * D3 * 2)]
        transformB = [[0] * (D2 * D3) for i in range(D1 * D2 * D3 * 2)]
        for i in range(D1):
            for j in range(D2):
                for k in range(D3):
                    tmp = (j + k * D2 + i * D2 * D3)
                    transformA[tmp * 2][j + i * D2] = 1
                    transformB[tmp * 2 + 1][k + j * D3] = 1

        logging.debug("connect A->C")
        net.connect('A', 'C', transform=transformA)
        logging.debug("connect B->C")
        net.connect('B', 'C', transform=transformB)

        # now compute the products and do the appropriate summing
        logging.debug("make_array: output D")
        net.make_array('D', N , D1 * D3,
            radius=radius,
            neuron_type='lif')

        def product(x):
            return x[0]*x[1]
        # the mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)

        net.connect('C', 'D',
            index_post=[i / D2 for i in range(D1 * D2 * D3)], func=product)

        Aprobe = net.make_probe('A', dt_sample=0.01, pstc=0.01)
        Bprobe = net.make_probe('B', dt_sample=0.01, pstc=0.01)
        Cprobe = net.make_probe('C', dt_sample=0.01, pstc=0.01)
        Dprobe = net.make_probe('D', dt_sample=0.01, pstc=0.01)

        prod_probe = net._probe_decoded_signals(
            net.ensembles['C'].origin['product'].sigs,
            dt_sample=0.01,
            pstc=.01)

        net.run(1)

        Dmat = np.dot(Amat, Bmat)
        data = Dprobe.get_data()

        with Plotter(self.Simulator) as plt:
            for i in range(D1):
                for k in range(D3):
                    plt.subplot(D1, D3, i * D3 + k + 1)
                    plt.title('D[%i, %i]' % (i, k))
                    plt.plot(data[:, i * D3 + k])
                    plt.axhline(Dmat[i, k])
                    plt.ylim(-radius, radius)
            plt.savefig('test_old_api.test_matrix_mul.pdf')
            plt.close()

        self.assertTrue(np.allclose(Aprobe.get_data()[50:, 0], 0.5,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(Aprobe.get_data()[50:, 1], -0.5,
                                    atol=.1, rtol=.01))

        self.assertTrue(np.allclose(Bprobe.get_data()[50:, 0], 0,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(Bprobe.get_data()[50:, 1], -1,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(Bprobe.get_data()[50:, 2], .7,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(Bprobe.get_data()[50:, 3], 0,
                                    atol=.1, rtol=.01))

        for i in range(D1):
            for k in range(D3):
                self.assertTrue(np.allclose(
                    data[-10:, i * D3 + k],
                    Dmat[i, k],
                    atol=0.1, rtol=0.1), (
                        data[-10:, i * D3 + k],
                        Dmat[i, k]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
