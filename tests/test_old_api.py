try:
    import unittest2 as unittest
except ImportError:
    import unittest
import os

import numpy as np

import nengo
import nengo.old_api as nef

from helpers import Plotter, rmse, simulates, SimulatesMetaclass


class TestOldAPI(unittest.TestCase):
    __metaclass__ = SimulatesMetaclass

    @simulates
    def test_prod(self, simulator):

        def product(x):
            return x[0]*x[1]

        N = 250
        seed = 123
        net = nef.Network('Matrix Multiplication', seed=seed,
                          simulator=simulator)

        net.make_input('sin', value=np.sin)
        net.make_input('neg', value=[-.5])
        net.make_array('p', 2 * N, 1, dimensions=2, radius=1.5)
        net.make_array('D', N, 1, dimensions=1)
        net.connect('sin', 'p', transform=[[1], [0]])
        net.connect('neg', 'p', transform=[[0], [1]])
        net.connect('p', 'D', func=product, pstc=0.01)

        p_raw = net._probe_decoded_signals(
            [net.ensembles['p'].origin['product'].sigs[0]],
            dt_sample=.01, pstc=.01)

        probe_p = net.make_probe('p', dt_sample=.01, pstc=.01)
        probe_d = net.make_probe('D', dt_sample=.01, pstc=.01)

        net.run(6)

        data_p = probe_p.get_data()
        data_d = probe_d.get_data()
        data_r = p_raw.get_data()

        with Plotter(simulator) as plt:
            print '1'
            plt.subplot(211);
            plt.plot(data_p)
            plt.plot(np.sin(np.arange(0, 6, .01)))
            plt.subplot(212);
            plt.plot(data_d)
            plt.plot(data_r)
            plt.plot(-.5 * np.sin(np.arange(0, 6, .01)))
            plt.savefig('test_old_api.test_prod.pdf')
            print 'should have saved'
            plt.close()

        assert np.allclose(data_p[:, 0], np.sin(np.arange(0, 6, .01)),
                          atol=.1, rtol=.01)
        assert np.allclose(data_p[20:, 1], -0.5,
                          atol=.1, rtol=.01)

        def match(a, b):
            assert np.allclose(a, b, .1, .1)

        match(data_d[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))
        match(data_r[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))

    @simulates
    def test_multidim_probe(self, simulator):
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

        net = nef.Network('V', seed=seed, simulator=simulator)

        # values should stay within the range (-radius,radius)
        radius = 2.0

        # make 2 matrices to store the input
        print "make_array: input matrices A and B"
        net.make_array('A', neurons=N, array_size=D1 * D2,
            radius=radius, neuron_type='lif')
        net.make_array('B', neurons=N, array_size=D2 * D3,
            radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        inputA = net.make_input('input A', value=Amat.flatten())
        inputB = net.make_input('input B', value=Bmat.flatten())
        print "connect: input matrices A and B"
        net.connect('input A', 'A')
        net.connect('input B', 'B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        print "make_array: intermediate C"
        net.make_array('C', 4 * N, D1 * D2 * D3,
            dimensions=2,
            radius=1.5 * radius,
            encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
            neuron_type='lif')

        transformA=[[0] * (D1 * D2) for i in range(D1 * D2 * D3 * 2)]
        transformB=[[0] * (D2 * D3) for i in range(D1 * D2 * D3 * 2)]
        for i in range(D1):
            for k in range(D3):
                for j in range(D2):
                    tmp = (j + k * D2 + i * D2 * D3)
                    transformA[tmp * 2][j + i * D2] = 1
                    transformB[tmp * 2 + 1][k + j * D3] = 1

        print transformA
        #print transformB

        print "connect A->C"
        net.connect('A', 'C', transform=transformA)
        #print "connect B->C"
        net.connect('B', 'C', transform=transformB)

        Cprobe = net.make_probe('C', dt_sample=0.01, pstc=0.01)

        net.run(1)

        print Cprobe.get_data().shape
        print Amat
        print Bmat
        #assert Cprobe.get_data().shape == (100, D1 * D2 * D3, 2)
        data = Cprobe.get_data()

        with Plotter(simulator) as plt:
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
                    assert np.allclose(
                            data[-10:, 2 * tmp],
                            Amat[i, j],
                            atol=0.1, rtol=0.1), (
                                data[-10:, 2 * tmp],
                                Amat[i, j])

                    assert np.allclose(
                            data[-10:, 1 + 2 * tmp],
                            Bmat[j, k],
                            atol=0.1, rtol=0.1)

    @simulates
    def test_matrix_mul(self, simulator):
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
                          simulator=simulator)

        # values should stay within the range (-radius,radius)
        radius = 1

        # make 2 matrices to store the input
        print "make_array: input matrices A and B"
        net.make_array('A', neurons=N, array_size=D1 * D2,
            radius=radius, neuron_type='lif')
        net.make_array('B', neurons=N, array_size=D2 * D3,
            radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        inputA = net.make_input('input A', value=Amat.ravel())
        inputB = net.make_input('input B', value=Bmat.ravel())
        print "connect: input matrices A and B"
        net.connect('input A', 'A')
        net.connect('input B', 'B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        print "make_array: intermediate C"
        net.make_array('C', 4 * N, D1 * D2 * D3,
            dimensions=2,
            radius=1.5 * radius,
            encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]],
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

        print "connect A->C"
        net.connect('A', 'C', transform=transformA)
        print "connect B->C"
        net.connect('B', 'C', transform=transformB)

        # now compute the products and do the appropriate summing
        print "make_array: output D"
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

        with Plotter(simulator) as plt:
            for i in range(D1):
                for k in range(D3):
                    plt.subplot(D1, D3, i * D3 + k + 1)
                    plt.title('D[%i, %i]' % (i, k))
                    plt.plot(data[:, i * D3 + k])
                    plt.axhline(Dmat[i, k])
                    plt.ylim(-radius, radius)
            plt.savefig('test_old_api.test_matrix_mul.pdf')
            plt.close()

        assert np.allclose(Aprobe.get_data()[50:, 0], 0.5,
                          atol=.1, rtol=.01)
        assert np.allclose(Aprobe.get_data()[50:, 1], -0.5,
                          atol=.1, rtol=.01)

        assert np.allclose(Bprobe.get_data()[50:, 0], 0,
                          atol=.1, rtol=.01)
        assert np.allclose(Bprobe.get_data()[50:, 1], -1,
                          atol=.1, rtol=.01)
        assert np.allclose(Bprobe.get_data()[50:, 2], .7,
                          atol=.1, rtol=.01)
        assert np.allclose(Bprobe.get_data()[50:, 3], 0,
                          atol=.1, rtol=.01)

        for i in range(D1):
            for k in range(D3):
                assert np.allclose(
                        data[-10:, i * D3 + k],
                        Dmat[i, k],
                        atol=0.1, rtol=0.1), (
                            data[-10:, i * D3 + k],
                            Dmat[i, k])


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
