from pprint import pprint
from unittest import TestCase

from matplotlib import pyplot as plt
import nose
import numpy as np

from nengo.nonlinear import LIF
from nengo.model import Model

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


class TestNewAPI(TestCase):

    show = False

    def test_direct_mode_simple(self):
        """
        """
        model = Model('Runtime Test', seed=123, backend='numpy')
        model.make_node('in', output=np.sin)
        model.probe('in')
        res = model.run(0.01)
        data = res['in']
        print data.dtype
        print data
        assert np.allclose(data.flatten(), np.sin(np.arange(0, 0.0095, .001)))


    def test_basic_1(self, N=1000):
        """
        Create a network with sin(t) being represented by
        a population of spiking neurons. Assert that the
        decoded value from the population is close to the
        true value (which is input to the population).

        Expected duration of test: about .7 seconds
        """

        model = Model('Runtime Test', seed=123, backend='numpy')

        model.make_node('in', output=np.sin)
        model.make_ensemble('A', LIF(N), 1)
        model.connect('in', 'A')
        model.probe('A', sample_every=0.01, pstc=0.001)  # 'A'
        model.probe('A', sample_every=0.01, pstc=0.01)  # 'A_1'
        model.probe('A', sample_every=0.01, pstc=0.1)  # 'A_2'
        model.probe('in', sample_every=0.01, pstc=0.01)

        pprint(model.o)
        res = model.run(1.0)

        target = np.sin(np.arange(0, 1000, 10) / 1000.)
        target.shape = (100, 1)

        for A, label in (('A', 'fast'), ('A_1', 'med'), ('A_2', 'slow')):
            data = np.asarray(res[A]).flatten()
            plt.plot(data, label=label)

        in_data = np.asarray(res['in']).flatten()

        plt.plot(in_data, label='in')
        plt.legend(loc='upper left')

        #print in_probe.get_data()
        #print net.sim.sim_step

        if self.show:
            plt.show()

        # target is off-by-one at the sampling frequency of dt=0.001
        print rmse(target, res['in'])
        assert rmse(target, res['in']) < .001
        print rmse(target, res['A'])
        assert rmse(target, res['A']) < .3
        print rmse(target, res['A_1'])
        assert rmse(target, res['A_1']) < .03
        print rmse(target, res['A_2'])
        assert rmse(target, res['A_2']) < 0.1

    def test_basic_5K(self):
        return self.test_basic_1(5000)

    def test_matrix_mul(self):
        # Adjust these values to change the matrix dimensions
        #  Matrix A is D1xD2
        #  Matrix B is D2xD3
        #  result is D1xD3
        D1 = 1
        D2 = 2
        D3 = 3
        seed = 123
        N = 50

        model = Model('Matrix Multiplication', seed=seed, backend='numpy')

        # values should stay within the range (-radius,radius)
        radius = 1

        # make 2 matrices to store the input
        model.make_ensemble('A', LIF(N), D1*D2, radius=radius)
        model.make_ensemble('B', LIF(N), D2*D3, radius=radius)

        # connect inputs to them so we can set their value
        model.make_node('input A', [0] * D1 * D2)
        model.make_node('input B', [0] * D2 * D3)
        model.connect('input A', 'A')
        model.connect('input B', 'B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        model.make_ensemble('C', LIF(4 * N), D1 * D2 * D3, # dimensions=2,
                            radius=1.5*radius)
                            # encoders=[[1,1], [1,-1], [-1,1], [-1,-1]])

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
                    ix = (j + k * D2 + i * D2 * D3) * 2
                    transformA[ix][j + i * D2] = 1
                    transformB[ix + 1][k + j * D3] = 1

        model.connect('A', 'C', transform=transformA)
        model.connect('B', 'C', transform=transformB)

        # now compute the products and do the appropriate summing
        model.make_ensemble('D', LIF(N), D1 * D3, radius=radius)

        def product(x):
            return x[0] * x[1]
        # the mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)
        model.connect('C', 'D', index_post=[i / D2 for i in range(D1*D2*D3)],
                      func=product)

        model.get('input A').origin['X'].decoded_output.set_value(
            np.asarray([.5, -.5]).astype('float32'))
        model.get('input B').origin['X'].decoded_output.set_value(
            np.asarray([0, 1, -1, 0]).astype('float32'))

        pprint(model.o)

        Dprobe = model.probe('D')

        model.run(1)

        net_data = Dprobe.get_data()
        print net_data.shape
        plt.plot(net_data[:, 0])
        plt.plot(net_data[:, 1])
        if self.show:
            plt.show()

        nose.SkipTest('test correctness')
