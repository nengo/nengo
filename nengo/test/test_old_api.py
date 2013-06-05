from unittest import TestCase

from matplotlib import pyplot as plt
import nose
import numpy as np

from nengo.simulator import Simulator
from nengo.old_api import Network

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


class TestOldAPI(TestCase):
    # -- Tests are in a class so that 
    #    nengo_ocl can automatically run all
    #    member unit tests for other simulators by
    #    subclassing this class and overriding this attribute.
    Simulator = Simulator

    show = False

    def test_direct_mode_simple(self):
        """
        """
        net = Network('Runtime Test', dt=0.001, seed=123,
                     Simulator=self.Simulator)
        net.make_input('in', value=np.sin)
        p = net.make_probe('in', dt_sample=0.001, pstc=0.001)
        net.run(0.01)
        data = p.get_data()
        print data.dtype
        print data
        assert np.allclose(data.flatten(),
                           np.sin(np.arange(0, 0.0095, .001)))


    def test_basic_1(self, N=1000):
        """
        Create a network with sin(t) being represented by
        a population of spiking neurons. Assert that the
        decoded value from the population is close to the
        true value (which is input to the population).

        Expected duration of test: about .7 seconds
        """

        net = Network('Runtime Test', dt=0.001, seed=123,
                     Simulator=self.Simulator)
        print 'make_input'
        net.make_input('in', value=np.sin)
        print 'make A'
        net.make('A', N, 1)
        print 'connecting in -> A'
        net.connect('in', 'A')
        A_fast_probe = net.make_probe('A', dt_sample=0.01, pstc=0.001)
        A_med_probe = net.make_probe('A', dt_sample=0.01, pstc=0.01)
        A_slow_probe = net.make_probe('A', dt_sample=0.01, pstc=0.1)
        in_probe = net.make_probe('in', dt_sample=0.01, pstc=0.01)

        net.run(1.0)

        target = np.sin(np.arange(0, 1000, 10) / 1000.)
        target.shape = (100, 1)

        for speed in 'fast', 'med', 'slow':
            probe = locals()['A_%s_probe' % speed]
            data = np.asarray(probe.get_data()).flatten()
            plt.plot(data, label=speed)

        in_data = np.asarray(in_probe.get_data()).flatten()

        plt.plot(in_data, label='in')
        plt.legend(loc='upper left')

        #print in_probe.get_data()
        #print net.sim.sim_step

        if self.show:
            plt.show()

        # target is off-by-one at the sampling frequency of dt=0.001
        print rmse(target, in_probe.get_data())
        assert rmse(target, in_probe.get_data()) < .001
        print rmse(target, A_fast_probe.get_data())
        assert rmse(target, A_fast_probe.get_data()) < .1, (
            rmse(target, A_fast_probe.get_data()))
        print rmse(target, A_med_probe.get_data())
        assert rmse(target, A_med_probe.get_data()) < .01
        print rmse(target, A_slow_probe.get_data())
        assert rmse(target, A_slow_probe.get_data()) < 0.1

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

        net=Network('Matrix Multiplication', seed=seed,
                   Simulator=self.Simulator)

        # values should stay within the range (-radius,radius)
        radius=1

        # make 2 matrices to store the input
        print "make_array: input matrices A and B"
        net.make_array('A',N,D1*D2,radius=radius, neuron_type='lif')
        net.make_array('B',N,D2*D3,radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        net.make_input('input A',[0]*D1*D2)
        net.make_input('input B',[0]*D2*D3)
        print "connect: input matrices A and B"
        net.connect('input A','A')
        net.connect('input B','B')

        # the C matrix holds the intermediate product calculations
        #  need to compute D1*D2*D3 products to multiply 2 matrices together
        print "make_array: intermediate C"
        net.make_array('C',4 * N,D1*D2*D3,dimensions=2,radius=1.5*radius,
            encoders=[[1,1],[1,-1],[-1,1],[-1,-1]], neuron_type='lif')

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
        transformA=[[0]*(D1*D2) for i in range(D1*D2*D3*2)]
        transformB=[[0]*(D2*D3) for i in range(D1*D2*D3*2)]
        for i in range(D1):
            for j in range(D2):
                for k in range(D3):
                    transformA[(j+k*D2+i*D2*D3)*2][j+i*D2]=1
                    transformB[(j+k*D2+i*D2*D3)*2+1][k+j*D3]=1

        net.connect('A','C',transform=transformA)
        net.connect('B','C',transform=transformB)


        # now compute the products and do the appropriate summing
        print "make_array: output D"
        net.make_array('D',N,D1*D3,radius=radius, neuron_type='lif')

        def product(x):
            return x[0]*x[1]
        # the mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)    

        # XXX index_post is not implemented
        net.connect('C','D',index_post=[i/D2 for i in range(D1*D2*D3)],func=product)

        if 0:
            # is there a portable API for doing this?

            net.get_object('input A').origin['X'].decoded_output.set_value(
                np.asarray([.5, -.5]).astype('float32'))
            net.get_object('input B').origin['X'].decoded_output.set_value(
                np.asarray([0, 1, -1, 0]).astype('float32'))

        Dprobe = net.make_probe('D', dt_sample=0.01, pstc=0.1)

        net.run(1)


        net_data = Dprobe.get_data()
        print net_data.shape
        plt.plot(net_data[:, 0])
        plt.plot(net_data[:, 1])
        if self.show:
            plt.show()

        nose.SkipTest('test correctness')
