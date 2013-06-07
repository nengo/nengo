import os
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

    show = int(os.getenv("NENGO_TEST_SHOW", 0))

    def test_counters(self):
        net = Network('foo', dt=0.001, seed=123,
                     Simulator=self.Simulator)

        simtime_probe = net._raw_probe(net.simtime, dt_sample=.001)
        steps_probe = net._raw_probe(net.steps, dt_sample=.001)
        net.run(0.003)
        simtime_data = simtime_probe.get_data()
        steps_data = steps_probe.get_data()
        assert np.allclose(simtime_data.flatten(), [.001, .002, .003])
        assert np.allclose(steps_data.flatten(), [1, 2, 3])


    def test_direct_mode_simple(self):
        """
        """
        net = Network('Runtime Test', dt=0.001, seed=123,
                     Simulator=self.Simulator)
        net.make_input('in', value=np.sin)
        p = net.make_probe('in', dt_sample=0.001, pstc=0.0)
        rawp = net._raw_probe(net.inputs['in'], dt_sample=.001)
        st_probe = net._raw_probe(net.simtime, dt_sample=.001)
        net.run(0.01)

        data = p.get_data()
        raw_data = rawp.get_data()
        st_data = st_probe.get_data()
        print data.dtype
        print st_data
        print raw_data
        assert np.allclose(st_data.ravel(),
                           np.arange(0.001, 0.0105, .001))
        assert np.allclose(raw_data.ravel(),
                           np.sin(np.arange(0, 0.0095, .001)))
        # -- the make_probe call induces a one-step delay
        #    on readout even when the pstc is really small.
        assert np.allclose(data.ravel()[1:],
                           np.sin(np.arange(0, 0.0085, .001)))


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
        in_probe = net.make_probe('in', dt_sample=0.01, pstc=0.0)

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

    def test_vector_input_constant(self):
        # Adjust these values to change the matrix dimensions
        #  Matrix A is D1xD2
        #  Matrix B is D2xD3
        #  result is D1xD3
        D1 = 1
        D2 = 2
        seed = 123
        N = 50

        net=Network('Matrix Multiplication', seed=seed,
                   Simulator=self.Simulator)

        # make 2 matrices to store the input
        print "make_array: input matrices A and B"
        net.make_array('A', neurons=N, array_size=D1*D2, 
            neuron_type='lif')

        # connect inputs to them so we can set their value
        net.make_input('input A', value=[.5, -.5])
        net.connect('input A', 'A')
        inprobe = net.make_probe('input A', dt_sample=0.01, pstc=0.1)
        sprobe = net._probe_signals(
            net.ensembles['A'].input_signals, dt_sample=0.01, pstc=0.01)
        Aprobe = net.make_probe('A', dt_sample=0.01, pstc=0.1)

        net.run(1)

        in_data = inprobe.get_data()
        s_data = sprobe.get_data()
        A_data = Aprobe.get_data()

        plt.subplot(311); plt.plot(in_data)
        plt.subplot(312); plt.plot(s_data)
        plt.subplot(313); plt.plot(A_data)
        if self.show:
            plt.show()

        assert np.allclose(in_data[-10:], [.5, -.5], atol=.01, rtol=.01)
        assert np.allclose(s_data[-10:], [.5, -.5], atol=.01, rtol=.01)
        assert np.allclose(A_data[-10:], [.5, -.5], atol=.01, rtol=.01)


    
    def test_prod(self):

        def product(x):
            return x[0]*x[1]
        #from nengo_theano import Network

        N = 250
        seed = 123
        net=Network('Matrix Multiplication', seed=seed)
                   #Simulator=self.Simulator)

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

        plt.subplot(211);
        plt.plot(data_p)
        plt.plot(np.sin(np.arange(0, 6, .01)))
        plt.subplot(212);
        plt.plot(data_d)
        plt.plot(data_r)
        plt.plot(-.5 * np.sin(np.arange(0, 6, .01)))

        if self.show:
            plt.show()

        assert np.allclose(data_p[:, 0], np.sin(np.arange(0, 6, .01)),
                          atol=.1, rtol=.01)
        assert np.allclose(data_p[20:, 1], -0.5,
                          atol=.1, rtol=.01)

        def match(a, b):
            assert np.allclose(a, b, .1, .1)

        match(data_d[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))

        match(data_r[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))


    def test_matrix_mul(self):
        # Adjust these values to change the matrix dimensions
        #  Matrix A is D1xD2
        #  Matrix B is D2xD3
        #  result is D1xD3
        D1 = 1
        D2 = 2
        D3 = 2
        seed = 123
        N = 500

        net=Network('Matrix Multiplication', seed=seed,
                   Simulator=self.Simulator)

        # values should stay within the range (-radius,radius)
        radius=1

        # make 2 matrices to store the input
        print "make_array: input matrices A and B"
        net.make_array('A', neurons=N, array_size=D1*D2, 
            radius=radius, neuron_type='lif')
        net.make_array('B', neurons=N, array_size=D2*D3, 
            radius=radius, neuron_type='lif')

        # connect inputs to them so we can set their value
        inputA = net.make_input('input A', value=[.5, -.5])
        inputB = net.make_input('input B', value=[0, -1, 1, 0])
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

        print "connect A->C"
        net.connect('A','C',transform=transformA)
        print "connect B->C"
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

        Aprobe = net.make_probe('A', dt_sample=0.01, pstc=0.1)
        Bprobe = net.make_probe('B', dt_sample=0.01, pstc=0.1)
        Cprobe = net.make_probe('C', dt_sample=0.01, pstc=0.1)
        Dprobe = net.make_probe('D', dt_sample=0.01, pstc=0.1)

        net.run(1)

        print Aprobe.get_data().shape
        plt.subplot(411); plt.plot(Aprobe.get_data())
        plt.subplot(412); plt.plot(Bprobe.get_data())
        plt.subplot(413); plt.plot(Cprobe.get_data())
        plt.subplot(414); plt.plot(Dprobe.get_data())
        if self.show:
            plt.show()

        raise nose.SkipTest('test correctness')
