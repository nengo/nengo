from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

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

    def test_basic_1(self):
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
        net.make('A', 1000, 1)
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
        assert rmse(target, A_fast_probe.get_data()) < .35
        print rmse(target, A_med_probe.get_data())
        assert rmse(target, A_med_probe.get_data()) < .035
        print rmse(target, A_slow_probe.get_data())
        assert rmse(target, A_slow_probe.get_data()) < 0.1

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
        assert np.allclose(data[1:].flatten(),
                           np.sin(np.arange(0, 0.0085, .001)))

