"""This is a file to test the network array function, both with make_array, 
and by using the array_size parameter in the network.make command.

"""

import unittest

# Attempt to import nef from nef_theano, if that fails assume this script is being run
# from nengo itself. This is to support running this unittest from nengo itself
try:
    from .. import nef_theano as nef
    is_nef_theano = True
except ImportError:
    import nef
    is_nef_theano = False

if is_nef_theano:
    import matplotlib.pyplot as plt
    import numpy as np
else:
    import numeric as np

class TestArray(unittest.TestCase):
    def setUp(self):
        self.runtime = 2            # Runtime in secs
        self.sample_dt = 0.01       # dt between each sample of the probes
        self.timesteps = self.runtime / self.sample_dt 
        self.pstc = 0.01

        self.rand_seed = 50

        print "%s: Building network" % (self.__class__.__name__)
        self.neurons = 40

        self.net = nef.Network('Array Test', seed = self.rand_seed)

        self.net.make_input('in', np.arange(-1, 1, .34), zero_after = 1.0)
        #self.net.make_input('in', value = 1, zero_after = 1.0)
        self.net.make_array('A', neurons = self.neurons, array_size = 1, dimensions = 6)
        self.net.make('A2', neurons = self.neurons, array_size = 2, dimensions = 3)
        self.net.make('B', neurons = self.neurons, array_size = 3, dimensions = 2)
        self.net.make('B2', neurons = self.neurons, array_size = 6, dimensions = 1)

        self.net.connect('in', 'A')
        self.net.connect('in', 'A2')
        self.net.connect('in', 'B')
        self.net.connect('in', 'B2')
        

    def runTest(self):
        # Make probes
        if is_nef_theano:
            self.Ip  = self.net.make_probe('in', dt_sample = self.sample_dt, pstc = self.pstc)
            self.Ap  = self.net.make_probe('A' , dt_sample = self.sample_dt, pstc = self.pstc)
            self.A2p = self.net.make_probe('A2', dt_sample = self.sample_dt, pstc = self.pstc)
            self.Bp  = self.net.make_probe('B' , dt_sample = self.sample_dt, pstc = self.pstc)
            self.B2p = self.net.make_probe('B2', dt_sample = self.sample_dt, pstc = self.pstc)
        else:
            # TODO: Implement net.make_probe for nengo?
            self.Ip  = None
        
        print "%s: Starting simulation" % (self.__class__.__name__)
        self.net.run(self.runtime)

    
    def plot(self):
        if is_nef_theano:
            # Generate t axis for plots
            t = np.linspace(self.sample_dt, self.runtime, self.timesteps)

            plt.ioff(); plt.close(); 
            plt.subplot(5,1,1); plt.ylim([-1.5,1.5])
            plt.plot(t, self.Ip.get_data(), 'x'); plt.title('Input')
            plt.subplot(5,1,2); plt.ylim([-1.5,1.5])
            plt.plot(self.Ap.get_data()); plt.title('A, array_size = 1, dim = 6')
            plt.subplot(5,1,3); plt.ylim([-1.5,1.5])
            plt.plot(self.A2p.get_data()); plt.title('A2, array_size = 2, dim = 3')
            plt.subplot(5,1,4); plt.ylim([-1.5,1.5])
            plt.plot(self.Bp.get_data()); plt.title('B, array_size = 3, dim = 2')
            plt.subplot(5,1,5); plt.ylim([-1.5,1.5])
            plt.plot(self.B2p.get_data()); plt.title('B2, array_size = 6, dim = 1')
            plt.tight_layout()
            plt.show()
        else:
            # TODO: Implement plotting functionality for nengo?
            return

testarray = TestArray()
testarray.setUp()
testarray.runTest()
testarray.plot()