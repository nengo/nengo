import logging

import numpy as np

import nengo
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest
from nengo.helpers import piecewise

logger = logging.getLogger(__name__)

class TestConnection(SimulatorTestCase):

    def test_neurons_to_neurons(self):
        N = 30

        m = nengo.Model('test_neurons_to_neurons', seed=123)
        with m:
            a = nengo.Ensemble(nengo.LIF(N), dimensions=1)
            input = nengo.Node(output=np.sin)
            inh = nengo.Node(piecewise({0:0,2.5:1}))
            ideal = nengo.Node(piecewise({0:np.sin,2.5:0}))
    
            nengo.Connection(input, a)
    
            con = nengo.Connection(inh, a.neurons, transform=[[-2.5]]*N)
    
            in_p = nengo.Probe(input, 'output')
            a_p = nengo.Probe(a, 'decoded_output', filter=0.1)
            inh_p = nengo.Probe(inh, 'output')
            ideal_p = nengo.Probe(ideal, 'output')

        sim = self.Simulator(m, dt=0.001)
        sim.run(5.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t_probe)
            plt.plot(t, sim.data(in_p), label='Input')
            plt.plot(t, sim.data(a_p), label='Neuron approx, filter=0.1')
            plt.plot(t, sim.data(inh_p), label='Inhib signal')
            plt.plot(t, sim.data(ideal_p), label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_neurons_to_neurons.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a_p)[-10:], sim.data(ideal_p)[-10:],
                                    atol=.1, rtol=.01))


    def test_decoded_to_neurons(self):
        N = 30

        m = nengo.Model('test_decoded_to_neurons', seed=123)
        with m:
            a = nengo.Ensemble(nengo.LIF(N), dimensions=1)
            b = nengo.Ensemble(nengo.LIF(N), dimensions=1)
            input = nengo.Node(output=np.sin)
            inh = nengo.Node(piecewise({0:0,2.5:1}))
            ideal = nengo.Node(piecewise({0:np.sin,2.5:0}))
    
            nengo.Connection(input, a)
            nengo.Connection(inh, b)
    
            con = nengo.DecodedConnection(b, a.neurons, transform=[[-2.5]]*N)
    
            in_p = nengo.Probe(input, 'output')
            a_p = nengo.Probe(a, 'decoded_output', filter=0.1)
            b_p = nengo.Probe(b, 'decoded_output', filter=0.1)
            inh_p = nengo.Probe(inh, 'output')
            ideal_p = nengo.Probe(ideal, 'output')

        sim = self.Simulator(m, dt=0.001)
        sim.run(5.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t_probe)
            plt.plot(t, sim.data(in_p), label='Input')
            plt.plot(t, sim.data(a_p), label='Neuron approx, pstc=0.1')
            plt.plot(t, sim.data(b_p), label='Neuron approx of inhib sig, pstc=0.1')
            plt.plot(t, sim.data(inh_p), label='Inhib signal')
            plt.plot(t, sim.data(ideal_p), label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_decoded_to_neurons.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a_p)[-10:], sim.data(ideal_p)[-10:],
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data(b_p)[-10:], sim.data(inh_p)[-10:],
                                    atol=.1, rtol=.01))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
