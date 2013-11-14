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
        a = m.make_ensemble('A', nengo.LIF(N), dimensions=1)
        m.make_node('in', output=np.sin)
        m.make_node('inh', piecewise({0:0,2.5:1}))
        m.make_node('ideal', piecewise({0:np.sin,2.5:0}))

        m.connect('in', 'A')

        con = m.connect('inh', a.neurons, transform=[[-2.5]]*N)

        m.probe('in')
        m.probe('A', filter=0.1)
        m.probe('inh')
        m.probe('ideal')

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(5.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approx, filter=0.1')
            plt.plot(t, sim.data('inh'), label='Inhib signal')
            plt.plot(t, sim.data('ideal'), label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_neurons_to_neurons.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[-10:], sim.data('ideal')[-10:],
                                    atol=.1, rtol=.01))


    def test_decoded_to_neurons(self):
        N = 30

        m = nengo.Model('test_decoded_to_neurons', seed=123)
        a = m.make_ensemble('A', nengo.LIF(N), dimensions=1)
        b = m.make_ensemble('B', nengo.LIF(N), dimensions=1)
        m.make_node('in', output=np.sin)
        m.make_node('inh', piecewise({0:0,2.5:1}))
        m.make_node('ideal', piecewise({0:np.sin,2.5:0}))

        m.connect('in', 'A')
        m.connect('inh', 'B')

        con = m.connect('B', a.neurons, transform=[[-2.5]]*N)

        m.probe('in')
        m.probe('A', filter=0.1)
        m.probe('B', filter=0.1)
        m.probe('inh')
        m.probe('ideal')

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(5.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approx, pstc=0.1')
            plt.plot(t, sim.data('B'), label='Neuron approx of inhib sig, pstc=0.1')
            plt.plot(t, sim.data('inh'), label='Inhib signal')
            plt.plot(t, sim.data('ideal'), label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_decoded_to_neurons.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[-10:], sim.data('ideal')[-10:],
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('B')[-10:], sim.data('inh')[-10:],
                                    atol=.1, rtol=.01))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
