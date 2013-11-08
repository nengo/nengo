import logging

import numpy as np

import nengo
import nengo.helpers
from nengo.networks.oscillator import Oscillator
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestOscillator(SimulatorTestCase):
    def test_oscillator(self):
        model = nengo.Model('Oscillator')
        inputs = {0:[1,0],0.5:[0,0]}
        model.make_node('Input', nengo.helpers.piecewise(inputs))

        tau = 0.1
        freq = 5
        model.add(Oscillator('T', tau, freq, neurons=nengo.LIF(100)))
        model.connect('Input', 'T.In')

        model.make_ensemble('A', nengo.LIF(100), dimensions=2)
        model.connect('A', 'A', filter=tau,
                      transform=[[1, -freq*tau], [freq*tau, 1]])
        model.connect('Input', 'A')

        model.probe('Input')
        model.probe('A', filter=0.01)
        model.probe('T.Oscillator', filter=0.01)
        sim = model.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(3.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(model.t)
            plt.plot(t, sim.data('A'), label='Manual')
            plt.plot(t, sim.data('T.Oscillator'), label='Template')
            plt.plot(t, sim.data('Input'), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_oscillator.test_oscillator.pdf')
            plt.close()

        self.assertTrue(rmse(sim.data('A'), sim.data('T.Oscillator')) < 0.3)

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
