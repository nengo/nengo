import logging

import nengo
import nengo.helpers
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)


class TestOscillator(SimulatorTestCase):
    def test_oscillator(self):
        model = nengo.Model('Oscillator')

        with model:
            inputs = {0: [1, 0], 0.5: [0, 0]}
            input = nengo.Node(nengo.helpers.piecewise(inputs), label='Input')

            tau = 0.1
            freq = 5
            T = nengo.networks.Oscillator(
                tau, freq, label="Oscillator", neurons=nengo.LIF(100))
            nengo.Connection(input, T.input)

            A = nengo.Ensemble(nengo.LIF(100), label='A', dimensions=2)
            nengo.DecodedConnection(A, A, filter=tau,
                                    transform=[[1, -freq*tau], [freq*tau, 1]])
            nengo.Connection(input, A)

            in_probe = nengo.Probe(input, "output")
            A_probe = nengo.Probe(A, "decoded_output", filter=0.01)
            T_probe = nengo.Probe(T.ensemble, "decoded_output", filter=0.01)

        sim = self.Simulator(model, dt=0.001)
        sim.run(3.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(model.t_probe)
            plt.plot(t, sim.data(A_probe), label='Manual')
            plt.plot(t, sim.data(model.t_probe), label='Template')
            plt.plot(t, sim.data(in_probe), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_oscillator.test_oscillator.pdf')
            plt.close()

        self.assertTrue(rmse(sim.data(A_probe), sim.data(T_probe)) < 0.3)

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
