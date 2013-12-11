import logging

import nengo
import nengo.helpers
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)


class TestIntegrator(SimulatorTestCase):
    def test_integrator(self):
        model = nengo.Model('Integrator')
        with model:
            inputs = {0: 0, 0.2: 1, 1: 0, 2: -2, 3: 0, 4: 1, 5: 0}
            input = nengo.Node(nengo.helpers.piecewise(inputs))

            tau = 0.1
            T = nengo.networks.Integrator(
                tau, neurons=nengo.LIF(100), dimensions=1)
            nengo.Connection(input, T.input, filter=tau)

            A = nengo.Ensemble(nengo.LIF(100), dimensions=1)
            nengo.Connection(A, A, transform=[[1]], filter=tau)
            nengo.Connection(input, A, transform=[[tau]], filter=tau)

            input_p = nengo.Probe(input, 'output')
            A_p = nengo.Probe(A, 'decoded_output', filter=0.01)
            T_p = nengo.Probe(T.ensemble, 'decoded_output', filter=0.01)

        sim = self.Simulator(model, dt=0.001)
        sim.run(6.0)

        with Plotter(self.Simulator) as plt:
            t = sim.trange()
            plt.plot(t, sim.data(A_p), label='Manual')
            plt.plot(t, sim.data(T_p), label='Template')
            plt.plot(t, sim.data(input_p), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_integrator.test_integrator.pdf')
            plt.close()

        self.assertTrue(rmse(sim.data(A_p), sim.data(T_p)) < 0.2)

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
