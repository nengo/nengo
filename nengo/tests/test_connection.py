import logging

import numpy as np

import nengo
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest
from nengo.helpers import piecewise

logger = logging.getLogger(__name__)


class TestConnection(SimulatorTestCase):
    def test_node_to_neurons(self):
        name = 'node_to_neurons'
        N = 30

        m = nengo.Model(name, seed=123)
        a = nengo.Ensemble(nengo.LIF(N), dimensions=1)
        inn = nengo.Node(output=np.sin)
        inh = nengo.Node(piecewise({0: 0, 2.5: 1}))
        nengo.Connection(inn, a)
        con = nengo.Connection(inh, a.neurons, transform=[[-2.5]]*N)

        inn_p = nengo.Probe(inn, 'output')
        a_p = nengo.Probe(a, 'decoded_output', filter=0.1)
        inh_p = nengo.Probe(inh, 'output')

        sim = self.Simulator(m)
        sim.run(5.0)
        t = sim.trange()
        ideal = np.sin(t)
        ideal[t >= 2.5] = 0

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data(inn_p), label='Input')
            plt.plot(t, sim.data(a_p), label='Neuron approx, filter=0.1')
            plt.plot(t, sim.data(inh_p), label='Inhib signal')
            plt.plot(t, ideal, label='Ideal output')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a_p)[-10:], 0, atol=.1, rtol=.01))

    def test_ensemble_to_neurons(self):
        name = 'ensemble_to_neurons'
        N = 30

        m = nengo.Model(name, seed=123)
        a = nengo.Ensemble(nengo.LIF(N), dimensions=1)
        b = nengo.Ensemble(nengo.LIF(N), dimensions=1)
        inn = nengo.Node(output=np.sin)
        inh = nengo.Node(piecewise({0: 0, 2.5: 1}))
        nengo.Connection(inn, a)
        nengo.Connection(inh, b)
        con = nengo.Connection(b, a.neurons, transform=[[-2.5]]*N)

        inn_p = nengo.Probe(inn, 'output')
        a_p = nengo.Probe(a, 'decoded_output', filter=0.1)
        b_p = nengo.Probe(b, 'decoded_output', filter=0.1)
        inh_p = nengo.Probe(inh, 'output')

        sim = self.Simulator(m)
        sim.run(5.0)
        t = sim.trange()
        ideal = np.sin(t)
        ideal[t >= 2.5] = 0

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data(inn_p), label='Input')
            plt.plot(t, sim.data(a_p), label='Neuron approx, pstc=0.1')
            plt.plot(
                t, sim.data(b_p), label='Neuron approx of inhib sig, pstc=0.1')
            plt.plot(t, sim.data(inh_p), label='Inhib signal')
            plt.plot(t, ideal, label='Ideal output')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a_p)[-10:], 0, atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data(b_p)[-10:], 1, atol=.1, rtol=.01))

    def test_neurons_to_ensemble(self):
        name = 'neurons_to_ensemble'
        N = 20

        m = nengo.Model(name, seed=123)
        a = nengo.Ensemble(nengo.LIF(N * 2), dimensions=2)
        b = nengo.Ensemble(nengo.LIF(N * 3), dimensions=3)
        c = nengo.Ensemble(nengo.LIF(N), dimensions=N*2)
        nengo.Connection(a.neurons, b, transform=-10 * np.ones((3, N*2)))
        nengo.Connection(a.neurons, c)

        a_p = nengo.Probe(a, 'decoded_output', filter=0.01)
        b_p = nengo.Probe(b, 'decoded_output', filter=0.01)
        c_p = nengo.Probe(c, 'decoded_output', filter=0.01)

        sim = self.Simulator(m)
        sim.run(5.0)
        t = sim.trange()

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data(a_p), label='A')
            plt.plot(t, sim.data(b_p), label='B')
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.all(sim.data(b_p)[-10:] < 0))

    def test_neurons_to_node(self):
        name = 'neurons_to_node'
        N = 30

        m = nengo.Model(name, seed=123)
        a = nengo.Ensemble(nengo.LIF(N), dimensions=1)
        out = nengo.Node(lambda t, x: x, dimensions=N)
        nengo.Connection(a.neurons, out, filter=None)

        a_spikes = nengo.Probe(a, 'spikes')
        out_p = nengo.Probe(out, 'output')

        sim = self.Simulator(m)
        sim.run(0.6)
        t = sim.trange()

        with Plotter(self.Simulator) as plt:
            ax = plt.subplot(111)
            try:
                from nengo.matplotlib import rasterplot
                rasterplot(t, sim.data(a_spikes), ax=ax)
                rasterplot(t, sim.data(out_p), ax=ax)
            except ImportError:
                pass
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(
            sim.data(a_spikes)[:-1], sim.data(out_p)[1:]))

    def test_neurons_to_neurons(self):
        name = 'neurons_to_neurons'
        N1, N2 = 30, 50

        m = nengo.Model(name, seed=123)
        a = nengo.Ensemble(nengo.LIF(N1), dimensions=1)
        b = nengo.Ensemble(nengo.LIF(N2), dimensions=1)
        inp = nengo.Node(output=1)
        nengo.Connection(inp, a)
        nengo.Connection(a.neurons, b.neurons, transform=-1 * np.ones((N2, N1)))

        inp_p = nengo.Probe(inp, 'output')
        a_p = nengo.Probe(a, 'decoded_output', filter=0.1)
        b_p = nengo.Probe(b, 'decoded_output', filter=0.1)

        sim = self.Simulator(m)
        sim.run(5.0)
        t = sim.trange()

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data(inp_p), label='Input')
            plt.plot(t, sim.data(a_p), label='A, represents input')
            plt.plot(t, sim.data(b_p), label='B, should be 0')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a_p)[-10:], 1, atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data(b_p)[-10:], 0, atol=.1, rtol=.01))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
