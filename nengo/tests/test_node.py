import numpy as np

import nengo
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest
from nengo.tests.helpers import assert_allclose

import logging
logger = logging.getLogger(__name__)

class TestNode(SimulatorTestCase):

    def test_simple(self):
        dt = 0.001
        m = nengo.Model('test_simple', seed=123)

        with m:
            input = nengo.Node(output=np.sin)
            p = nengo.Probe(input, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t_probe), sim.data(p), label='sin')
            plt.legend(loc='best')
            plt.savefig('test_node.test_simple.pdf')
            plt.close()

        sim_t = sim.data(m.t_probe).ravel()
        sim_in = sim.data(p).ravel()
        t = dt * np.arange(len(sim_t))
        self.assertTrue(np.allclose(sim_t, t))
        self.assertTrue(np.allclose(sim_in[1:], np.sin(t[:-1]))) # 1-step delay

    def test_connected(self):
        dt = 0.001
        m = nengo.Model('test_connected', seed=123)

        with m:
            input = nengo.Node(output=np.sin, label='input')
            output = nengo.Node(output=np.square, label='output')
            nengo.Connection(input, output, filter=None)  # Direct connection
            p_in = nengo.Probe(input, 'output')
            p_out = nengo.Probe(output, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t_probe), sim.data(p_in), label='sin')
            plt.plot(sim.data(m.t_probe), sim.data(p_out), label='sin squared')
            plt.plot(sim.data(m.t_probe), np.sin(sim.data(m.t_probe)), label='ideal sin')
            plt.plot(sim.data(m.t_probe), np.sin(sim.data(m.t_probe))**2, label='ideal squared')
            plt.legend(loc='best')
            plt.savefig('test_node.test_connected.pdf')
            plt.close()

        sim_t = sim.data(m.t_probe).ravel()
        sim_sin = sim.data(p_in).ravel()
        sim_sq = sim.data(p_out).ravel()
        t = dt * np.arange(len(sim_t))

        self.assertTrue(np.allclose(sim_t, t))
        self.assertTrue(np.allclose(sim_sin[1:], np.sin(t[:-1]))) # 1-step delay
        self.assertTrue(np.allclose(sim_sq[1:], sim_sin[:-1]**2)) # 1-step delay

    def test_passthrough(self):
        dt = 0.001
        m = nengo.Model("test_passthrough", seed=0)

        with m:
            in1 = nengo.Node(output=np.sin)
            in2 = nengo.Node(output=lambda t: t)
            passthrough = nengo.Node()
            out = nengo.Node(output=lambda x: x)

            nengo.Connection(in1, passthrough, filter=None)
            nengo.Connection(in2, passthrough, filter=None)
            nengo.Connection(passthrough, out, filter=None)

            in1_p = nengo.Probe(in1, 'output')
            in2_p = nengo.Probe(in2, 'output')
            out_p = nengo.Probe(out, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t_probe), sim.data(in1_p)+sim.data(in2_p), label='in+in2')
            plt.plot(sim.data(m.t_probe)[:-2], sim.data(out_p)[2:], label='out')
            plt.legend(loc='best')
            plt.savefig('test_node.test_passthrough.pdf')
            plt.close()

        # One-step delay between first and second nonlinearity
        sim_in = sim.data(in1_p)[:-1] + sim.data(in2_p)[:-1]
        sim_out = sim.data(out_p)[1:]
        self.assertTrue(np.allclose(sim_in, sim_out))

    def test_circular(self):
        dt = 0.001
        m = nengo.Model("test_circular", seed=0)

        with m:
            a = nengo.Node(output=lambda x:x+1)
            b = nengo.Node(output=lambda x:x+1)
            nengo.Connection(a, b, filter=None)
            nengo.Connection(b, a, filter=None)

            a_p = nengo.Probe(a, 'output')
            b_p = nengo.Probe(b, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        assert_allclose(self, logger, sim.data(a_p), sim.data(b_p))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
