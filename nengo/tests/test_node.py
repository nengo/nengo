import logging

import numpy as np

import nengo
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest
from nengo.tests.helpers import assert_allclose


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
            plt.plot(sim.trange(), sim.data(p), label='sin')
            plt.legend(loc='best')
            plt.savefig('test_node.test_simple.pdf')
            plt.close()

        sim_t = sim.trange()
        sim_in = sim.data(p).ravel()
        t = dt * np.arange(len(sim_t))
        self.assertTrue(np.allclose(sim_t, t))
        # 1-step delay
        self.assertTrue(np.allclose(sim_in[1:], np.sin(t[:-1])))

    def test_connected(self):
        dt = 0.001
        m = nengo.Model('test_connected', seed=123)

        input = nengo.Node(output=np.sin, label='input')
        output = nengo.Node(output=lambda t, x: np.square(x), dimensions=1,
                            label='output')
        nengo.Connection(input, output, filter=None)  # Direct connection
        p_in = nengo.Probe(input, 'output')
        p_out = nengo.Probe(output, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            t = sim.trange()
            plt.plot(t, sim.data(p_in), label='sin')
            plt.plot(t, sim.data(p_out), label='sin squared')
            plt.plot(t, np.sin(t), label='ideal sin')
            plt.plot(t, np.sin(t) ** 2, label='ideal squared')
            plt.legend(loc='best')
            plt.savefig('test_node.test_connected.pdf')
            plt.close()

        sim_t = sim.trange()
        sim_sin = sim.data(p_in).ravel()
        sim_sq = sim.data(p_out).ravel()
        t = dt * np.arange(len(sim_t))

        self.assertTrue(np.allclose(sim_t, t))
        # 1-step delay
        self.assertTrue(np.allclose(sim_sin[1:], np.sin(t[:-1])))
        self.assertTrue(np.allclose(sim_sq[1:], sim_sin[:-1] ** 2))

    def test_passthrough(self):
        dt = 0.001
        m = nengo.Model("test_passthrough", seed=0)

        in1 = nengo.Node(output=np.sin)
        in2 = nengo.Node(output=lambda t: t)
        passthrough = nengo.Node(dimensions=1)
        out = nengo.Node(output=lambda t, x: x, dimensions=1)

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
            plt.plot(sim.trange(),
                     sim.data(in1_p)+sim.data(in2_p),
                     label='in+in2')
            plt.plot(sim.trange()[:-2],
                     sim.data(out_p)[2:],
                     label='out')
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
            a = nengo.Node(output=lambda t, x: x+1, dimensions=1)
            b = nengo.Node(output=lambda t, x: x+1, dimensions=1)
            nengo.Connection(a, b, filter=None)
            nengo.Connection(b, a, filter=None)

            a_p = nengo.Probe(a, 'output')
            b_p = nengo.Probe(b, 'output')

        sim = self.Simulator(m, dt=dt)
        runtime = 0.5
        sim.run(runtime)

        assert_allclose(self, logger, sim.data(a_p), sim.data(b_p))


    def test_named_inputs(self):
        dt = 0.001
        m = nengo.Model("test_named_inputs", seed=0)

        ys = []
        zs = []
        def hello(t, y, z):
            ys.append(float(y))
            zs.append(float(z))
            return 0

        with m:
            a = nengo.Node(np.sin)  # -- still works
            b = nengo.Node(output=hello,
                           inputs={
                               'y': nengo.Vector(dimensions=1),
                               'z': nengo.Vector(dimensions=1),
                           }
                          )
            nengo.Connection(a, b.y, filter=None)
            nengo.Connection(a, b.z, filter=None)

        sim = self.Simulator(m, dt=dt)
        sim.run(0.01)
        assert all(y == z for y, z in zip(ys, zs))
        assert np.allclose(ys[9], 0.009, rtol=1e-4)



if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
