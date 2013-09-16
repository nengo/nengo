import numpy as np

import nengo
from nengo.objects import Node, PassthroughNode
import nengo.old_api as nef
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest


class TestNode(SimulatorTestCase):

    def test_simple(self):
        dt = 0.001
        m = nengo.Model('test_simple', seed=123)
        m.make_node('in', output=np.sin)
        m.probe('in')

        sim = m.simulator(dt=dt, sim_class=self.Simulator)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t), sim.data('in'), label='sin')
            plt.legend(loc='best')
            plt.savefig('test_node.test_simple.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(m.t).ravel(),
                                    np.arange(dt, runtime, dt)))
        # Two step delay!
        self.assertTrue(np.allclose(sim.data('in')[2:].ravel(),
                                    np.sin(np.arange(dt, runtime-dt*2, dt))))

    def test_connected(self):
        dt = 0.001
        m = nengo.Model('test_connected', seed=123)
        m.make_node('in', output=np.sin)
        # Not using make_node, as make_node connects time to node
        m.add(Node('out', output=np.square))
        m.connect('in', 'out', filter=None)  # Direct connection
        m.probe('in')
        m.probe('out')

        sim = m.simulator(dt=dt, sim_class=self.Simulator)
        runtime = 0.5
        sim.run(runtime)

        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t), sim.data('in'), label='sin')
            plt.plot(sim.data(m.t), sim.data('out'), label='sin squared')
            plt.legend(loc='best')
            plt.savefig('test_node.test_connected.pdf')
            plt.close()

        # One step delay!
        self.assertTrue(np.allclose(sim.data('in')[2:].ravel(),
                                    np.sin(np.arange(dt, runtime-dt*2, dt))))
        # One step delay!
        self.assertTrue(np.allclose(np.square(sim.data('in')[:-1]),
                                    sim.data('out')[1:]))
        
    def test_passthrough(self):
        dt = 0.001
        m = nengo.Model("test_passthrough", seed=0)
        
        m.make_node("in", output=np.sin)
        m.make_node("in2", output=lambda t: t)
        m.add(PassthroughNode("pass"))
        m.add(Node("out", output=lambda x: x))
        
        m.connect("in", "pass", filter=None)
        m.connect("in2", "pass", filter=None)
        m.connect("pass", "out", filter=None)
        
        m.probe("in")
        m.probe("in2")
        m.probe("out")
        
        sim = m.simulator(dt=dt, sim_class=self.Simulator)
        runtime = 0.5
        sim.run(runtime)
        
        with Plotter(self.Simulator) as plt:
            plt.plot(sim.data(m.t), sim.data('in')+sim.data('in2'), label='in+in2')
            plt.plot(sim.data(m.t)[:-2], sim.data('out')[2:], label='out')
            plt.legend(loc='best')
            plt.savefig('test_node.test_passthrough.pdf')
            plt.close()
        
        # Two step delay between first and second nonlinearity due to passthrough
        self.assertTrue(np.allclose(sim.data('in')[:-2]+sim.data('in2')[:-2],
                                    sim.data('out')[2:]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
