import logging

import numpy as np

import nengo
from nengo.core import ShapeMismatch
from nengo.objects import Ensemble
import nengo.old_api as nef
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestEnsembleEncoders(unittest.TestCase):

    def _test_encoders(self, n_dimensions):
        n_neurons = 10
        other_args = {'name': 'A',
                      'neurons': nengo.LIF(n_neurons),
                      'dimensions': n_dimensions}

        logger.debug("No encoders")
        self.assertIsNotNone(Ensemble(encoders=None, **other_args))

        logger.debug("All encoders")
        encoders = np.random.randn(n_neurons, n_dimensions)
        ens = Ensemble(encoders=encoders, **other_args)
        self.assertTrue(np.allclose(encoders, ens.encoders))

        # A previously accepted example that is no longer accepted
        logger.debug("One encoder that all neurons will share")
        with self.assertRaises(ShapeMismatch):
            encoders = np.random.randn(n_dimensions)
            ens = Ensemble(encoders=encoders, **other_args)


    def test_encoders(self):
        self._test_encoders(2)
        self._test_encoders(10)

    def test_encoders_one_dimension(self):
        self._test_encoders(1)

    def test_encoders_high_dimension(self):
        self._test_encoders(20)


class TestEnsemble(SimulatorTestCase):
    def test_constant_scalar(self):
        """A Network that represents a constant value."""
        N = 30
        val = 0.5

        m = nengo.Model('test_constant_scalar', seed=123)
        m.make_node('in', output=val)
        m.make_ensemble('A', nengo.LIF(N), 1)
        m.connect('in', 'A')
        m.probe('in')
        m.probe('A', filter=0.1)

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(1.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_constant_scalar.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('in').ravel(), val,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('A')[-10:], val,
                                    atol=.1, rtol=.01))

    def test_constant_vector(self):
        """A network that represents a constant 3D vector."""
        N = 30
        vals = [0.6, 0.1, -0.5]

        m = nengo.Model('test_constant_vector', seed=123)
        m.make_node('in', output=vals)
        m.make_ensemble('A', nengo.LIF(N * len(vals)), len(vals))
        m.connect('in', 'A')
        m.probe('in')
        m.probe('A', filter=0.1)

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(1.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_ensemble.test_constant_vector.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('in')[-10:], vals,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('A')[-10:], vals,
                                    atol=.1, rtol=.01))

    def test_scalar(self):
        """A network that represents sin(t)."""
        N = 30

        m = nengo.Model('test_scalar', seed=123)
        m.make_node('in', output=np.sin)
        m.make_ensemble('A', nengo.LIF(N), 1)
        m.connect('in', 'A')
        m.probe('in')
        m.probe('A', filter=0.02)

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(5.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approximation, pstc=0.02')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_scalar.pdf')
            plt.close()

        target = np.sin(np.arange(4999) / 1000.)
        target.shape = (4999, 1)
        logger.debug("[New API] input RMSE: %f", rmse(target, sim.data('in')))
        logger.debug("[New API] A RMSE: %f", rmse(target, sim.data('A')))
        self.assertTrue(rmse(target, sim.data('in')) < 0.001)
        self.assertTrue(rmse(target, sim.data('A')) < 0.1)

    def test_vector(self):
        """A network that represents sin(t), cos(t), arctan(t)."""
        N = 40

        m = nengo.Model('test_vector', seed=123)
        m.make_node('in', output=lambda t: [np.sin(t), np.cos(t), np.arctan(t)])
        m.make_ensemble('A', nengo.LIF(N * 3), 3, radius=2)
        m.connect('in', 'A')
        m.probe('in')
        m.probe('A', filter=0.02)

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(5)

        with Plotter(self.Simulator) as plt:
            t = sim.data(m.t)
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approximation, pstc=0.02')
            plt.legend(loc='best', prop={'size': 10})
            plt.savefig('test_ensemble.test_vector.pdf')
            plt.close()

        target = np.vstack((np.sin(np.arange(4999) / 1000.),
                            np.cos(np.arange(4999) / 1000.),
                            np.arctan(np.arange(4999) / 1000.))).T
        logger.debug("In RMSE: %f", rmse(target, sim.data('in')))
        self.assertTrue(rmse(target, sim.data('in') < 0.001))
        self.assertTrue(rmse(target, sim.data('A')) < 0.1)

    def test_product(self):
        def product(x):
            return x[0] * x[1]

        m = nengo.Model('test_product', seed=124)

        N = 40
        m.make_node('sin', output=np.sin)
        m.make_node('-0.5', output=-.5)
        m.make_ensemble('factors', nengo.LIF(2 * N), dimensions=2, radius=1.5)
        m.make_ensemble('product', nengo.LIF(N), dimensions=1)
        m.connect('sin', 'factors', transform=[[1], [0]])
        m.connect('-0.5', 'factors', transform=[[0], [1]])
        conn = m.connect('factors', 'product', function=product, filter=0.01)

        m.probe('sin', sample_every=.01)
        m.probe(conn, sample_every=.01)
        m.probe('factors', sample_every=.01, filter=.01)
        m.probe('product', sample_every=.01, filter=.01)

        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(6)

        with Plotter(self.Simulator) as plt:
            plt.subplot(211)
            plt.plot(sim.data('factors'))
            plt.plot(np.sin(np.arange(0, 6, .01)))
            plt.plot(sim.data('sin'))
            plt.subplot(212)
            plt.plot(sim.data('product'))
            plt.plot(sim.data(conn))
            plt.plot(-.5 * np.sin(np.arange(0, 6, .01)))
            plt.savefig('test_ensemble.test_prod.pdf')
            plt.close()

        self.assertTrue(np.allclose(
            sim.data('factors')[:, 0], np.sin(np.arange(0, 6, .01)),
            atol=.1, rtol=.01))
        self.assertTrue(np.allclose(
            sim.data('factors')[20:, 1], -0.5,
            atol=.1, rtol=.01))

        def match(a, b):
            self.assertTrue(np.allclose(a, b, .1, .1))

        match(sim.data('product')[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))
        match(sim.data(conn)[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))



if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
