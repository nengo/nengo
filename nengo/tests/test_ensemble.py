import logging

import numpy as np

import nengo
from nengo.builder import ShapeMismatch
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestEnsembleEncoders(unittest.TestCase):
    def _test_encoders(self, n_neurons=10, n_dimensions=3, encoders=None):
        if encoders is None:
            encoders = np.random.randn(n_neurons, n_dimensions)
            orig_encoders = encoders.copy()

        args = {'name': 'A',
                'neurons': nengo.LIF(n_neurons),
                'dimensions': n_dimensions}

        model = nengo.Model('_test_encoders')
        ens = model.add(nengo.Ensemble(encoders=encoders, **args))
        sim = model.simulator(dt=0.001)
        self.assertTrue(np.allclose(orig_encoders, sim.model.get(ens).encoders))

    def test_encoders(self):
        self._test_encoders(n_dimensions=3)

    def test_encoders_one_dimension(self):
        self._test_encoders(n_dimensions=1)

    def test_encoders_high_dimension(self):
        self._test_encoders(n_dimensions=20)

    def test_encoders_wrong_shape(self):
        n_neurons, n_dimensions = 10, 3
        encoders = np.random.randn(n_dimensions)
        with self.assertRaises(ShapeMismatch):
            self._test_encoders(n_neurons, n_dimensions, encoders)

    def test_encoders_negative_neurons(self):
        with self.assertRaises(ValueError):
            self._test_encoders(n_neurons=-1, n_dimensions=1)

    def test_encoders_no_dimensions(self):
        with self.assertRaises(ValueError):
            self._test_encoders(n_neurons=1, n_dimensions=0)


class TestEnsemble(SimulatorTestCase):
    def _test_constant_scalar(self, nl):
        """A Network that represents a constant value."""
        N = 30
        val = 0.5

        m = nengo.Model('test_constant_scalar', seed=123)
        m.make_node('in', output=val)
        m.make_ensemble('A', nl(N), 1)
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
            plt.savefig('test_ensemble.%s.test_constant_scalar.pdf'
                        % nl.__name__)
            plt.close()

        self.assertTrue(np.allclose(sim.data('in').ravel(), val,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('A')[-10:], val,
                                    atol=.1, rtol=.01))

    def _test_constant_vector(self, nl):
        """A network that represents a constant 3D vector."""
        N = 30
        vals = [0.6, 0.1, -0.5]

        m = nengo.Model('test_constant_vector', seed=123)
        m.make_node('in', output=vals)
        m.make_ensemble('A', nl(N * len(vals)), len(vals))
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
            plt.savefig('test_ensemble.%s.test_constant_vector.pdf'
                        % nl.__name__)
            plt.close()

        self.assertTrue(np.allclose(sim.data('in')[-10:], vals,
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('A')[-10:], vals,
                                    atol=.1, rtol=.01))

    def _test_scalar(self, nl):
        """A network that represents sin(t)."""
        N = 30

        m = nengo.Model('test_scalar', seed=123)
        m.make_node('in', output=np.sin)
        m.make_ensemble('A', nl(N), 1)
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
            plt.savefig('test_ensemble.%s.test_scalar.pdf' % nl.__name__)
            plt.close()

        target = np.sin(np.arange(5000) / 1000.)
        target.shape = (-1, 1)
        logger.debug("[New API] input RMSE: %f", rmse(target, sim.data('in')))
        logger.debug("[New API] A RMSE: %f", rmse(target, sim.data('A')))
        self.assertTrue(rmse(target, sim.data('in')) < 0.001)
        self.assertTrue(rmse(target, sim.data('A')) < 0.1)

    def _test_vector(self, nl):
        """A network that represents sin(t), cos(t), arctan(t)."""
        N = 40

        m = nengo.Model('test_vector', seed=123)
        m.make_node('in', output=lambda t: [np.sin(t), np.cos(t), np.arctan(t)])
        m.make_ensemble('A', nl(N * 3), 3, radius=2)
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
            plt.savefig('test_ensemble.%s.test_vector.pdf' % nl.__name__)
            plt.close()

        target = np.vstack((np.sin(np.arange(5000) / 1000.),
                            np.cos(np.arange(5000) / 1000.),
                            np.arctan(np.arange(5000) / 1000.))).T
        logger.debug("In RMSE: %f", rmse(target, sim.data('in')))
        self.assertTrue(rmse(target, sim.data('in') < 0.001))
        self.assertTrue(rmse(target, sim.data('A')) < 0.1)

    def _test_product(self, nl):
        def product(x):
            return x[0] * x[1]

        m = nengo.Model('test_product', seed=124)

        N = 80
        m.make_node('sin', output=np.sin)
        m.make_node('-0.5', output=-.5)
        factors = m.make_ensemble(
            'factors', nl(2 * N), dimensions=2, radius=1.5)
        if nl != nengo.Direct:
            factors.encoders = np.tile([[1, 1],[-1, 1],[1, -1],[-1, -1]],
                                       (factors.n_neurons // 4, 1))
        m.make_ensemble('product', nl(N), dimensions=1)
        m.connect('sin', 'factors', transform=[[1], [0]])
        m.connect('-0.5', 'factors', transform=[[0], [1]])
        conn = m.connect('factors', 'product', function=product, filter=0.01)

        m.probe('sin', sample_every=.01)
        # m.probe(conn, sample_every=.01)  # FIXME
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
            #plt.plot(sim.data(conn))
            plt.plot(-.5 * np.sin(np.arange(0, 6, .01)))
            plt.savefig('test_ensemble.%s.test_prod.pdf' % nl.__name__)
            plt.close()

        self.assertTrue(rmse(sim.data('factors')[:, 0],
                             np.sin(np.arange(0, 6, .01))) < 0.1)
        self.assertTrue(rmse(sim.data('factors')[20:, 1], -0.5) < 0.1)

        def match(a, b):
            self.assertTrue(rmse(a, b) < 0.1)

        match(sim.data('product')[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))
        #match(sim.data(conn)[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))

    def test_direct(self):
        self._test_constant_scalar(nengo.Direct)
        self._test_constant_vector(nengo.Direct)
        self._test_scalar(nengo.Direct)
        self._test_vector(nengo.Direct)
        self._test_product(nengo.Direct)

    def test_lif(self):
        self._test_constant_scalar(nengo.LIF)
        self._test_constant_vector(nengo.LIF)
        self._test_scalar(nengo.LIF)
        self._test_vector(nengo.LIF)
        self._test_product(nengo.LIF)

    def test_lifrate(self):
        self._test_constant_scalar(nengo.LIFRate)
        self._test_constant_vector(nengo.LIFRate)
        self._test_scalar(nengo.LIFRate)
        self._test_vector(nengo.LIFRate)
        self._test_product(nengo.LIFRate)


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
