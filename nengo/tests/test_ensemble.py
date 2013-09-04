import logging

import numpy as np

import nengo
from nengo.objects import Ensemble
import nengo.old_api as nef
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestEnsembleEncoders(unittest.TestCase):

    @staticmethod
    def _random_encoders(n_neurons, *dims):
        encoders = np.random.randn(*dims)
        normed = np.copy(encoders)
        if normed.shape == ():
            normed.shape = (1,)
        if normed.shape == (dims[0],):
            normed = np.tile(normed, (n_neurons, 1))
        norm = np.sum(normed * normed, axis=1)[:, np.newaxis]
        normed /= np.sqrt(norm)
        return encoders, normed

    def _test_encoders(self, n_dimensions):
        n_neurons = 10
        other_args = {'name': 'A',
                      'neurons': nengo.LIF(n_neurons),
                      'dimensions': n_dimensions}

        logger.debug("No encoders")
        self.assertIsNotNone(Ensemble(encoders=None, **other_args))

        logger.debug("One encoder that all neurons will share")
        encoders, normed = self._random_encoders(n_neurons, n_dimensions)
        ens = Ensemble(encoders=encoders, **other_args)
        self.assertTrue(np.allclose(normed, ens.encoders),
                        (normed, ens.encoders))

        logger.debug("All encoders")
        encoders, normed = self._random_encoders(
            n_neurons, n_neurons, n_dimensions)
        ens = Ensemble(encoders=encoders, **other_args)
        self.assertTrue(np.allclose(normed, ens.encoders),
                        (normed, ens.encoders))

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
        simulator = self.Simulator
        params = dict(simulator=simulator, seed=123)
        N = 30
        val = 0.5

        # old api
        net = nef.Network('test_constant_scalar', **params)
        net.make_input('in', value=[val])
        net.make('A', N, 1)
        net.connect('in', 'A')

        net.make_probe('in', dt_sample=0.001, pstc=0.0)
        net.make_probe('A', dt_sample=0.001, pstc=0.1)
        net = net.run(1, dt=0.001)

        in_data = net.model.data['in']
        a_data = net.model.data['A']

        with Plotter(simulator) as plt:
            t = net.model.data[net.model.t]
            plt.plot(t, in_data, label='Input')
            plt.plot(t, a_data, label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_constant_scalar-old.pdf')
            plt.close()

        self.assertTrue(np.allclose(in_data.ravel(), val, atol=.05, rtol=.05))
        self.assertTrue(np.allclose(a_data[-10:], val, atol=.05, rtol=.05))

        # New API
        m = nengo.Model('test_constant_scalar', **params)
        m.make_node('in', output=val)
        m.make_ensemble('A', nengo.LIF(N), 1)
        m.connect('in', 'A')

        m.probe('in')
        m.probe('A', filter=0.1)
        m = m.run(1, dt=0.001)

        with Plotter(simulator) as plt:
            t = m.data[m.t]
            plt.plot(t, m.data['in'], label='Input')
            plt.plot(t, m.data['A'], label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_constant_scalar-new.pdf')
            plt.close()

        self.assertTrue(np.allclose(m.data['in'].ravel(), val,
                                    atol=.05, rtol=.05))
        self.assertTrue(np.allclose(m.data['A'][-10:], val, atol=.05, rtol=.05))

    def test_constant_vector(self):
        """A network that represents a constant 3D vector."""
        simulator = self.Simulator
        params = dict(simulator=simulator, seed=123)
        N = 30
        vals = [0.6, 0.1, -0.5]

        # Old API
        net = nef.Network('test_constant_vector', **params)
        net.make_input('in', value=vals)
        net.make('A', N * len(vals), len(vals))
        net.connect('in', 'A', transform=np.eye(len(vals)))

        net.make_probe('in', dt_sample=0.001, pstc=0.0)
        net.make_probe('A', dt_sample=0.001, pstc=0.1)
        net = net.run(1, dt=0.001)

        in_data = net.model.data['in']
        a_data = net.model.data['A']

        with Plotter(simulator) as plt:
            t = net.model.data[net.model.t]
            plt.plot(t, in_data, label='Input')
            plt.plot(t, a_data, label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_ensemble.test_constant_vector-old.pdf')
            plt.close()

        self.assertTrue(np.allclose(in_data[-10:], vals, atol=.05, rtol=.05))
        self.assertTrue(np.allclose(a_data[-10:], vals, atol=.05, rtol=.05))

        # New API
        m = nengo.Model('test_constant_vector', **params)
        m.make_node('in', output=vals)
        m.make_ensemble('A', nengo.LIF(N * len(vals)), len(vals))
        m.connect('in', 'A')

        m.probe('in')
        m.probe('A', filter=0.1)
        m = m.run(1, dt=0.001)

        with Plotter(simulator) as plt:
            t = m.data[m.t]
            plt.plot(t, m.data['in'], label='Input')
            plt.plot(t, m.data['A'], label='Neuron approximation, pstc=0.1')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_ensemble.test_constant_vector-new.pdf')
            plt.close()

        self.assertTrue(np.allclose(m.data['in'][-10:], vals,
                                    atol=.05, rtol=.05))
        self.assertTrue(np.allclose(m.data['A'][-10:], vals,
                                    atol=.05, rtol=.05))

    def test_scalar(self):
        """A network that represents sin(t)."""
        simulator = self.Simulator
        params = dict(simulator=simulator, seed=123)
        N = 30
        target = np.sin(np.arange(4999) / 1000.)
        target.shape = (4999, 1)

        # Old API
        net = nef.Network('test_scalar', **params)
        net.make_input('in', value=np.sin)
        net.make('A', N, 1)
        net.connect('in', 'A')

        net.make_probe('in', dt_sample=0.001, pstc=0.0)
        net.make_probe('A', dt_sample=0.001, pstc=0.02)
        net = net.run(5, dt=0.001)

        in_data = net.model.data['in']
        a_data = net.model.data['A']

        with Plotter(simulator) as plt:
            t = net.model.data[net.model.t]
            plt.plot(t, in_data, label='Input')
            plt.plot(t, a_data, label='Neuron approximation, pstc=0.02')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_scalar-old.pdf')
            plt.close()

        logger.debug("[Old API] input RMSE: %f", rmse(target, in_data))
        logger.debug("[Old API] A RMSE: %f", rmse(target, a_data))
        self.assertTrue(rmse(target, in_data) < 0.001)
        self.assertTrue(rmse(target, a_data) < 0.1)

        # New API
        m = nengo.Model('test_scalar', **params)
        m.make_node('in', output=np.sin)
        m.make_ensemble('A', nengo.LIF(N), 1)
        m.connect('in', 'A')

        m.probe('in')
        m.probe('A', filter=0.02)
        m = m.run(5, dt=0.001)

        with Plotter(simulator) as plt:
            t = m.data[m.t]
            plt.plot(t, m.data['in'], label='Input')
            plt.plot(t, m.data['A'], label='Neuron approximation, pstc=0.02')
            plt.legend(loc=0)
            plt.savefig('test_ensemble.test_scalar-new.pdf')
            plt.close()

        logger.debug("[New API] input RMSE: %f", rmse(target, m.data['in']))
        logger.debug("[New API] A RMSE: %f", rmse(target, m.data['A']))
        self.assertTrue(rmse(target, m.data['in']) < 0.001)
        self.assertTrue(rmse(target, m.data['A']) < 0.1)

        # Check old/new API similarity
        logger.debug("Old/New API RMSE: %f", rmse(a_data, m.data['A']))
        self.assertTrue(rmse(a_data, m.data['A']) < 0.1)

    def test_vector(self):
        """A network that represents sin(t), cos(t), arctan(t)."""
        simulator = self.Simulator
        params = dict(simulator=simulator, seed=123, dt=0.001)
        N = 40
        target = np.vstack((np.sin(np.arange(4999) / 1000.),
                            np.cos(np.arange(4999) / 1000.),
                            np.arctan(np.arange(4999) / 1000.))).T

        # Old API
        net = nef.Network('test_vector', **params)
        net.make_input('sin', value=np.sin)
        net.make_input('cos', value=np.cos)
        net.make_input('arctan', value=np.arctan)
        net.make('A', N * 3, 3, radius=2)
        net.connect('sin', 'A', transform=[[1], [0], [0]])
        net.connect('cos', 'A', transform=[[0], [1], [0]])
        net.connect('arctan', 'A', transform=[[0], [0], [1]])

        net.make_probe('sin', dt_sample=0.001, pstc=0.0)
        net.make_probe('cos', dt_sample=0.001, pstc=0.0)
        net.make_probe('arctan', dt_sample=0.001, pstc=0.0)
        net.make_probe('A', dt_sample=0.001, pstc=0.02)
        net = net.run(5, dt=0.001)

        sin_data = net.model.data['sin']
        cos_data = net.model.data['cos']
        arctan_data = net.model.data['arctan']
        a_data = net.model.data['A']

        with Plotter(simulator) as plt:
            t = net.model.data[net.model.t]
            plt.plot(t, sin_data, label='sin')
            plt.plot(t, cos_data, label='cos')
            plt.plot(t, arctan_data, label='arctan')
            plt.plot(t, a_data, label='Neuron approximation, pstc=0.02')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_ensemble.test_vector-old.pdf')
            plt.close()

        logger.debug("[Old API] sin RMSE: %f", rmse(target[:,0], sin_data))
        logger.debug("[Old API] cos RMSE: %f", rmse(target[:,1], cos_data))
        logger.debug("[Old API] atan RMSE: %f", rmse(target[:,2], arctan_data))
        logger.debug("[Old API] A RMSE: %f", rmse(target, a_data))
        self.assertTrue(rmse(target, a_data) < 0.1)

        # New API
        m = nengo.Model('test_vector', **params)
        m.make_node('sin', output=np.sin)
        m.make_node('cos', output=np.cos)
        m.make_node('arctan', output=np.arctan)
        m.make_ensemble('A', nengo.LIF(N * 3), 3, radius=2)
        m.connect('sin', 'A', transform=[[1], [0], [0]])
        m.connect('cos', 'A', transform=[[0], [1], [0]])
        m.connect('arctan', 'A', transform=[[0], [0], [1]])

        m.probe('sin')
        m.probe('cos')
        m.probe('arctan')
        m.probe('A', filter=0.02)
        m = m.run(5, dt=0.001)

        with Plotter(simulator) as plt:
            t = m.data[m.t]
            plt.plot(t, m.data['sin'], label='sin')
            plt.plot(t, m.data['cos'], label='cos')
            plt.plot(t, m.data['arctan'], label='arctan')
            plt.plot(t, m.data['A'], label='Neuron approximation, pstc=0.02')
            plt.legend(loc=0, prop={'size': 10})
            plt.savefig('test_ensemble.test_vector-new.pdf')
            plt.close()

        # Not sure why, but this isn't working...
        logger.debug("[New API] sin RMSE: %f", rmse(target[:,0], m.data['sin']))
        logger.debug("[New API] cos RMSE: %f", rmse(target[:,1], m.data['cos']))
        logger.debug("[New API] atan RMSE: %f",
                     rmse(target[:,2], m.data['arctan']))
        logger.debug("[New API] A RMSE: %f", rmse(target, m.data['A']))
        self.assertTrue(rmse(target, m.data['A']) < 0.1)

        # Check old/new API similarity
        logger.debug("Old/New API RMSE: %f", rmse(a_data, m.data['A']))
        self.assertTrue(rmse(a_data, m.data['A']) < 0.1)

    def test_product(self):
        def product(x):
            return x[0]*x[1]

        N = 250
        seed = 123
        m = nengo.Model('Matrix Multiplication', seed=seed,
                          simulator=self.Simulator)

        m.make_node('sin', value=np.sin)
        m.make_node('neg', value=-.5)
        m.make_array('p', 2 * N, 1, dimensions=2, radius=1.5)
        m.make_array('D', N, 1, dimensions=1)
        m.connect('sin', 'p', transform=[[1], [0]])
        m.connect('neg', 'p', transform=[[0], [1]])
        prod = m.connect('p', 'D', func=product, pstc=0.01)

        net.make_probe(prod, dt_sample=.01, pstc=None)
        net.make_probe('p', dt_sample=.01, pstc=.01)
        net.make_probe('D', dt_sample=.01, pstc=.01)

        net = net.run(6, dt=0.001)

        data_p = net.model.data['p']
        data_d = net.model.data['D']
        data_r = net.model.data[prod]

        with Plotter(self.Simulator) as plt:
            plt.subplot(211);
            plt.plot(data_p)
            plt.plot(np.sin(np.arange(0, 6, .01)))
            plt.subplot(212);
            plt.plot(data_d)
            plt.plot(data_r)
            plt.plot(-.5 * np.sin(np.arange(0, 6, .01)))
            plt.savefig('test_old_api.test_prod.pdf')
            plt.close()

        self.assertTrue(np.allclose(data_p[:, 0], np.sin(np.arange(0, 6, .01)),
                                    atol=.1, rtol=.01))
        self.assertTrue(np.allclose(data_p[20:, 1], -0.5,
                                    atol=.1, rtol=.01))

        def match(a, b):
            self.assertTrue(np.allclose(a, b, .1, .1))

        match(data_d[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))
        match(data_r[:, 0], -0.5 * np.sin(np.arange(0, 6, .01)))



if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
