try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.old_api as nef

from helpers import simulates, SimulatesMetaclass


class TestModel(unittest.TestCase):
    __metaclass__ = SimulatesMetaclass

    @simulates
    def test_counters(self, simulator):
        params = dict(simulator=simulator, seed=123, dt=0.001)

        # Old API
        net = nef.Network('test_counters', **params)
        simtime_probe = net._raw_probe(net.model.simtime, dt_sample=.001)
        steps_probe = net._raw_probe(net.model.steps, dt_sample=.001)
        net.run(0.003)
        simtime_data = simtime_probe.get_data()
        steps_data = steps_probe.get_data()
        assert np.allclose(simtime_data.flatten(), [.001, .002, .003])
        assert np.allclose(steps_data.flatten(), [1, 2, 3])

        # New API
        m = nengo.Model('test_counters', **params)
        m.probe(m.simtime)
        m.probe(m.steps)
        m.run(0.003)
        assert np.allclose(m.data[m.simtime].flatten(), [.001, .002, .003])
        assert np.allclose(m.data[m.steps].flatten(), [1, 2, 3])


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
