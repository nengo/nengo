
import numpy as np

import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest

import logging
logger = logging.getLogger(__name__)

class TestProbe(SimulatorTestCase):

    def test_long_name(self):
        m = nengo.Model('test_long_name', seed=123)
        m.make_ensemble(("This is an extremely long name that will test "
                         "if we can access sim data with long names"),
                        nengo.LIF(10), 1)
        m.probe("This is an extremely long name that will test "
                "if we can access sim data with long names")

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(0.01)

        self.assertIsNotNone(sim.data(
            "This is an extremely long name that will test "
            "if we can access sim data with long names"))

    def test_multirun(self):
        """Test probing the time on multiple runs"""
        rng = np.random.RandomState(2239)

        rtol = 1e-4 # a bit higher, since model.t accumulates error over time

        model = nengo.Model("Multi-run")
        sim = model.simulator(sim_class=self.Simulator)
        dt = sim.model.dt

        # t_stops = [0.123, 0.283, 0.821, 0.921]
        t_stops = dt * rng.randint(low=100, high=2000, size=10)

        t_sum = 0
        for ti in t_stops:
            sim.run(ti)
            sim_t = sim.data(model.t).flatten()
            t = dt * np.arange(1, len(sim_t)+1)
            self.assertTrue(np.allclose(sim_t, t, rtol=rtol))
            # assert_allclose(self, logger, sim_t, t, rtol=rtol)

            t_sum += ti
            print t_sum, sim_t[-1]
            self.assertTrue(np.allclose(sim_t[-1], t_sum, rtol=rtol))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
