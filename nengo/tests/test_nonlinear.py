try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

from nengo import LIF


class TestNonlinear(unittest.TestCase):
    def test_lif_rate_basic(self):
        """Test that the rate model approximately matches the
        dynamic model.

        """
        lif = LIF(10)
        J = np.arange(0, 5, .5)
        voltage = np.zeros(10)
        reftime = np.zeros(10)

        spikes = []
        spikes_ii = np.zeros(10)

        for ii in range(1000):
            lif.step_math0(.001, J + lif.bias, voltage, reftime, spikes_ii)
            spikes.append(spikes_ii.copy())

        sim_rates = np.sum(spikes, axis=0)
        math_rates = lif.rates(J)
        self.assertTrue(np.allclose(sim_rates, math_rates, atol=1, rtol=0.02))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
