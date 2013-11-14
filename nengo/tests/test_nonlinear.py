import numpy as np

import nengo
from nengo.tests.helpers import unittest

import logging
logger = logging.getLogger(__name__)


class TestNonlinearBuiltins(unittest.TestCase):
    def test_lif_builtin(self):
        """Test that the dynamic model approximately matches the rates

        N.B.: Uses the built-in lif equations, NOT the passed simulator
        """
        rng = np.random.RandomState(85243)

        lif = nengo.LIF(10)
        lif.set_gain_bias(rng.uniform(80, 100, (10,)),
                          rng.uniform(-1, 1, (10,)))
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
    nengo.log(debug=True, path='log.txt')
    unittest.main()
