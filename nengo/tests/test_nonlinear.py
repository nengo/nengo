import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.builder import Signal, Constant
from nengo.builder import DotInc, ProdUpdate
from nengo.nonlinearities import Direct, LIF, LIFRate
from nengo.tests.helpers import SimulatorTestCase, unittest, rms

import logging
logger = logging.getLogger(__name__)


class TestNonlinearBuiltins(unittest.TestCase):
    def test_lif_builtin(self):
        """Test that the dynamic model approximately matches the rates

        N.B.: Uses the built-in lif equations, NOT the passed simulator
        """
        lif = LIF(10)
        lif.set_gain_bias(np.random.uniform(80, 100, (10,)),
                          np.random.uniform(-1, 1, (10,)))
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
