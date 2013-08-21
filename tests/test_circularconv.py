try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
from nengo.objects import Constant
from nengo.templates.circularconv import CircularConvolution

from helpers import Plotter, rmse, simulates, SimulatesMetaclass


class TestCircularConv(unittest.TestCase):
    def _test_cconv(self, simulator, D, neurons_per_product):
        # D is dimensionality of semantic pointers

        m = nengo.Model(.001)
        rng = np.random.RandomState(1234)

        A = m.add(Constant(D, value=rng.randn(D)))
        B = m.add(Constant(D, value=rng.randn(D)))

        CircularConvolution(m, A, B,
            neurons_per_product=neurons_per_product)

        sim = self.Simulator(m)
        sim.run_steps(10)

        raise nose.SkipTest()

    @simulates
    @unittest.skip("Not implemented yet")
    def test_small(self, simulator):
        return self._test_cconv(simulator, D=4, neurons_per_product=3)

    @simulates
    @unittest.skip("Not implemented yet")
    def test_med(self, simulator):
        return self._test_cconv(simulator, D=50, neurons_per_product=128)

    @simulates
    @unittest.skip("Not implemented yet")
    def test_large(self, simulator):
        return self._test_cconv(simulator, D=512, neurons_per_product=128)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
