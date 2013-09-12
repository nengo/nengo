import numpy as np

import nengo
from nengo.core import Constant
from nengo.templates.circularconv import CircularConvolution
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestCircularConv(SimulatorTestCase):
    def _test_cconv(self, D, neurons_per_product):
        # D is dimensionality of semantic pointers

        m = nengo.Model('test_cconv_' + str(D))
        rng = np.random.RandomState(1234)

        A = m.add(Constant(rng.randn(D)))
        B = m.add(Constant(rng.randn(D)))

        CircularConvolution(m, A, B,
            neurons_per_product=neurons_per_product)

        sim = m.simulator(sim_class=self.Simulator)
        sim.run_steps(10)

        # -- XXX
        #    We're missing correctness testing, but we've already run
        #    smoke test of the code in CircularConvolution.

    def test_small(self):
        return self._test_cconv(D=4, neurons_per_product=3)

    def test_med(self):
        return self._test_cconv(D=50, neurons_per_product=128)

    def test_large(self):
        return self._test_cconv(D=512, neurons_per_product=128)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
