import numpy as np

import nengo
from nengo.core import Constant
from nengo.templates.circularconv import *
from nengo.tests.helpers import SimulatorTestCase, unittest, assert_allclose

import logging
logger = logging.getLogger(__name__)

class TestCircularConv(SimulatorTestCase):
    def test_helpers(self):
        """Test the circular convolution helper functions in Numpy"""
        from nengo.templates.circularconv import \
            _input_transform, _output_transform

        rng = np.random.RandomState(43232)

        dims = 1000
        invert_a = True
        invert_b = False
        x = rng.randn(dims)
        y = rng.randn(dims)
        z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

        dims2 = 2*dims - (2 if dims % 2 == 0 else 1)
        inA = _input_transform(dims, first=True, invert=invert_a)
        inB = _input_transform(dims, first=False, invert=invert_b)
        outC = _output_transform(dims)

        XY = np.zeros((dims2,2))
        XY += np.dot(inA, x)
        XY += np.dot(inB, y)

        C = XY[:,0] * XY[:,1]
        z1 = np.dot(outC, C)

        assert_allclose(self, logger, z0, z1)

    def test_direct(self):
        """Test direct-mode circular convolution"""

        dims = 1000

        rng = np.random.RandomState(8238)
        a = rng.randn(dims)
        b = rng.randn(dims)
        c = circconv(a, b)

        m = nengo.Model("")
        A = m.add(Constant(n=dims, value=a))
        B = m.add(Constant(n=dims, value=b))
        C = m.add(Signal(n=dims, name="C"))

        DirectCircularConvolution(m, A, B, C)

        sim = self.Simulator(m)
        sim.run_steps(10)
        c2 = sim.signals[C]

        # assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-5)
        assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-3)

    def _test_cconv(self, D, neurons_per_product):
        # D is dimensionality of semantic pointers

        m = nengo.Model('test_cconv_' + str(D))
        rng = np.random.RandomState(1234)

        A = m.add(Constant(D, value=rng.randn(D)))
        B = m.add(Constant(D, value=rng.randn(D)))

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
