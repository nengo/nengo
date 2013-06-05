import unittest
import nose
import numpy as np
from nengo.simulator_objects import SimModel
from nengo.simulator import Simulator
from nengo.templates.circularconv import CircularConvolution



class TestCircularConv(unittest.TestCase):
    def Simulator(self, m):
        return Simulator(m)

    def test_small(self, D=4, neurons_per_product=3):
        # D is dimensionality of semantic pointers

        m = SimModel(.001)
        rng = np.random.RandomState(1234)

        A = m.signal(D, value=rng.randn(D))
        B = m.signal(D, value=rng.randn(D))

        CircularConvolution(m, A, B,
            neurons_per_product=neurons_per_product)

        sim = self.Simulator(m)
        sim.run_steps(10)

        raise nose.SkipTest()

    def test_med(self):
        return self.test_small(D=50, neurons_per_product=128)

    def test_large(self):
        return self.test_small(D=512, neurons_per_product=128)


