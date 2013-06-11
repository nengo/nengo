import unittest

from nengo.nonlinear import Direct
from nengo.simulator_objects import SimModel


class TestSet(unittest.TestCase):
    """SimModel is made up of a bunch of sets.
    This test suite makes sure that the simulator objects
    interact with these sets as we expect them to.

    """

    def test_nonlinear(self):
        sm = SimModel()
        f = lambda x: x
        nl1 = Direct(1, 1, f)
        nl2 = Direct(2, 2, f)
        sm.nonlinearity(nl1)
        self.assertEqual(len(sm.nonlinearities), 1)
        sm.nonlinearity(nl1)
        self.assertEqual(len(sm.nonlinearities), 1)  # Already in there
        sm.nonlinearity(nl2)
        self.assertEqual(len(sm.nonlinearities), 2)
        sm.nonlinearity(Direct(1, 1, f))  # Same params as nl1, but different
        self.assertEqual(len(sm.nonlinearities), 3)
