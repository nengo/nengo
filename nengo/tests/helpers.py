try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest
import os
import os.path

try:
    # For Python >=3.3
    from unittest.mock import Mock
except:
    from mock import Mock

import numpy as np

import nengo.simulator


class Plotter(object):
    plot = int(os.getenv("NENGO_TEST_PLOT", 0))

    def __init__(self, simulator):
        self.dirname = simulator.__module__ + ".plots"

    def __enter__(self):
        if self.plot:
            import matplotlib.pyplot as plt
            self.plt = plt
            if not os.path.exists(self.dirname):
                os.mkdir(self.dirname)
            # Patch savefig
            self.oldsavefig = self.plt.savefig
            self.plt.savefig = self.savefig
        else:
            self.plt = Mock()
        return self.plt

    def __exit__(self, type, value, traceback):
        if self.plot:
            self.plt.savefig = self.oldsavefig

    def savefig(self, fname, **kwargs):
        if self.plot:
            return self.oldsavefig(os.path.join(self.dirname, fname), **kwargs)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


class SimulatorTestCase(unittest.TestCase):
    """
    Base class for TestCase classes that use self.Simulator(m)
    to produce a simulator for Model `m`.

    External projects that wish to run all SimulatorTestCase subclasses
    as unit tests using a different simulator can achieve that result by
    creating a load_tests function that overrides the simulator.

    .. code-block:: python

       def load_tests(loader, tests, pattern):
           def _flattentestcases(test_suite_or_case):
               try:
                   suite = iter(test_suite_or_case)
               except TypeError:
                   yield test_suite_or_case
               else:
                   for test in suite:
                       for subtest in _flattentestcases(test):
                           yield subtest
           suite = unittest.TestSuite()
           nengo_tests = loader.discover('nengo.tests')
           for test in _flattentestcases(nengo_tests):
               if hasattr(test, 'Simulator'):
                   test.Simulator = simulator
                   suite.addTest(test)
           return suite

    The `simulator` above should behave like nengo.simulator.Simulator;
    i.e., it takes in a model as its only argument.
    Additional arguments can be added by creating a function to do so.
    For example, in `nengo_ocl`:

    .. code-block:: python

       def simulator(*args, **kwargs):
           rval = sim_ocl.Simulator(ctx, *args, **kwargs)
           rval.alloc_all()
           rval.plan_all()
           return rval

    """
    Simulator = nengo.simulator.Simulator
