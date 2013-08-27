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


def all_testcases(test_suite_or_case):
    """A test suite contains nested test suites and test cases.
    This generator function dives into test suites and only yields test cases.

    """
    try:
        suite = iter(test_suite_or_case)
    except TypeError:
        yield test_suite_or_case
    else:
        for test in suite:
            for subtest in all_testcases(test):
                yield subtest


def load_nengo_tests(simulator):
    """This function returns a function that unittest will call
    to run all SimulatorTestCases with the provided simulator.

    The function returned by this function must be
    called load_tests, and should be in a test_*.py module
    in the test folder for the simulator.

    For example, to run all nengo simultor tests in a project called
    `my_simulator`, create my_simulator/tests/test_sim.py containing

    .. code-block:: python

       from my_simulator import MySimulator
       from nengo.tests.helpers import load_nengo_tests
       load_tests = load_nengo_tests(MySimulator)

    """
    def load_tests(loader, tests, pattern, simulator=simulator):
        return simulator_test_suite(simulator, loader)
    return load_tests


def simulator_test_suite(simulator, loader=None):
    """This function returns a test suite with all SimulatorTestCases
    using the provided simulator.

    This function just returns a test suite, actually running those tests
    requires more effort. This function is provided for more advanced
    unit testing setups; for simple setups using `unittest`,
    use `nengo.tests.helpers.load_nengo_tests`.

    """
    if loader is None:
        loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    nengo_tests = loader.discover('nengo.tests')
    for test in all_testcases(nengo_tests):
        if hasattr(test, 'Simulator'):
            test.Simulator = simulator
            suite.addTest(test)
    return suite


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
    as unit tests using a different simulator can achieve that result
    with the helper functions
    `nengo.tests.helpers.load_nengo_tests` (recommended) or
    `nengo.tests.helpers.simulator_test_suite` (advanced).

    """
    Simulator = nengo.simulator.Simulator
