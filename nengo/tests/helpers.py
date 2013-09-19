try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest
import os
import os.path

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
        if tests is None:
            tests = unittest.TestSuite()
        tests.addTests(simulator_test_suite(simulator))
        return tests
    return load_tests


def simulator_test_suite(simulator):
    """This function returns a test suite with all SimulatorTestCases
    using the provided simulator.

    This function just returns a test suite, actually running those tests
    requires more effort. This function is provided for more advanced
    unit testing setups; for simple setups using `unittest`,
    use `nengo.tests.helpers.load_nengo_tests`.

    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    nengo_tests = loader.discover('nengo.tests')
    for test in all_testcases(nengo_tests):
        if hasattr(test, 'Simulator'):
            test.Simulator = simulator
            suite.addTest(test)
    return suite


class NengoTestLoader(unittest.TestLoader):
    """TestLoader that can access Nengo tests.

    This is for use in other packages that want to use Nengo test.
    Passing this loader to various unittest calls will allow
    for running individual Nengo tests.
    By default, if a name is passed, it tries the local module first.
    If that local module doesn't have the appropriate test,
    then `nengo.tests` is searched instead.

    A common use case for this loader is at the bottom of simulator
    tests files that access Nengo's tests.
    For example, to run either all or only specific Nengo simultor tests
    in a project called `my_simulator`, at the bottom of
    my_simulator/tests/test_sim.py, write:

    .. code-block:: python

       from nengo.tests.helpers import NengoTestLoader
       if __name__ == '__main__':
           unittest.main(testLoader=NengoTestLoader(MySimulator))

    """
    def __init__(self, simulator):
        self.simulator = simulator

    def loadTestsFromName(self, name, module=None):
        # First try using default behavior
        try:
            return unittest.TestLoader.loadTestsFromName(self, name, module)
        except AttributeError:
            # Test wasn't found, so try loading from nengo's tests instead
            name = 'nengo.tests.' + name
            suite = unittest.TestLoader.loadTestsFromName(self, name, None)
            for test in all_testcases(suite):
                if hasattr(test, 'Simulator'):
                    test.Simulator = self.simulator
            return suite


class Plotter(object):
    plot = int(os.getenv("NENGO_TEST_PLOT", 0))

    class Mock(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return Plotter.Mock()

        @classmethod
        def __getattr__(cls, name):
            if name in ('__file__', '__path__'):
                return '/dev/null'
            elif name[0] == name[0].upper():
                mockType = type(name, (), {})
                mockType.__module__ = __name__
                return mockType
            else:
                return Plotter.Mock()

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
            self.plt = Plotter.Mock()
        return self.plt

    def __exit__(self, type, value, traceback):
        if self.plot:
            self.plt.savefig = self.oldsavefig

    def savefig(self, fname, **kwargs):
        if self.plot:
            return self.oldsavefig(os.path.join(self.dirname, fname), **kwargs)


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


def assert_allclose(self, logger, a, b, atol=1e-8, rtol=1e-5):
    a, b = a.flatten(), b.flatten()
    mask = np.abs(a - b) > atol + rtol * np.abs(b)
    if mask.any():
        nz = mask.nonzero()[0]
        if len(nz) > 10: nz = nz[:10]
        logger.debug("allclose failed: %d offending entries\n%s" % (
                mask.sum(),
                "\n".join(
                    "(a=%14e, b=%14e, adiff=%14e, rdiff=%14e)" %
                    (a[m], b[m], np.abs(a[m]-b[m]), np.abs((a[m]-b[m])/b[m]))
                    for m in nz)))
        self.assertTrue(
            False, "allclose failed: %d offending entries" % (mask.sum()))

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def rmse(a, b):
    return rms(a - b)
