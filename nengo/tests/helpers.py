try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest
import os
import os.path
import sys

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

    There are many such test classes (TODO: add them to class registry).

    External projects that wish to run all SimulatorTestCase subclasses
    as unit tests using a different simulator can achieve that result by
    including the following sort of code among their own unit tests:

    .. code-block:: python

        globals().update(simulator_test_suite(simulator_allocator))

    The `simulator_allocator` should behave like nengo.simulator.Simulator.
    For example, the original test suite could be run like this:

    .. code-block:: python

        globals().update(simulator_test_suite(nengo.simulator.Simulator))

    """

    def Simulator(self, model):
        return nengo.simulator.Simulator(model)


def simulator_suite(simulator_allocator, name_suffix=''):
    done = set()
    rval = {}
    def define_new_test_case(base_class):
        if base_class in done or not hasattr(base_class, "Simulator"):
            return
        done.add(base_class)
        class MyTestCase(base_class):
            simulator_test_case_ignore = True
            def Simulator(self, model):
                return simulator_allocator(model)
                rval = sim_npy.Simulator(model)
                rval.alloc_all()
                rval.plan_all()
                return rval
        MyTestCase.__name__ = base_class.__name__
        rval[base_class.__name__ + name_suffix] = MyTestCase
    def search(thing):
        try:
            iter(thing)
        except TypeError:
            return define_new_test_case(type(thing))
        for obj in thing:
            search(obj)
    #N.B. -- each of the test modules is imported outside of the
    #        main package hierarchy like a main script, so
    #        all relative imports in test_modules are re-imported
    #        even if they have already been imported by the main
    #        environment.... in particular THIS FILE (and the
    #        SimulatorTestCase) base class is re-imported
    #        once for each test module.
    search(
        unittest.defaultTestLoader.discover(
            nengo.tests.__path__[0]))
    return rval

