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

simulators = (nengo.simulator.Simulator,)


class Plotter(object):
    plot = int(os.getenv("NENGO_TEST_PLOT", 0))

    def __init__(self, simulator):
        self.dirname = simulator.__module__

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


# Inspired by Xavier Decoret's post: http://stackoverflow.com/a/4455312
def simulates(f, simulators=simulators):
    setattr(f, "simulators", getattr(f, "simulators", ()) + simulators)
    return f


class SimulatesMetaclass(type):
    def __new__(meta, name, bases, dict):
        for methodname, method in dict.items():
            if hasattr(method, "simulators"):
                dict.pop(methodname)
                simulators = getattr(method, "simulators")
                delattr(method, "simulators")
                for simulator in simulators:
                    def method_for_sim(self, method=method, sim=simulator):
                        method(self, sim)
                    methodname_for_sim = methodname + "(" + str(simulator) + ")"
                    dict[methodname_for_sim] = method_for_sim
        return type.__new__(meta, name, bases, dict)
