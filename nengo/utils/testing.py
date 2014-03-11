from __future__ import absolute_import

import os
import os.path

import numpy as np


class Plotter(object):
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

    def __init__(self, simulator, nl=None, plot=None):
        if plot is None:
            self.plot = int(os.getenv("NENGO_TEST_PLOT", 0))
        else:
            self.plot = plot
        self.nl = nl
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
            if self.nl is not None:
                fname = self.nl.__name__ + '.' + fname
            return self.oldsavefig(os.path.join(self.dirname, fname), **kwargs)


def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


def rmse(a, b):
    return rms(a - b)
