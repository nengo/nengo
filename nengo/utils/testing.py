from __future__ import absolute_import

import inspect
import numpy as np
import os
import re


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


def allclose(t, target, signals, plotter=None, filename=None, labels=None,
             atol=1e-8, rtol=1e-5, buf=0, delay=0):
    """Perform an allclose check between two signals, with the potential to
    buffer both ends of the signal, account for a delay, and make a plot."""

    signals = np.asarray(signals)
    vector_in = signals.ndim < 2
    if vector_in:
        signals.shape = (1, -1)

    slice1 = slice(buf, -buf - delay)
    slice2 = slice(buf + delay, -buf)

    if plotter is not None:
        with plotter as plt:
            if labels is None:
                labels = [None] * len(signals)
            elif isinstance(labels, str):
                labels = [labels]

            bound = atol + rtol * np.abs(target)

            # signal plot
            ax = plt.subplot(2, 1, 1)
            ax.plot(t, target, 'k:')
            for signal, label in zip(signals, labels):
                ax.plot(t, signal, label=label)
            ax.plot(t[slice2], (target + bound)[slice1], 'k--')
            ax.plot(t[slice2], (target - bound)[slice1], 'k--')
            ax.set_ylabel('signal')
            legend = (ax.legend(loc=2, bbox_to_anchor=(1., 1.))
                      if labels[0] is not None else None)

            # error plot
            errors = np.array([signal[slice2] - target[slice1]
                               for signal in signals])
            ymax = 1.1 * max(np.abs(errors).max(), bound.max())

            ax = plt.subplot(2, 1, 2)
            ax.plot(t[slice2], np.zeros_like(t[slice2]), 'k:')
            for error, label in zip(errors, labels):
                plt.plot(t[slice2], error, label=label)
            ax.plot(t[slice2], bound[slice1], 'k--')
            ax.plot(t[slice2], -bound[slice1], 'k--')
            ax.set_ylim((-ymax, ymax))
            ax.set_xlabel('time')
            ax.set_ylabel('error')

            if filename is not None:
                plt.savefig(filename, bbox_inches='tight',
                            bbox_extra_artists=(legend,) if legend else ())
            else:
                plt.show()
            plt.close()

    close = [np.allclose(signal[slice2], target[slice1], atol=atol, rtol=rtol)
             for signal in signals]
    return close[0] if vector_in else close


def find_modules(root_path, prefix=[], pattern='^test_.*\\.py$'):
    """Find matching modules (files) in all subdirectories of a given path.

    Parameters
    ----------
    root_path : string
        The path of the directory in which to begin the search.
    prefix : string or list, optional
        A string or list of strings to append to each returned modules list.
    pattern : string, optional
        A regex pattern for matching individual file names. Defaults to
        looking for all testing scripts.

    Returns
    -------
    modules : list
        A list of modules. Each item in the list is a list of strings
        containing the module path.
    """
    if isinstance(prefix, str):
        prefix = [prefix]
    elif not isinstance(prefix, list):
        raise TypeError("Invalid prefix type '%s'" % type(prefix).__name__)

    modules = []
    for name in os.listdir(root_path):
        path = os.path.join(root_path, name)
        if os.path.isdir(path):
            modules.extend(
                find_modules(path, prefix=prefix + [name], pattern=pattern))
        elif re.search(pattern, name):
            name, ext = os.path.splitext(name)
            modules.append(prefix + [name])

    return modules


def load_functions(modules, pattern='^test_', arg_pattern='^Simulator$'):
    """Load matching functions from a list of modules.

    Parameters
    ----------
    modules : list
        A list of testing modules to load, generated by `find_testmodules`.
    pattern : string
        A regex pattern for matching the function names. Defaults to looking
        for all testing functions.
    arg_pattern : string
        A regex pattern for matching the argument names. At least one argument
        must match the pattern. Defaults to selecting tests with
        a 'Simulator' argument.

    Returns
    -------
    tests : dict
        A dictionary of test functions, where the key is composed of the
        module and function name, and the value is the function handle.

    Examples
    --------
    To load all Nengo tests and add them to the current namespace, do

        nengo_dir = os.path.dirname(nengo.__file__)
        modules = find_modules(nengo_dir, prefix='nengo')
        tests = load_functions(modules)
        locals().update(tests)
    """
    tests = {}
    for module in modules:
        m = __import__('.'.join(module), globals(), locals(), ['*'])
        for k in dir(m):
            if re.search(pattern, k):
                test = getattr(m, k)
                args = inspect.getargspec(test).args
                if any(re.search(arg_pattern, arg) for arg in args):
                    tests['.'.join(['test'] + module + [k])] = test
            if k.startswith('pytest'):
                # TODO: different files with different implementations of the
                #   same pytest hook will break here!
                tests[k] = getattr(m, k)

    return tests
