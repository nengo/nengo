from __future__ import absolute_import

import inspect
import os
import re
import sys
import time
import warnings

import numpy as np
import pytest

from nengo.utils.compat import is_string


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

    def __init__(self, simulator, module, function, nl=None, plot=None):
        if plot is None:
            self.plot = int(os.getenv("NENGO_TEST_PLOT", 0))
        else:
            self.plot = plot

        self.dirname = "%s.plots" % simulator.__module__
        if nl is not None:
            self.dirname = os.path.join(self.dirname, nl.__name__)

        modparts = module.__name__.split('.')
        modparts = modparts[1:]
        modparts.remove('tests')
        self.filename = "%s.%s.pdf" % ('.'.join(modparts), function.__name__)

    def __enter__(self):
        if self.plot:
            import matplotlib.pyplot as plt
            self.plt = plt
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
        else:
            self.plt = Plotter.Mock()
        return self.plt

    def __exit__(self, type, value, traceback):
        if self.plot:
            if hasattr(self.plt, 'saveas'):
                if self.plt.saveas is None:
                    del self.plt.saveas
                    self.plt.close('all')
                    return
                self.filename = self.plt.saveas
                del self.plt.saveas
            self.plt.tight_layout()
            self.plt.savefig(os.path.join(self.dirname, self.filename))
            self.plt.close('all')


class Timer(object):
    """A context manager for timing a block of code.

    Attributes
    ----------
    duration : float
        The difference between the start and end time (in seconds).
        Usually this is what you care about.
    start : float
        The time at which the timer started (in seconds).
    end : float
        The time at which the timer ended (in seconds).

    Example
    -------
    >>> import time
    >>> with Timer() as t:
    ...    time.sleep(1)
    >>> assert t.duration >= 1
    """

    TIMER = time.clock if sys.platform == "win32" else time.time

    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None

    def __enter__(self):
        self.start = Timer.TIMER()
        return self

    def __exit__(self, type, value, traceback):
        self.end = Timer.TIMER()
        self.duration = self.end - self.start


class WarningCatcher(object):
    def __enter__(self):
        self.catcher = warnings.catch_warnings(record=True)
        self.record = self.catcher.__enter__()

    def __exit__(self, type, value, traceback):
        self.catcher.__exit__(type, value, traceback)


class warns(WarningCatcher):
    def __init__(self, warning_type):
        self.warning_type = warning_type

    def __exit__(self, type, value, traceback):
        if not any(r.category is self.warning_type for r in self.record):
            pytest.fail("DID NOT RAISE")

        super(warns, self).__exit__(type, value, traceback)


def allclose(t, target, signals, plt=None, show=False,  # noqa:C901
             labels=None, atol=1e-8, rtol=1e-5, buf=0, delay=0):
    """Perform an allclose check between two signals, with the potential to
    buffer both ends of the signal, account for a delay, and make a plot."""
    target = target.squeeze()
    if target.ndim > 1:
        raise ValueError("Can only pass one target signal")

    signals = np.asarray(signals)
    vector_in = signals.ndim < 2
    if signals.ndim > 2:
        raise ValueError("'signals' cannot have more than two dimensions")
    elif vector_in:
        signals.shape = (1, -1)

    nt = t.size
    if signals.shape[1] != nt:
        raise ValueError("'signals' must have time along the second axis")

    slice1 = slice(buf, nt - buf - delay)
    slice2 = slice(buf + delay, nt - buf)

    if plt is not None:
        if labels is None:
            labels = [None] * len(signals)
        elif is_string(labels):
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
        if labels[0] is not None:
            ax.legend(loc=2, bbox_to_anchor=(1., 1.))

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

        if show:
            plt.show()

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
    if is_string(prefix):
        prefix = [prefix]
    elif not isinstance(prefix, list):
        raise TypeError("Invalid prefix type '%s'" % type(prefix).__name__)

    modules = []
    for path, dirs, files in os.walk(root_path):
        base = prefix + os.path.relpath(path, root_path).split(os.sep)
        for filename in files:
            if re.search(pattern, filename):
                name, ext = os.path.splitext(filename)
                modules.append(base + [name])

    return modules


def load_functions(modules, pattern='^test_', arg_pattern='^Simulator$'):
    """Load matching functions from a list of modules.

    Parameters
    ----------
    modules : list
        A list of testing modules to load, generated by `find_testmodules`.
    pattern : string, optional
        A regex pattern for matching the function names. Defaults to looking
        for all testing functions.
    arg_pattern : string, optional
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

    Notes
    -----
    - This was created to load py.test tests. Therefore, this function also
      loads functions that start with `pytest`, since these functions act as
      hooks into py.test.
    - TODO: currently, all `pytest` functions are loaded into the same
      namespace, which means that if two different files imlement the same
      py.test hook, only the latter of these will be respected.
    - TODO: py.test also allows test functions to be implemented in classes;
      these tests cannot currently be loaded by this function.
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
            if k.startswith('pytest'):  # automatically load py.test hooks
                # TODO: different files with different implementations of the
                #   same pytest hook will break here!
                tests[k] = getattr(m, k)

    return tests
