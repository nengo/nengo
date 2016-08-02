from __future__ import absolute_import

import inspect
import itertools
import logging
import os
import re
import sys
import threading
import time
import warnings

import numpy as np

from .compat import is_string, reraise
from .logging import CaptureLogHandler, console_formatter


class Mock(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    def __mul__(self, other):
        return 1.0

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()


class Recorder(object):
    def __init__(self, dirname, module_name, function_name):
        self.dirname = dirname
        self.module_name = module_name
        self.function_name = function_name

        if self.record and not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    @property
    def record(self):
        return self.dirname is not None

    @property
    def dirname(self):
        return self._dirname

    @dirname.setter
    def dirname(self, _dirname):
        if _dirname is not None and not os.path.exists(_dirname):
            os.makedirs(_dirname)

        self._dirname = _dirname

    def get_filename(self, ext=''):
        modparts = self.module_name.split('.')[1:]
        modparts.remove('tests')
        return "%s.%s.%s" % ('.'.join(modparts), self.function_name, ext)

    def get_filepath(self, ext=''):
        if not self.record:
            raise ValueError("Cannot construct path when not recording")
        return os.path.join(self.dirname, self.get_filename(ext=ext))

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()


class Plotter(Recorder):
    def __enter__(self):
        if self.record:
            import matplotlib.pyplot as plt
            self.plt = plt
        else:
            self.plt = Mock()
        return self.plt

    def __exit__(self, type, value, traceback):
        if self.record:
            if hasattr(self.plt, 'saveas') and self.plt.saveas is None:
                del self.plt.saveas
                self.plt.close('all')
                return

            if hasattr(self.plt, 'saveas'):
                self.filename = self.plt.saveas
                del self.plt.saveas
            else:
                self.filename = self.get_filename(ext='pdf')

            if len(self.plt.gcf().get_axes()) > 0:
                # tight_layout errors if no axes are present
                self.plt.tight_layout()

            savefig_kw = {'bbox_inches': 'tight'}
            if hasattr(self.plt, 'bbox_extra_artists'):
                savefig_kw['bbox_extra_artists'] = self.plt.bbox_extra_artists
                del self.plt.bbox_extra_artists

            self.plt.savefig(os.path.join(self.dirname, self.filename),
                             **savefig_kw)
            self.plt.close('all')


class Analytics(Recorder):
    DOC_KEY = 'documentation'

    def __init__(self, dirname, module_name, function_name):
        super(Analytics, self).__init__(dirname, module_name, function_name)

        self.data = {}
        self.doc = {}

    @staticmethod
    def load(path, module, function_name):
        modparts = module.split('.')
        modparts = modparts[1:]
        modparts.remove('tests')

        return np.load(os.path.join(path, "%s.%s.npz" % (
            '.'.join(modparts), function_name)))

    def __enter__(self):
        return self

    def add_data(self, name, data, doc=""):
        if name == self.DOC_KEY:
            raise ValueError("The name '{}' is reserved.".format(self.DOC_KEY))

        if self.record:
            self.data[name] = data
            if doc != "":
                self.doc[name] = doc

    def save_data(self):
        if len(self.data) == 0:
            return

        npz_data = dict(self.data)
        if len(self.doc) > 0:
            npz_data.update({self.DOC_KEY: self.doc})
        np.savez(self.get_filepath(ext='npz'), **npz_data)

    def __exit__(self, type, value, traceback):
        if self.record:
            self.save_data()


class Logger(Recorder):
    def __enter__(self):
        if self.record:
            self.handler = CaptureLogHandler()
            self.handler.setFormatter(console_formatter)
            self.logger = logging.getLogger()
            self.logger.addHandler(self.handler)
            self.old_level = self.logger.getEffectiveLevel()
            self.logger.setLevel(logging.INFO)
            self.logger.info("=== Test run at %s ===",
                             time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.logger = Mock()
        return self.logger

    def __exit__(self, type, value, traceback):
        if self.record:
            self.logger.removeHandler(self.handler)
            self.logger.setLevel(self.old_level)
            with open(self.get_filepath(ext='txt'), 'a') as fp:
                fp.write(self.handler.stream.getvalue())
            self.handler.close()
            del self.handler


class WarningCatcher(object):
    lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()
        try:
            self.catcher = warnings.catch_warnings(record=True)
            self.record = self.catcher.__enter__()
            warnings.simplefilter('always')
        except:
            self.lock.release()
            raise

    def __exit__(self, type, value, traceback):
        self.catcher.__exit__(type, value, traceback)
        self.lock.release()


class warns(WarningCatcher):
    def __init__(self, warning_type):
        import pytest
        self._pytest = pytest
        self.warning_type = warning_type

    def __exit__(self, type, value, traceback):
        if not any(r.category is self.warning_type for r in self.record):
            self._pytest.fail("DID NOT WARN")

        super(warns, self).__exit__(type, value, traceback)


def allclose(t, targets, signals,  # noqa: C901
             atol=1e-8, rtol=1e-5, buf=0, delay=0,
             plt=None, show=False, labels=None, individual_results=False):
    """Ensure all signal elements are within tolerances.

    Allows for delay, removing the beginning of the signal, and plotting.

    Parameters
    ----------
    t : array_like (T,)
        Simulation time for the points in `target` and `signals`.
    targets : array_like (T, 1) or (T, N)
        Reference signal or signals for error comparison.
    signals : array_like (T, N)
        Signals to be tested against the target signals.
    atol, rtol : float
        Absolute and relative tolerances.
    buf : float
        Length of time (in seconds) to remove from the beginnings of signals.
    delay : float
        Amount of delay (in seconds) to account for when doing comparisons.
    plt : matplotlib.pyplot or mock
        Pyplot interface for plotting the results, unless it's mocked out.
    show : bool
        Whether to show the plot immediately.
    labels : list of string, length N
        Labels of each signal to use when plotting.
    individual_results : bool
        If True, returns a separate `allclose` result for each signal.
    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    assert t.ndim == 1
    assert np.allclose(np.diff(t), dt)

    targets = np.asarray(targets)
    signals = np.asarray(signals)
    if targets.ndim == 1:
        targets = targets.reshape((-1, 1))
    if signals.ndim == 1:
        signals = signals.reshape((-1, 1))
    assert targets.ndim == 2 and signals.ndim == 2
    assert t.size == targets.shape[0]
    assert t.size == signals.shape[0]
    assert targets.shape[1] == 1 or targets.shape[1] == signals.shape[1]

    buf = int(np.round(buf / dt))
    delay = int(np.round(delay / dt))
    slice1 = slice(buf, len(t) - delay)
    slice2 = slice(buf + delay, None)

    if plt is not None:
        if labels is None:
            labels = [None] * len(signals)
        elif is_string(labels):
            labels = [labels]

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        def plot_target(ax, x, b=0, c='k'):
            bound = atol + rtol * np.abs(x)
            y = x - b
            ax.plot(t[slice2], y[slice1], c + ':')
            ax.plot(t[slice2], (y + bound)[slice1], c + '--')
            ax.plot(t[slice2], (y - bound)[slice1], c + '--')

        # signal plot
        ax = plt.subplot(2, 1, 1)
        for y, label in zip(signals.T, labels):
            ax.plot(t, y, label=label)

        if targets.shape[1] == 1:
            plot_target(ax, targets[:, 0], c='k')
        else:
            color_cycle = itertools.cycle(colors)
            for x in targets.T:
                plot_target(ax, x, c=next(color_cycle))

        ax.set_ylabel('signal')
        if labels[0] is not None:
            lgd = ax.legend(loc='upper left', bbox_to_anchor=(1., 1.))
            plt.bbox_extra_artists = (lgd,)

        ax = plt.subplot(2, 1, 2)
        if targets.shape[1] == 1:
            x = targets[:, 0]
            plot_target(ax, x, b=x, c='k')
            for y, label in zip(signals.T, labels):
                ax.plot(t[slice2], y[slice2] - x[slice1])
        else:
            color_cycle = itertools.cycle(colors)
            for x, y, label in zip(targets.T, signals.T, labels):
                c = next(color_cycle)
                plot_target(ax, x, b=x, c=c)
                ax.plot(t[slice2], y[slice2] - x[slice1], c, label=label)

        ax.set_xlabel('time')
        ax.set_ylabel('error')

        if show:
            plt.show()

    if individual_results:
        if targets.shape[1] == 1:
            return [np.allclose(y[slice2], targets[slice1, 0],
                                atol=atol, rtol=rtol) for y in signals.T]
        else:
            return [np.allclose(y[slice2], x[slice1], atol=atol, rtol=rtol)
                    for x, y in zip(targets.T, signals.T)]
    else:
        return np.allclose(signals[slice2, :], targets[slice1, :],
                           atol=atol, rtol=rtol)


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
            test = getattr(m, k)
            if callable(test) and re.search(pattern, k):
                args = inspect.getargspec(test).args
                if any(re.search(arg_pattern, arg) for arg in args):
                    tests['.'.join(['test'] + module + [k])] = test
            if k.startswith('pytest'):  # automatically load py.test hooks
                # TODO: different files with different implementations of the
                #   same pytest hook will break here!
                tests[k] = getattr(m, k)

    return tests


class ThreadedAssertion(object):
    """Performs assertions in parallel.

    Starts a number of threads, waits for each thread to execute some
    initialization code, and then executes assertions in each thread.
    """

    class AssertionWorker(threading.Thread):
        def __init__(self, parent, barriers, n):
            super(ThreadedAssertion.AssertionWorker, self).__init__()
            self.parent = parent
            self.barriers = barriers
            self.n = n
            self.assertion_result = None
            self.exc_info = (None, None, None)

        def run(self):
            self.parent.init_thread(self)

            self.barriers[self.n].set()
            for barrier in self.barriers:
                barrier.wait()

            try:
                self.parent.assert_thread(self)
                self.assertion_result = True
            except:
                self.assertion_result = False
                self.exc_info = sys.exc_info()
            finally:
                self.parent.finish_thread(self)

    def __init__(self, n_threads):
        barriers = [threading.Event() for _ in range(n_threads)]
        threads = [self.AssertionWorker(self, barriers, i)
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            if not t.assertion_result:
                reraise(*t.exc_info)

    def init_thread(self, worker):
        pass

    def assert_thread(self, worker):
        raise NotImplementedError()

    def finish_thread(self, worker):
        pass
