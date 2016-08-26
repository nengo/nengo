"""Utilities for progress tracking and display to the user."""

from __future__ import absolute_import, division

from datetime import timedelta
import importlib
import os
import sys
import time
import warnings

import numpy as np

from .stdlib import get_terminal_size
from .ipython import get_ipython
from ..exceptions import ValidationError
from ..rc import rc


class MemoryLeakWarning(UserWarning):
    pass


warnings.filterwarnings('once', category=MemoryLeakWarning)


def timestamp2timedelta(timestamp):
    if timestamp == -1:
        return "Unknown"
    return timedelta(seconds=np.ceil(timestamp))


def _load_class(name):
    mod_name, cls_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


class Progress(object):
    """Stores and tracks information about the progress of some process.

    This class is to be used as part of a ``with`` statement. Use ``step()`` to
    update the progress.

    Parameters
    ----------
    max_steps : int
        The total number of calculation steps of the process.

    Attributes
    ----------
    steps : int
        Number of completed steps.
    max_steps : int
        The total number of calculation steps of the process.
    start_time : float
        Time stamp of the time the process was started.
    end_time : float
        Time stamp of the time the process was finished or aborted.
    success : bool or None
        Whether the process finished successfully. ``None`` if the process
        did not finish yet.

    Examples
    --------

    >>> max_steps = 10
    >>> with Progress(max_steps) as progress:
    ...     for i in range(max_steps):
    ...         # do something
    ...         progress.step()

    """

    def __init__(self, max_steps):
        self.n_steps = 0
        self.max_steps = max_steps
        self.start_time = self.end_time = time.time()
        self.finished = False
        self.success = None

    @property
    def progress(self):
        """The current progress as a number from 0 to 1 (inclusive).

        Returns
        -------
        float
        """
        return min(1.0, self.n_steps / self.max_steps)

    def elapsed_seconds(self):
        """The number of seconds passed since entering the ``with`` statement.

        Returns
        -------
        float
        """
        if self.finished:
            return self.end_time - self.start_time
        else:
            return time.time() - self.start_time

    def eta(self):
        """The estimated number of seconds until the process is finished.

        Stands for estimated time of arrival (ETA).
        If no estimate is available -1 will be returned.

        Returns
        -------
        float
        """
        if self.progress > 0.:
            return (
                (1. - self.progress) * self.elapsed_seconds() / self.progress)
        else:
            return -1

    def __enter__(self):
        self.finished = False
        self.success = None
        self.n_steps = 0
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, dummy_exc_value, dummy_traceback):
        self.success = exc_type is None
        if self.success:
            self.n_steps = self.max_steps
        self.end_time = time.time()
        self.finished = True

    def step(self, n=1):
        """Advances the progress.

        Parameters
        ----------
        n : int
            Number of steps to advance the progress by.
        """
        self.n_steps += n


class ProgressBar(object):
    """Visualizes the progress of a process.

    This is an abstract base class that progress bar classes some inherit from.
    Progress bars should visually displaying the progress in some way.
    """

    supports_fast_ipynb_updates = False

    def __init__(self, task):
        self.task = task

    def update(self, progress):
        """Updates the displayed progress.

        Parameters
        ----------
        progress : :class:`Progress`
            The progress information to display.
        """
        raise NotImplementedError()


class NoProgressBar(ProgressBar):
    """A progress bar that does not display anything.

    Helpful in headless situations or when using Nengo as a library.
    """

    def __init__(self, task=None):
        super(NoProgressBar, self).__init__(task=task)

    def update(self, progress):
        pass


class TerminalProgressBar(ProgressBar):
    """A progress bar that is displayed as ASCII output on `stdout`."""

    def update(self, progress):
        if progress.finished:
            line = self._get_finished_line(progress)
        else:
            line = self._get_in_progress_line(progress)
        sys.stdout.write(line)
        sys.stdout.flush()

    def _get_in_progress_line(self, progress):
        line = "[{{}}] ETA: {eta}".format(
            eta=timestamp2timedelta(progress.eta()))
        percent_str = " {}... {}% ".format(
            self.task, int(100 * progress.progress))
        try:
            width, _ = get_terminal_size()
        except:
            width = 80
        progress_width = max(0, width - len(line))
        progress_str = (
            int(progress_width * progress.progress) * "#").ljust(
                progress_width)

        percent_pos = (len(progress_str) - len(percent_str)) // 2
        if percent_pos > 0:
            progress_str = (
                progress_str[:percent_pos] + percent_str +
                progress_str[percent_pos + len(percent_str):])

        return '\r' + line.format(progress_str)

    def _get_finished_line(self, progress):
        try:
            width, _ = get_terminal_size()
        except:
            width = 80
        line = "{} finished in {}.".format(
            self.task,
            timestamp2timedelta(progress.elapsed_seconds())).ljust(width)
        return '\r' + line + os.linesep


class WriteProgressToFile(ProgressBar):
    """Writes progress to a file.

    This is useful for remotely and intermittently monitoring progress.
    Note that this file will be overwritten on each update of the progress!

    Parameters
    ----------
    filename : str
        Path to the file to write the progress to.
    """

    def __init__(self, filename, task):
        self.filename = filename
        super(WriteProgressToFile, self).__init__(task)

    def update(self, progress):
        if progress.finished:
            text = "{} finished in {}.".format(
                self.task,
                timestamp2timedelta(progress.elapsed_seconds()))
        else:
            text = "{progress:.0f}%, ETA: {eta}".format(
                progress=100 * progress.progress,
                eta=timestamp2timedelta(progress.eta()))

        with open(self.filename, 'w') as f:
            f.write(text + os.linesep)


class AutoProgressBar(ProgressBar):
    """Suppresses the progress bar unless the ETA exceeds a threshold.

    Parameters
    ----------
    delegate : :class:`ProgressBar`
        The actual progress bar to display, if ETA is high enough.
    min_eta : float, optional
        The minimum ETA threshold for displaying the progress bar.
    """

    def __init__(self, delegate, min_eta=1.):
        self.delegate = delegate

        super(AutoProgressBar, self).__init__(delegate.task)

        self.min_eta = min_eta
        self._visible = False

    def update(self, progress):
        min_delay = progress.start_time + 0.1
        long_eta = progress.eta() > self.min_eta and min_delay < time.time()
        if self._visible:
            self.delegate.update(progress)
        elif long_eta or progress.finished:
            self._visible = True
            self.delegate.update(progress)

    @property
    def task(self):
        return self.delegate.task

    @task.setter
    def task(self, value):
        self.delegate.task = value


class ProgressUpdater(object):
    """Controls how often a progress bar is updated.

    This is an abstract base class that classes controlling the updates
    to a progress bar should inherit from.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar` instance
        The object to which updates are passed on.
    """

    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def update(self, progress):
        """Notify about changed progress and update progress bar if desired

        Parameters
        ----------
        progress : :class:`Progress`
            Changed progress information.
        """
        raise NotImplementedError()


class UpdateN(ProgressUpdater):
    """Updates a :class:`ProgressBar` every step, up to a maximum of ``n``.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar`
        The progress bar to relay the updates to.
    max_updates : int
        Maximum number of updates that will be relayed to the progress bar.

    Notes
    -----
    This is especially useful in the IPython 1.x notebook, since updating
    the notebook saves the output, which will create a large amount of memory
    and cause the notebook to crash.
    """

    def __init__(self, progress_bar, max_updates=100):
        super(UpdateN, self).__init__(progress_bar)
        self.max_updates = max_updates
        self.last_update_step = 0

    def update(self, progress):
        next_update_step = (self.last_update_step +
                            progress.max_steps / self.max_updates)
        if next_update_step < progress.n_steps or progress.finished:
            self.progress_bar.update(progress)
            self.last_update_step = progress.n_steps


class UpdateEveryN(ProgressUpdater):
    """Updates a :class:`ProgressBar` every ``n`` steps.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar`
        The progress bar to relay the updates to.
    every_n : int
        The number of steps in-between relayed updates.
    """

    def __init__(self, progress_bar, every_n=1000):
        super(UpdateEveryN, self).__init__(progress_bar)
        self.every_n = every_n
        self.next_update = every_n

    def update(self, progress):
        if self.next_update <= progress.n_steps or progress.finished:
            self.progress_bar.update(progress)
            assert self.every_n > 0
            self.next_update = progress.n_steps + self.every_n


class UpdateEveryT(ProgressUpdater):
    """Updates a :class:`ProgressBar` every ``t`` seconds.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar`
        The progress bar to relay the updates to.
    update_interval : float
        Number of seconds in-between relayed updates.
    """

    def __init__(self, progress_bar, every_t=0.05):
        super(UpdateEveryT, self).__init__(progress_bar)
        self.next_update = 0
        self.update_interval = every_t

    def update(self, progress):
        if self.next_update < time.time() or progress.finished:
            self.progress_bar.update(progress)
            self.next_update = time.time() + self.update_interval


class ProgressTracker(object):
    """Tracks the progress of some process with a progress bar.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps of the process.
    progress_bar : :class:`ProgressBar` or :class:`ProgressUpdater`
        The progress bar to display the progress.
    """
    def __init__(self, max_steps, progress_bar, task):
        self.progress = Progress(max_steps)
        self.progress_bar = wrap_with_progressupdater(
            task=task, progress_bar=progress_bar)

    def __enter__(self):
        self.progress.__enter__()
        self.progress_bar.update(self.progress)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress.__exit__(exc_type, exc_value, traceback)
        self.progress_bar.update(self.progress)

    def step(self, n=1):
        """Advance the progress and update the progress bar.

        Parameters
        ----------
        n : int
            Number of steps to advance the progress by.
        """
        self.progress.step(n)
        self.progress_bar.update(self.progress)


def get_default_progressbar(task):
    """The default progress bar to use depending on the execution environment.

    Returns
    -------
    :class:`ProgressBar`
    """
    try:
        pbar = rc.getboolean('progress', 'progress_bar')
        if pbar:
            return AutoProgressBar(TerminalProgressBar(task=task))
        else:
            return NoProgressBar()
    except ValueError:
        pass

    pbar = rc.get('progress', 'progress_bar')
    if pbar.lower() == 'auto':
        return AutoProgressBar(TerminalProgressBar(task=task))
    if pbar.lower() == 'none':
        return NoProgressBar()

    try:
        return _load_class(pbar)(task)
    except Exception as e:
        warnings.warn(str(e))
        return NoProgressBar()


def get_default_progressupdater(progress_bar):
    """The default progress updater.

    The default depends on the progress bar and execution environment.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar`
        The progress bar to obtain the default progess updater for.

    Returns
    -------
    :class:`ProgressUpdater`
    """
    updater = rc.get('progress', 'updater')

    if updater.lower() == 'auto':
        if get_ipython() is None or progress_bar.supports_fast_ipynb_updates:
            return UpdateEveryT
        else:
            return UpdateN
    else:
        try:
            return _load_class(updater)
        except Exception as e:
            warnings.warn(str(e))


def wrap_with_progressupdater(task, progress_bar=True):
    """Wraps a progress bar with the default progress updater.

    If it is already wrapped by an progress updater, then this does nothing.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar` or :class:`ProgressUpdater`
        The progress bar to wrap.

    Returns
    -------
    :class:`ProgressUpdater`
        The wrapped progress bar.
    """
    if progress_bar is False or progress_bar is None:
        return NoProgressBar()

    if progress_bar is True:
        progress_bar = get_default_progressbar(task)

    progress_bar.task = task

    if isinstance(progress_bar, ProgressUpdater):
        return progress_bar
    elif isinstance(progress_bar, ProgressBar):
        updater_class = get_default_progressupdater(progress_bar)
        return updater_class(progress_bar)
    else:
        raise ValidationError(
            "must be a boolean or instance of ProgressBar or ProgressUpdater "
            "(got %r)" % type(progress_bar).__name__,  attr='progress_bar')
