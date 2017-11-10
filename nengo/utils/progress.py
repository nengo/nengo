"""Utilities for progress tracking and display to the user."""

from __future__ import absolute_import, division

from datetime import timedelta
import importlib
import os
import sys
import threading
import time
import warnings

import numpy as np

from .stdlib import get_terminal_size
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
    name_during : str, optional
        Short description of the task to be used while it is running.
    name_after : str, optional
        Short description of the task to be used after it has
        finished. Defaults to ``name_during``.

    Attributes
    ----------
    max_steps : int, optional
        The total number of calculation steps of the process, if known.
    name_after : str
        Name of the task to be used after it has finished.
    name_during : str
        Name of the task to be used while it is running.
    steps : int
        Number of completed steps.
    success : bool or None
        Whether the process finished successfully. ``None`` if the process
        did not finish yet.
    time_end : float
        Time stamp of the time the process was finished or aborted.
    time_start : float
        Time stamp of the time the process was started.

    Examples
    --------

    >>> max_steps = 10
    >>> with Progress(max_steps=max_steps) as progress:
    ...     for i in range(max_steps):
    ...         # do something
    ...         progress.step()

    """

    def __init__(self, name_during='', name_after=None, max_steps=None):
        if max_steps is not None and max_steps <= 0:
            raise ValidationError("must be at least 1 (got %d)"
                                  % (max_steps,), attr="max_steps")
        self.n_steps = 0
        self.max_steps = max_steps
        self.name_during = name_during
        if name_after is None:
            name_after = name_during
        self.name_after = name_after
        self.time_start = self.time_end = time.time()
        self.finished = False
        self.success = None

    @property
    def progress(self):
        """The current progress as a number from 0 to 1 (inclusive).

        Returns
        -------
        float
        """
        if self.max_steps is None:
            return 0.
        return min(1.0, self.n_steps / self.max_steps)

    def elapsed_seconds(self):
        """The number of seconds passed since entering the ``with`` statement.

        Returns
        -------
        float
        """
        if self.finished:
            return self.time_end - self.time_start
        else:
            return time.time() - self.time_start

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
        self.time_start = time.time()
        return self

    def __exit__(self, exc_type, dummy_exc_value, dummy_traceback):
        self.success = exc_type is None
        if self.success and self.max_steps is not None:
            self.n_steps = self.max_steps
        self.time_end = time.time()
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

    def update(self, progress):
        """Updates the displayed progress.

        Parameters
        ----------
        progress : :class:`Progress`
            The progress information to display.
        """
        raise NotImplementedError()

    def close(self):
        """Closes the progress bar.

        Indicates that not further updates will be made.
        """
        pass


class NoProgressBar(ProgressBar):
    """A progress bar that does not display anything.

    Helpful in headless situations or when using Nengo as a library.
    """

    def update(self, progress):
        pass


class TerminalProgressBar(ProgressBar):
    """A progress bar that is displayed as ASCII output on ``stdout``."""

    def update(self, progress):
        if progress.finished:
            line = self._get_finished_line(progress)
        elif progress.max_steps is None:
            line = self._get_unknown_progress_line(progress)
        else:
            line = self._get_in_progress_line(progress)
        sys.stdout.write(line)
        sys.stdout.flush()

    def _get_in_progress_line(self, progress):
        line = "[{{}}] ETA: {eta}".format(
            eta=timestamp2timedelta(progress.eta()))
        percent_str = " {}... {}% ".format(
            progress.name_during, int(100 * progress.progress))
        width, _ = get_terminal_size()
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

    def _get_unknown_progress_line(self, progress):
        """Generates a progress line with continuously moving marker.

        This is to indicate processing while not knowing how far along we
        progressed with the processing.
        """
        duration = progress.elapsed_seconds()
        line = "[{{}}] duration: {duration}".format(
            duration=timestamp2timedelta(duration))
        text = " {}... ".format(progress.name_during)
        width, _ = get_terminal_size()
        marker = '>>>>'
        progress_width = max(0, width - len(line) + 2)
        index_width = progress_width + len(marker)
        i = int(10. * duration) % (index_width + 1)
        progress_str = (' ' * i) + marker + (' ' * (index_width - i))
        progress_str = progress_str[len(marker):-len(marker)]
        text_pos = (len(progress_str) - len(text)) // 2
        progress_str = (
            progress_str[:text_pos] + text +
            progress_str[text_pos + len(text):])
        return '\r' + line.format(progress_str)

    def _get_finished_line(self, progress):
        width, _ = get_terminal_size()
        line = "{} finished in {}.".format(
            progress.name_after,
            timestamp2timedelta(progress.elapsed_seconds())).ljust(width)
        return '\r' + line

    def close(self):
        sys.stdout.write(os.linesep)
        sys.stdout.flush()


class WriteProgressToFile(ProgressBar):
    """Writes progress to a file.

    This is useful for remotely and intermittently monitoring progress.
    Note that this file will be overwritten on each update of the progress!

    Parameters
    ----------
    filename : str
        Path to the file to write the progress to.
    """

    def __init__(self, filename):
        self.filename = filename
        super(WriteProgressToFile, self).__init__()

    def update(self, progress):
        if progress.finished:
            text = "{} finished in {}.".format(
                self.progress.name_after,
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

        super(AutoProgressBar, self).__init__()

        self.min_eta = min_eta
        self._visible = False

    def update(self, progress):
        min_delay = progress.time_start + 0.1
        long_eta = (progress.elapsed_seconds() + progress.eta() > self.min_eta
                    and min_delay < time.time())
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

    def close(self):
        self.delegate.close()


class ProgressTracker(object):
    """Tracks the progress of some process with a progress bar.

    Parameters
    ----------
    progress_bar : :class:`ProgressBar` or bool or None
        The progress bar to display the progress (or True to use the default
        progress bar, False/None to disable progress bar).
    total_progress : int
        Maximum number of steps of the process.
    update_interval : float, optional
        Time to wait (in seconds) between updates to progress bar display.
    """
    def __init__(self, progress_bar, total_progress, update_interval=0.1):
        self.progress_bar = to_progressbar(progress_bar)
        self.total_progress = total_progress
        self.update_interval = update_interval
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self._closing = False
        self.sub_progress = None

    def next_stage(self, name_during='', name_after=None, max_steps=None):
        """Begin tracking progress of a new stage.

        Parameters
        ----------
        max_steps : int, optional
            The total number of calculation steps of the process.
        name_during : str, optional
            Short description of the task to be used while it is running.
        name_after : str, optional
            Short description of the task to be used after it has
            finished. Defaults to *name_during*.
        """
        if self.sub_progress is not None:
            self.total_progress.step()
        self.sub_progress = Progress(name_during, name_after, max_steps)
        return self.sub_progress

    def __enter__(self):
        self._closing = False
        self.total_progress.__enter__()
        if not isinstance(self.progress_bar, NoProgressBar):
            self.update_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._closing = True
        self.total_progress.__exit__(exc_type, exc_value, traceback)
        if not isinstance(self.progress_bar, NoProgressBar):
            self.update_thread.join()
        self.progress_bar.update(self.total_progress)
        self.progress_bar.close()

    def update_loop(self):
        """Update the progress bar display (will run in a separate thread)."""

        while not self._closing:
            if (self.sub_progress is not None and
                    not self.sub_progress.finished):
                self.progress_bar.update(self.sub_progress)
            else:
                self.progress_bar.update(self.total_progress)
            time.sleep(self.update_interval)


def get_default_progressbar():
    """The default progress bar to use depending on the execution environment.

    Returns
    -------
    :class:`ProgressBar`
    """
    try:
        pbar = rc.getboolean('progress', 'progress_bar')
        if pbar:
            return AutoProgressBar(TerminalProgressBar())
        else:
            return NoProgressBar()
    except ValueError:
        pass

    pbar = rc.get('progress', 'progress_bar')
    if pbar.lower() == 'auto':
        return AutoProgressBar(TerminalProgressBar())
    elif pbar.lower() == 'none':
        return NoProgressBar()

    try:
        return _load_class(pbar)()
    except Exception as e:
        warnings.warn(str(e))
        return NoProgressBar()


def to_progressbar(progress_bar):
    """Converts to a `.ProgressBar` instance.

    Parameters
    ----------
    progress_bar : None, bool, or `.ProgressBar`
        Object to be converted to a `.ProgressBar`.

    Returns
    -------
    ProgressBar
        Return *progress_bar* if it is already a progress bar, the default
        progress bar if *progress_bar* is *True*, and `NoProgressBar` if it is
        *None* or *False*.
    """
    if progress_bar is False or progress_bar is None:
        progress_bar = NoProgressBar()
    if progress_bar is True:
        progress_bar = get_default_progressbar()
    return progress_bar
