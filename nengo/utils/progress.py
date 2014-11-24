"""Utilities for progress tracking and display to the user."""

from __future__ import absolute_import, division

from datetime import timedelta
import os
import sys
import time

import numpy as np

from .stdlib import get_terminal_size


def _timestamp2timedelta(timestamp):
    return timedelta(seconds=np.ceil(timestamp))


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


class NoProgressBar(object):
    """A progress bar that does not display anything.

    Helpful in headless situations or when using Nengo as a library.
    """

    def update(self, progress):
        pass


class TerminalProgressBar(object):
    """A progress bar that is displayed as ASCII output on `stdout`."""

    def update(self, progress):
        if progress.finished:
            line = self._get_finished_line(progress)
        else:
            line = self._get_in_progress_line(progress)
        sys.stdout.write(line)
        sys.stdout.flush()

    def _get_in_progress_line(self, progress):
        line = "[{{0}}] ETA: {eta}".format(
            eta=_timestamp2timedelta(progress.eta()))
        percent_str = " {0}% ".format(int(100 * progress.progress))

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

    def _get_finished_line(self, progress):
        width, _ = get_terminal_size()
        line = "Done in {0}.".format(
            _timestamp2timedelta(progress.elapsed_seconds())).ljust(width)
        return '\r' + line + os.linesep
