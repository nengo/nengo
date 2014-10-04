from __future__ import absolute_import, division

from datetime import timedelta
import os
import sys
import time

import numpy as np

from nengo.utils.compat import get_terminal_size


class Progress(object):
    def __init__(self, max_steps):
        self.steps = 0
        self.max_steps = max_steps
        self.start_time = self.end_time = time.time()
        self.finished = False

    @property
    def progress(self):
        return self.steps / self.max_steps

    @property
    def seconds_passed(self):
        if self.finished:
            return self.end_time - self.start_time
        else:
            return time.time() - self.start_time

    @property
    def eta(self):
        if self.progress > 0.:
            return (1. - self.progress) * self.seconds_passed / self.progress
        else:
            return -1

    def start(self):
        self.finished = False
        self.steps = 0
        self.start_time = time.time()

    def finish(self, success=True):
        if success:
            self.steps = self.max_steps
        self.end_time = time.time()
        self.finished = True

    def step(self, n=1):
        self.steps = min(self.steps + n, self.max_steps)


class ProgressBar(object):
    def __init__(self, update_interval=0.05):
        self.last_update = 0
        self.update_interval = update_interval

    def init(self):
        self._on_init()

    def _on_init(self):
        raise NotImplementedError()

    def update(self, progress):
        if progress.finished:
            self._on_finish(progress)
        elif self.last_update + self.update_interval < time.time():
            self._on_update(progress)
            self.last_update = time.time()

    def _on_update(self, progress):
        raise NotImplementedError()

    def _on_finish(self, progress):
        raise NotImplementedError()


class NoProgressBar(ProgressBar):
    def _on_init(self):
        pass

    def _on_update(self, progress):
        pass

    def _on_finish(self, progress):
        pass


class CmdProgressBar(ProgressBar):
    def _on_init(self):
        pass

    def _on_update(self, progress):
        line = "[{{}}] ETA: {eta}".format(eta=timedelta(
            seconds=np.ceil(progress.eta)))
        percent_str = " {}% ".format(int(100 * progress.progress))

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

        sys.stdout.write('\r' + line.format(progress_str))
        sys.stdout.flush()

    def _on_finish(self, progress):
        width, _ = get_terminal_size()
        line = "Done in {}.".format(
            timedelta(seconds=np.ceil(progress.seconds_passed))).ljust(width)
        sys.stdout.write('\r' + line + os.linesep)
        sys.stdout.flush()


def get_progressbar():
    return CmdProgressBar()
