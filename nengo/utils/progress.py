from __future__ import absolute_import, division

from datetime import timedelta
import os
import sys
import time

import numpy as np

from nengo.utils.compat import get_terminal_size


class ProgressBar(object):
    def __init__(self, max_steps, update_interval=100):
        self.steps = 0
        self.max_steps = max_steps
        self.last_update = 0
        self.start_time = self.end_time = time.time()
        self.finished = False
        self.update_interval = update_interval

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
        self.last_update = 0
        self._on_start()
        self.update()

    def _on_start(self):
        raise NotImplementedError()

    def finish(self):
        if success:
            self.steps = self.max_steps
        self.end_time = time.time()
        self.finished = True
        self._on_finish()

    def _on_finish(self):
        raise NotImplementedError()

    def step(self, n=1):
        self.steps = min(self.steps + n, self.max_steps)
        if self.last_update + self.update_interval < self.steps:
            self.update()

    def update(self):
        self.last_update = self.steps
        self._on_update()

    def _on_update(self):
        raise NotImplementedError()


class NoProgress(ProgressBar):
    def _on_start(self):
        pass

    def _on_update(self):
        pass

    def _on_finish(self):
        pass

class CmdProgress(ProgressBar):
    def _on_start(self):
        pass

    def _on_finish(self):
        width, _ = get_terminal_size()
        line = "Done in {}.".format(
            timedelta(seconds=np.ceil(self.seconds_passed))).ljust(width)
        sys.stdout.write('\r' + line + os.linesep)
        sys.stdout.flush()

    def _on_update(self):
        line = "[{{}}] ETA: {eta}".format(eta=timedelta(
            seconds=np.ceil(self.eta)))
        percent_str = " {}% ".format(int(100 * self.progress))

        width, _ = get_terminal_size()
        progress_width = max(0, width - len(line))
        progress_str = (int(progress_width * self.progress) * "#").ljust(
            progress_width)

        percent_pos = (len(progress_str) - len(percent_str)) // 2
        if percent_pos > 0:
            progress_str = (
                progress_str[:percent_pos] + percent_str +
                progress_str[percent_pos + len(percent_str):])

        sys.stdout.write('\r' + line.format(progress_str))
        sys.stdout.flush()


def get_progressbar():
    return CmdProgress
