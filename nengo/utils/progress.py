from __future__ import division

import os
import sys

from nengo.utils.compat import get_terminal_size


class ProgressBar(object):
    def __init__(self, max_steps, update_interval=100):
        self.steps = 0
        self.max_steps = max_steps
        self.last_update = 0
        self.update_interval = update_interval

    @property
    def progress(self):
        return self.steps / self.max_steps

    def start(self):
        self.steps = 0
        self.last_update = 0
        self._on_start()
        self.update()

    def _on_start(self):
        raise NotImplementedError()

    def finish(self):
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
        sys.stdout.write('\r' + os.linesep)
        sys.stdout.flush()

    def _on_update(self):
        width, _ = get_terminal_size()
        ticks = int(width * self.progress)
        sys.stdout.write('\r' + '#' * ticks)
        sys.stdout.flush()


def get_progressbar():
    return CmdProgress
