import time

import pytest
import numpy as np

import nengo.rc as rc
from nengo.exceptions import ValidationError
from nengo.utils.progress import (
    AutoProgressBar,
    Progress,
    ProgressBar,
    _load_class,
    WriteProgressToFile,
    get_default_progressbar,
)


class ProgressBarMock(ProgressBar):
    def __init__(self):
        super().__init__()
        self.n_update_calls = 0

    def update(self, progress):
        self.n_update_calls += 1


class TestProgress:
    def test_progress_calculation(self):
        with Progress(max_steps=10) as p:
            assert p.progress == 0.0
            for _ in range(5):
                p.step()
            assert p.progress == 0.5
            p.step(5)
            assert p.progress == 1.0

    def test_finished_property(self):
        with Progress(max_steps=10) as p:
            assert not p.finished
            p.step(5)
            assert not p.finished
        assert p.finished

    def test_success_property(self):
        with Progress(max_steps=10) as p:
            assert p.success is None
        assert p.success

        try:
            with Progress(max_steps=10) as p2:
                raise Exception()
        except Exception:
            pass
        assert not p2.success

    def test_elapsed_seconds(self, monkeypatch):
        t = 1.0
        monkeypatch.setattr(time, "time", lambda: t)

        with Progress(max_steps=10) as p:
            t = 10.0

        assert p.elapsed_seconds() == 9.0

    def test_eta(self):
        with Progress(max_steps=10) as p:
            assert p.eta() == -1  # no estimate available yet
            p.step()
            assert p.eta() >= 0.0

    def test_max_steps(self):
        with pytest.raises(ValidationError):
            with Progress(max_steps=0):
                pass

        with pytest.raises(ValidationError):
            with Progress(max_steps=-1):
                pass

    def test_unknown_number_of_steps(self, monkeypatch):
        t = 1.0
        monkeypatch.setattr(time, "time", lambda: t)

        with Progress() as p:
            p.step()
            t = 10.0
            assert p.progress == 0.0
            assert p.eta() == -1

        assert p.n_steps == 1
        assert p.elapsed_seconds() == 9.0


class TestAutoProgressBar:
    class ProgressMock:
        def __init__(self, eta, time_start=1234.5):
            self.eta = lambda: eta
            self.elapsed_seconds = lambda: 0
            self.time_start = time_start
            self.finished = False

    def test_progress_not_shown_if_eta_below_threshold(self):
        progress_mock = self.ProgressMock(0.2)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.0)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert progress_bar.n_update_calls == 0

    def test_progress_shown_if_eta_above_threshold(self):
        progress_mock = self.ProgressMock(20)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.0)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert progress_bar.n_update_calls >= 10

    def test_progress_shown_when_finished(self):
        progress_mock = self.ProgressMock(0.2)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.0)

        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls == 0
        progress_mock.finished = True
        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls >= 1

    def test_get_default_progressbar(self):
        """tests get_default_progressbar"""

        # create rc finalizer

        progress_mock = self.ProgressMock(0.2)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.0)

        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls == 0
        progress_mock.finished = True
        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls >= 1
        rc.set("progress", "progress_bar", "False")
        assert not rc.getboolean("progress", "progress_bar")
        assert str(get_default_progressbar()).startswith(
            "<nengo.utils.progress.NoProgressBar object at "
        )
        rc.set("progress", "progress_bar", "True")
        assert rc.getboolean("progress", "progress_bar")
        assert str(get_default_progressbar()).startswith(
            "<nengo.utils.progress.AutoProgressBar object at "
        )
        rc.set("progress", "progress_bar", "NotBool")
        assert str(get_default_progressbar()).startswith(
            "<nengo.utils.progress.NoProgressBar object at"
        )


def test_load_class():
    """tests _load_class"""
    name = "numpy.sin"
    _load_class(name) == np.sin


def test_write_progress_to_file():
    """tests WriteProgressToFile"""
    filename = "test_progress_file"
    prog = WriteProgressToFile(filename)

    class Test:
        def __init__(self, isfinished):
            self.finished = isfinished

        def myFunction(self):
            return 1

        progress = 0
        name_after = "newname"
        elapsed_seconds = myFunction

        def eta(self):
            return 0

    data = Test(False)
    prog.update(data)

    data = Test(True)
    prog.update(data)
