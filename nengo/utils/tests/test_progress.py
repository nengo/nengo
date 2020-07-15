from collections import namedtuple
import sys
import time

import pytest

import nengo.rc as rc
from nengo.exceptions import ValidationError
from nengo.utils.progress import (
    AutoProgressBar,
    get_default_progressbar,
    HtmlProgressBar,
    NoProgressBar,
    Progress,
    ProgressBar,
    ProgressTracker,
    TerminalProgressBar,
    to_progressbar,
    WriteProgressToFile,
)


Update = namedtuple(
    "Update", ("name_during", "name_after", "n_steps", "max_steps", "finished")
)


class ProgressBarMock(ProgressBar):
    def __init__(self):
        super().__init__()
        self.n_update_calls = 0
        self.updates = []
        self.closed = False

    def update(self, progress):
        self.n_update_calls += 1
        self.updates.append(
            Update(
                name_during=progress.name_during,
                name_after=progress.name_after,
                n_steps=progress.n_steps,
                max_steps=progress.max_steps,
                finished=progress.finished,
            )
        )

    def close(self):
        self.closed = True


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
                raise UnboundLocalError()
        except UnboundLocalError:
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
    class ProgressMock(Progress):
        def __init__(self, eta):
            super().__init__()
            self.eta = lambda: eta
            # make this 0 to easily control `long_eta` in AutoProgressBar.update
            self.elapsed_seconds = lambda: 0
            # make this small so `min_delay < time.time()` in AutoProgressBar.update
            self.time_start = 1.0

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

    def test_close(self):
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar)
        auto_progress.close()
        assert progress_bar.closed

    def test_get_default_progressbar(self):
        rc["progress"]["progress_bar"] = "False"
        assert isinstance(get_default_progressbar(), NoProgressBar)

        rc["progress"]["progress_bar"] = "True"
        assert isinstance(get_default_progressbar(), AutoProgressBar)

        rc["progress"]["progress_bar"] = "nengo.utils.progress.HtmlProgressBar"
        assert isinstance(get_default_progressbar(), HtmlProgressBar)

        rc["progress"]["progress_bar"] = "nengo.utils.progress.TerminalProgressBar"
        assert isinstance(get_default_progressbar(), TerminalProgressBar)


def test_write_progress_to_file(tmpdir):
    """Tests the WriteProgressToFile progress bar type"""

    def check_file(filename, startstring):
        with open(filename, "r") as fh:
            data = fh.read()
        assert data.startswith(startstring)

    filename = str(tmpdir.join("test_write_progress_file.txt"))
    progress = Progress(name_during="myprog", max_steps=2)
    bar = WriteProgressToFile(filename)

    with progress:
        bar.update(progress)
        check_file(filename, "0%, ETA")

        progress.step()
        bar.update(progress)
        check_file(filename, "50%, ETA")

        progress.step()
        bar.update(progress)
        check_file(filename, "100%, ETA")

    bar.update(progress)
    check_file(filename, "myprog finished in")


@pytest.mark.skipif(
    sys.platform == "win32", reason="Threading in Windows is less reliable"
)
def test_progress_tracker():
    update_interval = 0.001
    sleep_interval = 20 * update_interval
    stages = 4
    steps = 3
    total_progress = Progress(name_during="total_prog", max_steps=stages)
    progress_bar = ProgressBarMock()
    tracker = ProgressTracker(
        progress_bar, total_progress, update_interval=update_interval
    )

    assert not progress_bar.closed
    with tracker:
        for i in range(stages):
            sub_progress = tracker.next_stage(
                name_during="stage%d" % i, max_steps=steps
            )
            with sub_progress:
                for j in range(steps):
                    sub_progress.step()
                    time.sleep(sleep_interval)
                    assert progress_bar.updates[-1].name_during == "stage%d" % i
                    assert progress_bar.updates[-1].n_steps == j + 1

            time.sleep(sleep_interval)
            assert progress_bar.updates[-1].name_during == "total_prog"
            assert progress_bar.updates[-1].n_steps == i

    assert progress_bar.closed


def test_to_progress_bar(request, tmpdir):
    def finalizer(val=rc["progress"]["progress_bar"]):
        rc["progress"]["progress_bar"] = val

    request.addfinalizer(finalizer)

    # test false values
    for val in (None, False):
        assert isinstance(to_progressbar(val), NoProgressBar)

    # test true value
    rc["progress"]["progress_bar"] = "nengo.utils.progress.HtmlProgressBar"
    assert isinstance(to_progressbar(True), HtmlProgressBar)

    progress_bar = WriteProgressToFile(str(tmpdir.join("dummyfile.txt")))
    assert to_progressbar(progress_bar) is progress_bar
