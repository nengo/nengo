import time

from nengo.utils.progress import (
    AutoProgressBar, UpdateEveryN, UpdateEveryT, UpdateN, Progress,
    ProgressBar, ProgressTracker)


class ProgressBarMock(ProgressBar):
    def __init__(self, task="Testing"):
        self.n_update_calls = 0
        super(ProgressBarMock, self).__init__(task)

    def update(self, progress):
        self.n_update_calls += 1


class TestProgress(object):
    def test_progress_calculation(self):
        with Progress(10) as p:
            assert p.progress == 0.
            for _ in range(5):
                p.step()
            assert p.progress == 0.5
            p.step(5)
            assert p.progress == 1.

    def test_finished_property(self):
        with Progress(10) as p:
            assert not p.finished
            p.step(5)
            assert not p.finished
        assert p.finished

    def test_success_property(self):
        with Progress(10) as p:
            assert p.success is None
        assert p.success

        try:
            with Progress(10) as p2:
                raise Exception()
        except Exception:
            pass
        assert not p2.success

    def test_elapsed_seconds(self, monkeypatch):
        t = 1.
        monkeypatch.setattr(time, 'time', lambda: t)

        with Progress(10) as p:
            t = 10.

        assert p.elapsed_seconds() == 9.

    def test_eta(self):
        with Progress(10) as p:
            assert p.eta() == -1  # no estimate available yet
            p.step()
            assert p.eta() >= 0.


class TestAutoProgressBar(object):
    class ProgressMock(object):
        def __init__(self, eta, start_time=1234.5):
            self.eta = lambda: eta
            self.elapsed_seconds = lambda: 0
            self.start_time = start_time
            self.finished = False

    def test_progress_not_shown_if_eta_below_threshold(self):
        progress_mock = self.ProgressMock(0.2)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert progress_bar.n_update_calls == 0

    def test_progress_shown_if_eta_above_threshold(self):
        progress_mock = self.ProgressMock(20)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert progress_bar.n_update_calls >= 10

    def test_progress_shown_when_finished(self):
        progress_mock = self.ProgressMock(0.2)
        progress_bar = ProgressBarMock()
        auto_progress = AutoProgressBar(progress_bar, min_eta=10.)

        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls == 0
        progress_mock.finished = True
        auto_progress.update(progress_mock)
        assert progress_bar.n_update_calls >= 1


class TestUpdateN(object):
    def test_at_most_n_updates_are_performed(self):
        progress_bar = ProgressBarMock()
        updater = UpdateN(progress_bar, max_updates=3)

        with ProgressTracker(100, updater, "Testing") as p:
            for _ in range(100):
                p.step()

        assert progress_bar.n_update_calls > 0
        assert progress_bar.n_update_calls <= 3


class TestUpdateEveryN(object):
    def test_updates_every_n_steps(self):
        progress_bar = ProgressBarMock()
        updater = UpdateEveryN(progress_bar, every_n=5)

        with ProgressTracker(100, updater, "Testing") as p:
            progress_bar.n_update_calls = 0
            for _ in range(5):
                p.step()
            assert progress_bar.n_update_calls == 1

            p.step(2)
            assert progress_bar.n_update_calls == 1
            p.step(3)
            assert progress_bar.n_update_calls == 2


class TestUpdateEveryT(object):
    def test_updates_after_interval_has_passed(self, monkeypatch):
        progress_bar = ProgressBarMock()
        updater = UpdateEveryT(progress_bar, every_t=2.)
        t = 1.
        monkeypatch.setattr(time, 'time', lambda: t)

        with ProgressTracker(100, updater, "Testing") as p:
            p.step()  # Update is allowed to happen on first step.

            progress_bar.n_update_calls = 0
            p.step()
            assert progress_bar.n_update_calls == 0

            t = 2.
            p.step()
            assert progress_bar.n_update_calls == 0

            t = 4.
            p.step()
            assert progress_bar.n_update_calls == 1
