import time

import pytest

import nengo
from nengo.utils.progress import AutoProgressBar, Progress, ProgressObserver


class ProgressObserverMock(ProgressObserver):
    def __init__(self):
        self.n_update_calls = 0

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

    def test_notifies_observers(self):
        observer = ProgressObserverMock()
        with Progress(10, [observer]) as p:
            for _ in range(5):
                p.step()
        assert observer.n_update_calls >= 5

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
        except:  # pylint: disable=bare-except
            pass
        assert not p2.success

    def test_seconds_passed(self):
        with Progress(10) as p:
            time.sleep(1)
        assert p.seconds_passed > 0

    def test_eta(self):
        with Progress(10) as p:
            assert p.eta == -1  # no estimate available yet
            p.step()
            assert p.eta > 0.


class TestAutoProgressBar(object):
    class ProgressMock(object):
        def __init__(self, eta, start_time=1234.5):
            self.eta = eta
            self.start_time = start_time

    def test_progress_not_shown_if_eta_below_threshold(self):
        progress_mock = self.ProgressMock(0.2)
        observer = ProgressObserverMock()
        auto_progress = AutoProgressBar(observer, min_eta=10.)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert observer.n_update_calls == 0

    def test_progress_shown_if_eta_above_threshold(self):
        progress_mock = self.ProgressMock(20.)
        observer = ProgressObserverMock()
        auto_progress = AutoProgressBar(observer, min_eta=10.)

        for _ in range(10):
            auto_progress.update(progress_mock)

        assert observer.n_update_calls >= 10


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
