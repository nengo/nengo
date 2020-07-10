import numpy as np
import pytest

from nengo.utils.testing import signals_allclose, ThreadedAssertion


def add_close_noise(x, atol, rtol, rng):
    scale = rng.uniform(-rtol, rtol, size=x.shape)
    offset = rng.uniform(-atol, atol, size=x.shape)
    return x + scale * np.abs(x) + offset


@pytest.mark.parametrize("multiple_targets", (False, True))
def test_signals_allclose(multiple_targets, rng):
    nt = 100  # signal length
    ny = 3  # number of comparison signals
    atol = 1e-5
    rtol = 1e-3

    t = 0.01 * np.arange(nt)
    x = rng.uniform(-1, 1, size=(nt, ny if multiple_targets else 1))
    get_y = lambda close: add_close_noise(
        x, (1 if close else 3) * atol, (1 if close else 3) * rtol, rng
    )
    y_close = (
        get_y(True)
        if multiple_targets
        else np.column_stack([get_y(True) for _ in range(ny)])
    )
    y_far = (
        get_y(False)
        if multiple_targets
        else np.column_stack([get_y(False) for _ in range(ny)])
    )

    result = signals_allclose(
        t, x, y_close, atol=atol, rtol=rtol, individual_results=False
    )
    assert result is True

    result = signals_allclose(
        t, x, y_far, atol=atol, rtol=rtol, individual_results=False
    )
    assert result is False

    result = signals_allclose(
        t, x, y_close, atol=atol, rtol=rtol, individual_results=True
    )
    assert np.array_equal(result, np.ones(ny, dtype=bool))

    result = signals_allclose(
        t, x, y_far, atol=atol, rtol=rtol, individual_results=True
    )
    assert np.array_equal(result, np.zeros(ny, dtype=bool))

    # tests for `signals.ndim == 1` and `targets.ndim == 1`
    if not multiple_targets:
        assert x.shape[1] == 1
        assert signals_allclose(t, x[:, 0], y_close[:, 0], atol=atol, rtol=rtol)
        assert not signals_allclose(t, x[:, 0], y_far[:, 0], atol=atol, rtol=rtol)


def test_threadedassertion():
    class Test(ThreadedAssertion):
        def __init__(self, n_threads, assert_inds, **kwargs):
            self.assert_inds = assert_inds
            super().__init__(n_threads, **kwargs)

        def init_thread(self, worker):
            worker.init_param = "good"

        def assert_thread(self, worker):
            assert worker.init_param == "good", "Worker did not init"

            if worker.n in self.assert_inds:
                raise RuntimeError("Worker %d failed properly" % worker.n)

        def finish_thread(self, worker):
            worker.finish_param = "finished"

    # running with assert_inds=[] should pass
    test = Test(n_threads=3, assert_inds=[])
    assert all(thread.finish_param == "finished" for thread in test.threads)

    with pytest.raises(RuntimeError, match="Worker 1 failed properly"):
        Test(n_threads=3, assert_inds=[1])

    test = Test(n_threads=3, assert_inds=[1], run_on_init=False)
    with pytest.raises(RuntimeError, match="Worker 1 failed properly"):
        test.run()
    assert all(thread.finish_param == "finished" for thread in test.threads)

    test = Test(n_threads=3, assert_inds=[2], run_on_init=False)
    with pytest.raises(RuntimeError, match="Worker 2 failed properly"):
        test.run()
    assert all(thread.finish_param == "finished" for thread in test.threads)
