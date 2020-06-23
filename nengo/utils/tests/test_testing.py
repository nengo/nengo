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

    # tests for signals.ndim == 1
    if multiple_targets:

        def my_function2(t, y, label="sure"):
            return True

        def my_function3(loc, bbox_to_anchor):
            return True

        class myPlot:
            plot = my_function2
            legend = my_function3

            def set_ylabel(self):
                return True

            def set_xlabel(self):
                return True

        def my_function(a, b, c):
            return myPlot

        def my_function4():
            return True

        class Test:
            subplot = my_function
            show = my_function4

        result = signals_allclose(
            (1, 2),
            ((1, 2)),
            (1, 2),
            atol=atol,
            rtol=rtol,
            individual_results=True,
            plt=Test,
            labels="not None",
            show=True,
        )


def test_threadedassertion_errors():
    """tests threadedassertion throws appropriate errors"""
    with pytest.raises(AttributeError):
        ThreadedAssertion(1)
    a = ThreadedAssertion(0)
    # there is some magic going on here to
    # access the run command and get the specific error I want

    def function(a, b):
        return "string"

    class FakeException(BaseException):
        with_traceback = function

    class FakeThreadedAssertion(ThreadedAssertion):
        exc_info = [0, ValueError("I'm a fancy error message"), FakeException(), 3, 4]

    with pytest.raises(ValueError):
        a = FakeThreadedAssertion(1)
    parent = None
    barriers = None
    n = 1
    a.AssertionWorker(parent, barriers, n)
    # make a subclass of threaded assertion, use assert thread to assert false
    # then catch the assertion error
    # profit
