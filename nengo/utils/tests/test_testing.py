import numpy as np
import pytest

from nengo.utils.testing import signals_allclose


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
