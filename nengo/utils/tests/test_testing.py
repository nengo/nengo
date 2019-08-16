import errno
import os

import numpy as np
import pytest

from nengo.utils.testing import Analytics, Logger, signals_allclose


def test_analytics_empty():
    analytics = Analytics(
        "nengo.simulator.analytics",
        "nengo.utils.tests.test_testing",
        "test_analytics_empty",
    )
    with analytics:
        pass
    path = analytics.get_filepath(ext="npz")
    assert not os.path.exists(path)


def test_analytics_record():
    analytics = Analytics(
        "nengo.simulator.analytics",
        "nengo.utils.tests.test_testing",
        "test_analytics_record",
    )
    with analytics:
        analytics.add_data("test", 1, "Test analytics implementation")
        assert analytics.data["test"] == 1
        assert analytics.doc["test"] == "Test analytics implementation"
        with pytest.raises(ValueError):
            analytics.add_data("documentation", "")
    path = analytics.get_filepath(ext="npz")
    assert os.path.exists(path)
    os.remove(path)
    # This will remove the analytics directory, only if it's empty
    try:
        os.rmdir(analytics.dirname)
    except OSError as ex:
        assert ex.errno == errno.ENOTEMPTY


def test_analytics_norecord():
    analytics = Analytics(
        None, "nengo.utils.tests.test_testing", "test_analytics_norecord"
    )
    with analytics:
        analytics.add_data("test", 1, "Test analytics implementation")
        assert "test" not in analytics.data
        assert "test" not in analytics.doc
    with pytest.raises(ValueError):
        analytics.get_filepath(ext="npz")


def test_logger_record():
    logger_obj = Logger(
        "nengo.simulator.logs", "nengo.utils.tests.test_testing", "test_logger_record"
    )
    with logger_obj as logger:
        logger.info("Testing that logger records")
    path = logger_obj.get_filepath(ext="txt")
    assert os.path.exists(path)
    os.remove(path)
    # This will remove the logger directory, only if it's empty
    try:
        os.rmdir(logger_obj.dirname)
    except OSError as ex:
        assert ex.errno == errno.ENOTEMPTY


def test_logger_norecord():
    logger_obj = Logger(None, "nengo.utils.tests.test_testing", "test_logger_norecord")
    with logger_obj as logger:
        logger.info("Testing that logger doesn't record")
    with pytest.raises(ValueError):
        logger_obj.get_filepath(ext="txt")


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
