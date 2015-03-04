import errno
import logging
import os

import pytest

import nengo
from nengo.utils.testing import Analytics, Timer

logger = logging.getLogger(__name__)


def test_timer():
    with Timer() as timer:
        2 + 2
    assert timer.duration > 0.0
    assert timer.duration < 1.0  # Pretty bad worst case


def test_analytics_empty():
    analytics = Analytics('nengo.simulator.analytics',
                          'nengo.utils.tests.test_testing',
                          'test_analytics_empty')
    with analytics:
        pass
    path = analytics.get_filepath(ext='npz')
    assert not os.path.exists(path)


def test_analytics_record():
    analytics = Analytics('nengo.simulator.analytics',
                          'nengo.utils.tests.test_testing',
                          'test_analytics_record')
    with analytics:
        analytics.add_data('test', 1, "Test analytics implementation")
        assert analytics.data['test'] == 1
        assert analytics.doc['test'] == "Test analytics implementation"
        with pytest.raises(ValueError):
            analytics.add_data('documentation', '')
    path = analytics.get_filepath(ext='npz')
    assert os.path.exists(path)
    os.remove(path)
    # This will remove the analytics directory, only if it's empty
    try:
        os.rmdir(analytics.dirname)
    except OSError as ex:
        assert ex.errno == errno.ENOTEMPTY


def test_analytics_norecord():
    analytics = Analytics(None,
                          'nengo.utils.tests.test_testing',
                          'test_analytics_norecord')
    with analytics:
        analytics.add_data('test', 1, "Test analytics implementation")
        assert 'test' not in analytics.data
        assert 'test' not in analytics.doc
    with pytest.raises(ValueError):
        analytics.get_filepath(ext='npz')


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
