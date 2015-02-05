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


class MockWithName(object):
    def __init__(self, name):
        self.__name__ = name


def test_analytics_store():
    analytics = Analytics(nengo.Simulator,
                          MockWithName('nengo.utils.tests.test_testing'),
                          MockWithName('test_analytics_store'),
                          store=True)
    with analytics:
        analytics.add_data('test', 1, "Test analytics implementation")
        assert analytics.data['test'] == 1
        assert analytics.desc['test'] == "Test analytics implementation"
    datapath = os.path.join(analytics.dirname, analytics.data_filename)
    descpath = os.path.join(analytics.dirname, analytics.desc_filename)
    assert os.path.exists(datapath)
    assert os.path.exists(descpath)
    os.remove(datapath)
    os.remove(descpath)


def test_analytics_nostore():
    analytics = Analytics(nengo.Simulator,
                          MockWithName('nengo.utils.tests.test_testing'),
                          MockWithName('test_analytics_nostore'),
                          store=False)
    with analytics:
        analytics.add_data('test', 1, "Test analytics implementation")
        assert 'test' not in analytics.data
        assert 'test' not in analytics.desc
    datapath = os.path.join(analytics.dirname, analytics.data_filename)
    descpath = os.path.join(analytics.dirname, analytics.desc_filename)
    assert not os.path.exists(datapath)
    assert not os.path.exists(descpath)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
