import pytest

import nengo
from nengo.utils.cache import bytes2human, human2bytes


def test_bytes2human():
    assert bytes2human(1) == '1.0 B'
    assert bytes2human(10000) == '9.8 KB'
    assert bytes2human(100001221) == '95.4 MB'


def test_human2bytes():
    assert human2bytes('1 MB') == 1048576
    assert human2bytes('1.5 GB') == 1610612736
    assert human2bytes('14 B') == 14


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
