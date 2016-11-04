import pytest

import nengo.utils.probe
from nengo.exceptions import MovedError


def test_moved():
    with pytest.raises(MovedError):
        nengo.utils.probe.probe_all()
