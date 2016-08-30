import numpy as np

from nengo.builder.signal import Signal


def test_signal_offset():
    value = np.eye(3)
    s = Signal(value)
    assert s.offset == 0
    assert s[1].offset == value.strides[0]

    value_view = value[1]
    s = Signal(value_view)
    assert s.offset == 0
    assert s[0:].offset == 0
    assert s[1:].offset == value.strides[1]
