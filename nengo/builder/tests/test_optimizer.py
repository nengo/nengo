import numpy as np
import pytest

from nengo.builder.optimizer import _SigMerger
from nengo.builder.signal import Signal


def test_sigmerger_check():
    # 0-d signals
    assert _SigMerger.check([Signal(0), Signal(0)])
    assert not _SigMerger.check([Signal(0), Signal(1)])

    # compatible along first axis
    assert _SigMerger.check(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    assert _SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    assert not _SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    assert not _SigMerger.check(
        [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    assert not _SigMerger.check(
        [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    s1 = Signal(np.empty(5))
    s2 = Signal(np.empty(5))

    # mixed signal and view
    assert not _SigMerger.check([s1, s1[:3]])

    # mixed bases
    assert not _SigMerger.check([s1[:2], s2[2:]])

    # compatible views
    assert _SigMerger.check([s1[:2], s1[2:]])


def test_sigmerger_check_signals():
    # 0-d signals
    _SigMerger.check_signals([Signal(0), Signal(0)])
    with pytest.raises(ValueError):
        _SigMerger.check_signals([Signal(0), Signal(1)])

    # compatible along first axis
    _SigMerger.check_signals(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    _SigMerger.check_signals(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    with pytest.raises(ValueError):
        _SigMerger.check_signals(
            [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        _SigMerger.check_signals(
            [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    with pytest.raises(ValueError):
        _SigMerger.check_signals(
            [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    # compatible views
    s = Signal(np.empty(5))
    with pytest.raises(ValueError):
        _SigMerger.check_signals([s[:2], s[2:]])


def test_sigmerger_check_views():
    s1 = Signal(np.empty((5, 5)))
    s2 = Signal(np.empty((5, 5)))

    # compatible along first axis
    _SigMerger.check_views([s1[:1], s1[1:]])

    # compatible along second axis
    _SigMerger.check_views([s1[0:1, :1], s1[0:1, 1:]], axis=1)
    with pytest.raises(ValueError):
        _SigMerger.check_views([s1[0:1, :1], s1[0:1, 1:]], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        _SigMerger.check_views([s1[:1], s1[1:, 0]])

    # different bases
    with pytest.raises(ValueError):
        _SigMerger.check_views([s1[:2], s2[2:]])


def test_sigmerger_merge():
    s1 = Signal(np.array([[0, 1], [2, 3]]))
    s2 = Signal(np.array([[4, 5]]))

    sig, replacements = _SigMerger.merge([s1, s2])
    assert np.allclose(sig.initial_value, np.array([[0, 1], [2, 3], [4, 5]]))
    assert np.allclose(replacements[s1].initial_value, s1.initial_value)
    assert np.allclose(replacements[s2].initial_value, s2.initial_value)


def test_sigmerger_merge_views():
    s = Signal(np.array([[0, 1], [2, 3], [4, 5]]))
    v1, v2 = s[:2], s[2:]
    merged, _ = _SigMerger.merge_views([v1, v2])

    assert np.allclose(merged.initial_value, s.initial_value)
    assert v1.base is s
    assert v2.base is s
