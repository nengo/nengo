import numpy as np
import pytest

from nengo.builder.optimizer import (
    compatible, check_signals_mergeable, check_views_mergeable, merge_signals,
    merge_views)
from nengo.builder.signal import Signal


def test_compatible():
    # 0-d signals
    assert compatible([Signal(0), Signal(0)])
    assert not compatible([Signal(0), Signal(1)])

    # compatible along first axis
    assert compatible(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    assert compatible(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    assert not compatible(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    assert not compatible(
        [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    assert not compatible(
        [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    s1 = Signal(np.empty(5))
    s2 = Signal(np.empty(5))

    # mixed signal and view
    assert not compatible([s1, s1[:3]])

    # mixed bases
    assert not compatible([s1[:2], s2[2:]])

    # compatible views
    assert compatible([s1[:2], s1[2:]])


def test_check_signals_mergeable():
    # 0-d signals
    check_signals_mergeable([Signal(0), Signal(0)])
    with pytest.raises(ValueError):
        check_signals_mergeable([Signal(0), Signal(1)])

    # compatible along first axis
    check_signals_mergeable(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    check_signals_mergeable(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    with pytest.raises(ValueError):
        check_signals_mergeable(
            [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        check_signals_mergeable(
            [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    with pytest.raises(ValueError):
        check_signals_mergeable(
            [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    # compatible views
    s = Signal(np.empty(5))
    with pytest.raises(ValueError):
        check_signals_mergeable([s[:2], s[2:]])


def test_check_views_mergeable():
    s1 = Signal(np.empty((5, 5)))
    s2 = Signal(np.empty((5, 5)))

    # compatible along first axis
    check_views_mergeable([s1[:1], s1[1:]])

    # compatible along second axis
    check_views_mergeable([s1[0:1, :1], s1[0:1, 1:]], axis=1)
    with pytest.raises(ValueError):
        check_views_mergeable([s1[0:1, :1], s1[0:1, 1:]], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        check_views_mergeable([s1[:1], s1[1:, 0]])

    # different bases
    with pytest.raises(ValueError):
        check_views_mergeable([s1[:2], s2[2:]])


def test_merge_signals():
    s1 = Signal(np.array([[0, 1], [2, 3]]))
    s2 = Signal(np.array([[4, 5]]))

    replacements = {}
    assert np.allclose(
        merge_signals([s1, s2], replacements).initial_value,
        np.array([[0, 1], [2, 3], [4, 5]]))
    assert np.allclose(replacements[s1].initial_value, s1.initial_value)
    assert np.allclose(replacements[s2].initial_value, s2.initial_value)


def test_merge_views():
    s = Signal(np.array([[0, 1], [2, 3], [4, 5]]))
    v1, v2 = s[:2], s[2:]
    merged = merge_views([v1, v2])

    assert np.allclose(merged.initial_value, s.initial_value)
    assert v1.base is s
    assert v2.base is s
