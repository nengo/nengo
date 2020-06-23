import numpy as np
import pytest

from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.utils.simulator import validate_ops


def test_validate_ops():
    """tests validate_ops, including may_share_memory"""

    base1 = Signal(initial_value=np.ones((10, 4)))
    base2 = Signal(initial_value=np.ones((10, 4)))
    view1_a = base1[:5, :]
    view1_b = base1[5:, :]
    view1_ab = base1[:6, :]

    ops = [Operator() for _ in range(3)]

    # non-overlapping sets is OK
    validate_ops(sets={base1: [ops[1]], base2: [ops[2]]}, ups=[], incs=[])

    # missing set is bad
    with pytest.raises(AssertionError):
        validate_ops(sets={base1: [ops[1]], base2: []}, ups=[], incs=[])

    # multiple sets is bad
    with pytest.raises(AssertionError):
        validate_ops(sets={base1: [ops[1]], base2: ops}, ups=[], incs=[])

    # set base and view is bad
    with pytest.raises(AssertionError):
        validate_ops(sets={base1: [ops[1]], view1_a: [ops[2]]}, ups=[], incs=[])

    # set non-overlapping views is OK
    validate_ops(sets={view1_a: [ops[1]], view1_b: [ops[2]]}, ups=[], incs=[])

    # set overlapping views is bad
    with pytest.raises(AssertionError):
        validate_ops(sets={view1_ab: [ops[1]], view1_b: [ops[2]]}, ups=[], incs=[])

    # non-overlapping updates is OK
    validate_ops(ups={base1: [ops[1]], base2: [ops[2]]}, sets=[], incs=[])

    # missing update is bad
    with pytest.raises(AssertionError):
        validate_ops(ups={base1: [ops[1]], base2: []}, sets=[], incs=[])

    # multiple updates is bad
    with pytest.raises(AssertionError):
        validate_ops(ups={base1: [ops[1]], base2: ops}, sets=[], incs=[])

    # update base and view is bad
    with pytest.raises(AssertionError):
        validate_ops(ups={base1: [ops[1]], view1_a: [ops[2]]}, sets=[], incs=[])

    # update non-overlapping views is OK
    validate_ops(ups={view1_a: [ops[1]], view1_b: [ops[2]]}, sets=[], incs=[])

    # update overlapping views is bad
    with pytest.raises(AssertionError):
        validate_ops(ups={view1_ab: [ops[1]], view1_b: [ops[2]]}, sets=[], incs=[])
