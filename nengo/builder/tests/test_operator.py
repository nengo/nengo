import numpy as np
import pytest

from nengo.builder import Signal
from nengo.builder.operator import (
    SparseDotInc,
    Operator,
    Copy,
    ElementwiseInc,
    reshape_dot,
    DotInc,
)
from nengo.exceptions import BuildError


def test_operator_repr():
    """tests repr for the Operator class"""
    assert repr(Operator()).startswith("<Operator  at ")


def test_copy_operator(rng):
    """tests the Copy class in the operator file"""
    cp = Copy(1, 2)
    signals = [1, 2, [1, 2, 3]]
    dt = 0
    cp.dst_slice = np.array([1, 1, 3])  # repeat 1
    with pytest.raises(BuildError):
        cp.make_step(signals, dt, rng)


def test_elementwiseinc_builderror(rng):
    """tests the ElementwiseInc class for a build error"""
    ew = ElementwiseInc(0, 1, 2)
    signals = [np.array([1]), np.array([2]), np.array([5, 6])]
    dt = 0
    with pytest.raises(BuildError):
        ew.make_step(signals, dt, rng)


def test_reshape_dot(rng):
    """tests the reshape_dot function"""
    A = np.array(1)
    X = np.array(1)
    Y = np.array(1)
    reshape_dot(A, X, Y)
    A = np.array([])
    X = np.array([])
    Y = np.array([])
    reshape_dot(A, X, Y)
    X = np.zeros((1, 2))
    with pytest.raises(BuildError):
        reshape_dot(A, X, Y)


def test_dotinc(rng):
    """tests the DotInc class BuildErrors"""

    class Test:
        ndim = 2
        reshape = None
        shape = np.array([1, 2])
        initial_value = 2

        def __init__(self, ndim):
            self.ndim = ndim

        def __getitem__(self, other):
            return 1

    with pytest.raises(BuildError):
        DotInc(Test(2), Test(2), Test(2))

    with pytest.raises(BuildError):
        DotInc(Test(2), Test(1), Test(2))


def test_sparsedotinc_builderror():
    A = Signal(np.ones(2))
    X = Signal(np.ones(2))
    Y = Signal(np.ones(2))

    with pytest.raises(BuildError, match="must be a sparse Signal"):
        SparseDotInc(A, X, Y)
