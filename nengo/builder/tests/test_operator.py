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


def _test_operator_arg_attributes(OperatorType, argnames, args=None):
    args = {} if args is None else args
    for argname in argnames:
        args.setdefault(argname, argname)

    sim = OperatorType(**args)
    for argname in argnames:
        assert getattr(sim, argname) == args[argname]

    return args, sim


def test_operator_repr():
    """tests repr for the Operator class"""
    assert repr(Operator(tag="hi")).startswith("<Operator 'hi' at ")


def test_copy_operator(rng):
    """tests the Copy class in the operator file"""
    cp = Copy(1, 2)
    signals = [1, 2, [1, 2, 3]]
    dt = 0
    cp.dst_slice = np.array([1, 1, 3])  # repeat 1
    with pytest.raises(BuildError, match="Cannot have repeated indices"):
        cp.make_step(signals, dt, rng)


def test_elementwiseinc_builderror(rng):
    """tests the ElementwiseInc class for a build error"""
    ew = ElementwiseInc(0, 1, 2)
    signals = [np.array([1]), np.array([2]), np.array([5, 6])]
    dt = 0
    with pytest.raises(BuildError, match="Incompatible shapes in ElementwiseInc"):
        ew.make_step(signals, dt, rng)


def test_reshape_dot(rng):
    """tests the reshape_dot function"""
    scalar = np.array(1)
    vec = [np.ones(i) for i in range(4)]
    mat11 = np.ones((1, 1))
    mat23 = np.ones((2, 3))
    mat33 = np.ones((3, 3))

    # if A.shape == ():
    assert reshape_dot(A=scalar, X=scalar, Y=scalar) is True
    assert reshape_dot(A=scalar, X=vec[2], Y=vec[2]) is False
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=scalar, X=vec[3], Y=vec[1])

    # elif X.shape == ():
    assert reshape_dot(A=vec[1], X=scalar, Y=vec[1]) is True
    assert reshape_dot(A=vec[2], X=scalar, Y=vec[2]) is False
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=vec[2], X=scalar, Y=vec[1])

    # elif X.ndim == 1:
    assert reshape_dot(A=vec[0], X=vec[0], Y=vec[0]) is False
    assert reshape_dot(A=vec[1], X=vec[1], Y=vec[1]) is True
    assert reshape_dot(A=vec[1], X=vec[1], Y=scalar) is True
    assert reshape_dot(A=vec[2], X=vec[2], Y=vec[2]) is False
    assert reshape_dot(A=mat23, X=vec[3], Y=vec[2]) is False
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=mat23, X=vec[2], Y=vec[2])
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=mat23, X=vec[3], Y=vec[1])

    # else:
    assert reshape_dot(A=mat11, X=mat11, Y=mat11) is True
    assert reshape_dot(A=mat23, X=mat33, Y=mat23) is False
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=mat11, X=mat23, Y=mat23)
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=mat23, X=mat33, Y=mat33)
    with pytest.raises(BuildError, match="shape mismatch"):
        reshape_dot(A=mat23, X=mat33, Y=vec[2])


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

    with pytest.raises(BuildError, match="X must be a column vector"):
        DotInc(Test(2), Test(2), Test(2))

    with pytest.raises(BuildError, match="Y must be a column vector"):
        DotInc(Test(2), Test(1), Test(2))


def test_sparsedotinc_builderror():
    A = Signal(np.ones(2))
    X = Signal(np.ones(2))
    Y = Signal(np.ones(2))

    with pytest.raises(BuildError, match="must be a sparse Signal"):
        SparseDotInc(A, X, Y)
