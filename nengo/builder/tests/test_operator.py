import numpy as np
import pytest

from nengo.builder import Signal
from nengo.builder.operator import (
    BsrDotInc,
    Copy,
    DotInc,
    ElementwiseInc,
    Operator,
    Reset,
    reshape_dot,
    SimPyFunc,
    SparseDotInc,
    TimeUpdate,
)
from nengo.exceptions import BuildError
from nengo.transforms import SparseMatrix


def _test_operator_arg_attributes(OperatorType, argnames, args=None, non_signals=()):
    args = {} if args is None else args
    for argname in argnames:
        args.setdefault(argname, argname)

    sim = OperatorType(**args)
    for argname in argnames:
        assert getattr(sim, argname) == args[argname]

    signals = set()
    signals.update(sim.sets)
    signals.update(sim.incs)
    signals.update(sim.reads)
    signals.update(sim.updates)
    for argname in argnames:
        if argname in non_signals:
            continue

        signal = args[argname]
        assert signal in signals, "%r not added to sets/incs/reads/updates" % argname
        signals.remove(signal)

    assert len(signals) == 0, "Extra signals in sets/incs/reads/updates: %r" % signals

    return args, sim


def test_operator_str_repr():
    assert str(Operator(tag="hi")) == "Operator{ 'hi'}"
    assert repr(Operator(tag="hi")).startswith("<Operator 'hi' at ")


def test_timeupdate_op():
    argnames = ["step", "time"]
    _, sim = _test_operator_arg_attributes(TimeUpdate, argnames)
    assert str(sim) == "TimeUpdate{}"


def test_reset_op():
    argnames = ["dst"]
    _, sim = _test_operator_arg_attributes(Reset, argnames, args={"dst": "dval"})
    assert str(sim) == "Reset{dval}"


def test_copy_op(rng):
    argnames = ["src", "dst"]
    args = {"src": "sval", "dst": "dval"}
    _, sim = _test_operator_arg_attributes(Copy, argnames, args=args)
    assert str(sim) == "Copy{sval -> dval, inc=False}"

    cp = Copy(1, 2)
    signals = [1, 2, [1, 2, 3]]
    dt = 0
    cp.dst_slice = np.array([1, 1, 3])  # repeat 1
    with pytest.raises(BuildError, match="Cannot have repeated indices"):
        cp.make_step(signals, dt, rng)


def test_elementwiseinc_op(rng):
    argnames = ["A", "X", "Y"]
    args = {"A": "Av", "X": "Xv", "Y": "Yv"}
    _, sim = _test_operator_arg_attributes(ElementwiseInc, argnames, args=args)
    assert str(sim) == "ElementwiseInc{Av, Xv -> Yv}"

    ew = ElementwiseInc(0, 1, 2)
    signals = [np.array([1]), np.array([2]), np.array([5, 6])]
    dt = 0
    with pytest.raises(BuildError, match="Incompatible shapes in ElementwiseInc"):
        ew.make_step(signals, dt, rng)


def test_reshape_dot(rng):
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


def test_dotinc_op(rng):
    argnames = ["A", "X", "Y"]
    args = {
        "A": Signal(np.ones((3, 3)), name="Av"),
        "X": Signal(np.ones((3,)), name="Xv"),
        "Y": Signal(np.ones((3,)), name="Yv"),
    }
    _, sim = _test_operator_arg_attributes(DotInc, argnames, args=args)
    assert (
        str(sim)
        == "DotInc{Signal(name=Av, shape=(3, 3)), Signal(name=Xv, shape=(3,)) -> Signal(name=Yv, shape=(3,))}"
    )

    A = Signal(np.ones((2, 2)))
    X = Signal(np.ones((2, 2)))
    Y = Signal(np.ones(2))
    with pytest.raises(BuildError, match="X must be a column vector"):
        DotInc(A, X, Y)

    X = Signal(np.ones(2))
    Y = Signal(np.ones((2, 2)))
    with pytest.raises(BuildError, match="Y must be a column vector"):
        DotInc(A, X, Y)


def test_sparsedotinc_op():
    argnames = ["A", "X", "Y"]
    args = {
        "A": Signal(SparseMatrix(indices=[(0, 0)], data=[1], shape=(3, 3)), name="Av"),
        "X": Signal(np.ones((3,)), name="Xv"),
        "Y": Signal(np.ones((3,)), name="Yv"),
    }
    _test_operator_arg_attributes(DotInc, argnames, args=args)

    A = Signal(np.ones(2))
    X = Signal(np.ones(2))
    Y = Signal(np.ones(2))

    with pytest.raises(BuildError, match="must be a sparse Signal"):
        SparseDotInc(A, X, Y)


def test_simpyfunc_op():
    def my_func(t, x):
        return x

    argnames = ["output", "t", "x", "fn"]
    args = {"output": "outv", "t": "tv", "x": "xv", "fn": my_func}
    non_signals = ["fn"]
    _, sim = _test_operator_arg_attributes(
        SimPyFunc, argnames, args=args, non_signals=non_signals
    )
    assert str(sim) == "SimPyFunc{xv -> outv, fn='my_func'}"
