import numpy as np
import pytest

from nengo.utils.filter_design import (
    abcd_normalize,
    cont2discrete,
    normalize,
    ss2tf,
    ss2zpk,
    tf2ss,
    tf2zpk,
    zpk2ss,
    zpk2tf,
)


@pytest.mark.parametrize("dt", [1e-3, 1e-2, 1e-1])
def test_cont2discrete_zoh(dt, allclose):
    """test the function cont2discrete with zero-order hold"""
    taus = np.logspace(-np.log10(dt) - 1, np.log10(dt) + 3, 30)

    # test with lowpass filter, using analytic solution
    for tau in taus:
        num, den = [1], [tau, 1]
        d = -np.expm1(-dt / tau)
        num0, den0 = [0, d], [1, d - 1]
        num1, den1, _ = cont2discrete((num, den), dt)
        assert allclose(num0, num1)
        assert allclose(den0, den1)

    # test with alpha filter, using analytic solution
    for tau in taus:
        num, den = [1], [tau ** 2, 2 * tau, 1]
        a = dt / tau
        ea = np.exp(-a)
        num0 = [0, -a * ea + (1 - ea), ea * (a + ea - 1)]
        den0 = [1, -2 * ea, ea ** 2]
        num1, den1, _ = cont2discrete((num, den), dt)
        assert allclose(num0, num1)
        assert allclose(den0, den1)

    # test integrative filter, using analytic solution
    num, den = [1], [1, 0]
    num0, den0 = [0, dt], [1, -1]
    num1, den1, _ = cont2discrete((num, den), dt)
    assert allclose(num0, num1)
    assert allclose(den0, den1)


def test_cont2discrete_other_methods():
    dt = 1e-3

    # test with len(sys) == 3
    assert (
        repr(cont2discrete(([1], [1], [1]), dt))
        == "(array([1.0010005]), array([1.0010005]), 1.0, 0.001)"
    )

    # test with len(sys) == 5
    with pytest.raises(ValueError):
        cont2discrete(([1], [1], [1], [1], [1]), dt)

    # test method gbt and alpha None
    with pytest.raises(ValueError):
        cont2discrete(([1], [1], [1]), dt, method="gbt", alpha=None)

    # test method gbt and alpha invalid
    with pytest.raises(ValueError):
        cont2discrete(([1], [1], [1]), dt, method="gbt", alpha=2)

    # test method bilinear
    assert (
        repr(cont2discrete(([1], [1], [1]), dt, method="bilinear"))
        == "(array([1.0010005]), array([1.0010005]), 1.0, 0.001)"
    )

    # test method backward_diff
    assert (
        repr(cont2discrete(([1], [1], [1]), dt, method="backward_diff"))
        == "(array([1.001001]), array([1.001001]), 1.0, 0.001)"
    )

    # test bad method
    with pytest.raises(ValueError):
        cont2discrete(([1], [1], [1]), dt, method="not_a_method")


def test_tf2zpk():
    """test the tf2zpk function"""
    (a, b, c) = tf2zpk(1, 2)
    assert a.size == 0
    assert b.size == 0
    assert c == 0.5


def test_zpk2tf():
    """test the zpk2tf function"""
    (b, a) = zpk2tf([[1], [2]], [3], 3.0)
    z = [[1], [2]]
    z = np.atleast_1d(z)
    assert len(z.shape) > 1
    assert a.all() == 1
    assert b.all() == 1

    (b, a) = zpk2tf([0], [1], 3.0)
    z = [0]
    z = np.atleast_1d(z)
    assert len(z.shape) <= 1
    assert a.all() == 1
    assert b.all() == 0


@pytest.mark.filterwarnings("ignore:Badly conditioned filter coefficients")
def test_normalize():
    """test the normalize function, and exceptions"""
    with pytest.raises(ValueError):
        a = [[1], [2]]
        b = [1]
        normalize(b, a)
    with pytest.raises(ValueError):
        a = [1]
        b = [[[1]]]
        normalize(b, a)
    with pytest.warns(Warning):
        a = [1]
        b = [0]
        normalize(b, a)
    a = [1]
    b = [0, 0, 0]
    normalize(b, a)


def test_tf2ss():
    """test the tf2ss function, and exceptions"""
    with pytest.raises(ValueError):
        num = [1, 2]
        den = [1]
        tf2ss(num, den)


def test_abcd_normalize():
    """test the function abcd_normalize"""
    with pytest.raises(ValueError):
        abcd_normalize(None, None, None, None)


def test_ss2tf():
    """test the function ss2tf"""
    with pytest.raises(ValueError):
        ss2tf(None, None, None, None, 5)


def test_zpk2ss():
    """test the function zpk2ss"""

    predicted = (
        np.array([[3.0, -2.0], [1.0, 0.0]]),
        np.array([[1.0], [0.0]]),
        np.array([[3.0, -3.0]]),
        np.array([[0.0]]),
    )

    assert repr(zpk2ss([1], [1, 2], 3)) == repr(predicted)


def test_ss2zpk():
    """test the function ss2zpk"""
    predicted = (np.array([0.0]), np.array([1.0]), 1.0)
    assert repr(ss2zpk([1], [1], [1], [1])) == repr(predicted)
