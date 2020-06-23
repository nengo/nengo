import numpy as np
import pytest

from nengo.utils.filter_design import (
    cont2discrete,
    tf2zpk,
    zpk2tf,
    normalize,
    tf2ss,
    _none_to_empty_2d,
    _shape_or_none,
    _restore,
    abcd_normalize,
    ss2tf,
    zpk2ss,
    ss2zpk,
)


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


def test_none_to_empty_2d():
    """test the _none_to_empty_2d function"""
    assert np.array_equal(np.zeros((0, 0)), _none_to_empty_2d(None))


def test_shape_or_none():
    """test the _shape_or_none function"""
    assert _shape_or_none(None) == (None,) * 2


def test_restore():
    """test the _restore function and errors"""

    class Test:
        shape = (0, 0)

    M = Test
    shape = (0, 0)
    _restore(M, shape) == np.zeros(shape)
    with pytest.raises(ValueError):
        M = np.array([1, 2])
        shape = (1, 2)
        _restore(M, shape)


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
        np.array([0.0]),
    )

    assert repr(zpk2ss([1], [1, 2], 3)) == repr(predicted)


def test_ss2zpk():
    """test the function ss2zpk"""
    predicted = (np.array([0.0]), np.array([1.0]), 1.0)
    assert repr(ss2zpk([1], [1], [1], [1])) == repr(predicted)


@pytest.mark.parametrize("dt", [1e-3, 1e-2, 1e-1])
def test_cont2discrete_zoh(dt, allclose):
    """test the function ss2zpk"""
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

    if dt == 1e-3:
        # test with len(sys) == 3
        assert (
            repr(cont2discrete(([1], [1], [1]), 1e-3))
            == "(array([1.0010005]), array([1.0010005]), 1.0, 0.001)"
        )

        # test with len(sys) == 5
        with pytest.raises(ValueError):
            cont2discrete(([1], [1], [1], [1], [1]), 1e-3)

        # test method gbt and alpha None
        with pytest.raises(ValueError):
            cont2discrete(([1], [1], [1]), 1e-3, method="gbt", alpha=None)

        # test method gbt and alpha invalid
        with pytest.raises(ValueError):
            cont2discrete(([1], [1], [1]), 1e-3, method="gbt", alpha=2)

        # test method bilinear
        assert (
            repr(cont2discrete(([1], [1], [1]), 1e-3, method="bilinear"))
            == "(array([1.0010005]), array([1.0010005]), 1.0, 0.001)"
        )

        # test method backward_diff
        assert (
            repr(cont2discrete(([1], [1], [1]), 1e-3, method="backward_diff"))
            == "(array([1.001001]), array([1.001001]), 1.0, 0.001)"
        )

        # test bad method
        with pytest.raises(ValueError):
            cont2discrete(([1], [1], [1]), 1e-3, method="not_a_method")
