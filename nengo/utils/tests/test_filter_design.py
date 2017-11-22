import numpy as np
import pytest

from nengo.utils.filter_design import cont2discrete


@pytest.mark.parametrize('dt', [1e-3, 1e-2, 1e-1])
def test_cont2discrete_zoh(dt):
    taus = np.logspace(-np.log10(dt) - 1, np.log10(dt) + 3, 30)

    # test with lowpass filter, using analytic solution
    for tau in taus:
        num, den = [1], [tau, 1]
        d = -np.expm1(-dt / tau)
        num0, den0 = [0, d], [1, d - 1]
        num1, den1, _ = cont2discrete((num, den), dt)
        assert np.allclose(num0, num1)
        assert np.allclose(den0, den1)

    # test with alpha filter, using analytic solution
    for tau in taus:
        num, den = [1], [tau**2, 2*tau, 1]
        a = dt / tau
        ea = np.exp(-a)
        num0 = [0, -a*ea + (1 - ea), ea*(a + ea - 1)]
        den0 = [1, -2 * ea, ea**2]
        num1, den1, _ = cont2discrete((num, den), dt)
        assert np.allclose(num0, num1)
        assert np.allclose(den0, den1)

    # test integrative filter, using analytic solution
    num, den = [1], [1, 0]
    num0, den0 = [0, dt], [1, -1]
    num1, den1, _ = cont2discrete((num, den), dt)
    assert np.allclose(num0, num1)
    assert np.allclose(den0, den1)
