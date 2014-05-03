import logging
import pytest

import numpy as np

import nengo
from nengo.utils.filter_design import expm, cont2discrete

logger = logging.getLogger(__name__)


@pytest.mark.optional
def test_expm():
    import scipy.linalg as linalg
    rng = np.random.RandomState(10)

    for a in [np.eye(3), rng.randn(10, 10)]:
        assert np.allclose(linalg.expm(a), expm(a))


@pytest.mark.optional
def test_cont2discrete():
    import scipy.signal

    dt = 1e-3
    tau = 0.03
    num, den = [1], [tau**2, 2*tau, 1]
    num0, den0, _ = scipy.signal.cont2discrete((num, den), dt)
    num1, den1, _ = cont2discrete((num, den), dt)
    assert np.allclose(num0, num1)
    assert np.allclose(den0, den1)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
