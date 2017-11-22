import numpy as np
import pytest

from nengo.utils.expm import expm


def test_expm(rng):
    pytest.importorskip('scipy')
    import scipy.linalg
    for a in [np.eye(3), rng.randn(10, 10), -10 + rng.randn(10, 10)]:
        assert np.allclose(scipy.linalg.expm(a), expm(a))
