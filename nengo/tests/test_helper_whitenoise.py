import numpy as np
import pytest

import nengo
from nengo.helpers import whitenoise


def test_rms():
    t = np.linspace(0, 1, 1000)

    for rms_desired in [0, 0.5, 1, 100]:
        f = whitenoise(1, 100, rms=rms_desired)
        rms = np.sqrt(np.mean([f(tt) ** 2 for tt in t]))
        assert np.allclose(rms, rms_desired, atol=.1, rtol=.01)


def test_array():
    rms_desired = 0.5
    f = whitenoise(1, 5, rms=rms_desired, dimensions=5)

    t = np.linspace(0, 1, 1000)
    data = np.array([f(tt) for tt in t])
    rms = np.sqrt(np.mean(data**2, axis=0))
    assert np.allclose(rms, rms_desired, atol=.1, rtol=.01)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
