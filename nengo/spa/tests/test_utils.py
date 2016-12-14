import numpy as np
import pytest

from nengo.utils.compat import range
from nengo.utils.numpy import array_hash
from nengo.spa.utils import cyclic_vector


def circconvk(a, b, k=1, invert_a=False, invert_b=False, axis=-1):
    """Convolve a with b k times."""
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    return np.fft.ifft(A * B**k, axis=axis).real


@pytest.mark.parametrize('d, k', [(5, 3), (16, 4), (128, 13)])
def test_cyclic_vector(d, k, rng):
    n = 10
    a = rng.normal(scale=1./np.sqrt(d), size=(n, d))
    b = cyclic_vector(d, k, n=n)
    c = circconvk(a, b, k)
    assert np.allclose(a, c, atol=1e-7), (a, c)


@pytest.mark.parametrize('d, k', [(7, 5), (8, 4)])
def test_cyclic_vector_dist(d, k, rng):
    n = 1000
    v = cyclic_vector(d, k, n=n, rng=rng)
    v = np.round(v, decimals=8)
    v[np.abs(v) < 1e-8] = 0

    hashmap = {}
    counts = {}
    for i in range(n):
        vi = v[i]
        hi = array_hash(vi)
        hashmap.setdefault(hi, vi)
        if hi not in counts:
            counts[hi] = 1
        else:
            counts[hi] += 1

    # check that all generated values are cyclic
    for b in hashmap.values():
        a = rng.normal(scale=1./np.sqrt(d), size=d)
        c = circconvk(a, b, k)
        assert np.allclose(a, c, atol=1e-7), (a, c)

    # check that the std. dev. of the counts is roughly a Binomial dist.
    values = np.array(list(counts.values()))
    p = 1. / len(values)
    std = np.sqrt(n * p * (1 - p))
    assert np.allclose(values.std(), std, rtol=0.3, atol=0)


def test_cyclic_vector_errors(rng):
    with pytest.raises(ValueError):
        cyclic_vector(2, 1, rng=rng)
    with pytest.raises(ValueError):
        cyclic_vector(1, 2, rng=rng)
    with pytest.raises(ValueError):
        cyclic_vector(2, 2, rng=rng)

    a = cyclic_vector(3, 2, rng=rng)
    assert np.allclose(circconvk(a, a), [1, 0, 0])
