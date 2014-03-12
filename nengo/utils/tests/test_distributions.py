import numpy as np
import numpy.linalg  # noqa: F401
import pytest

import nengo
import nengo.utils.distributions as dists


@pytest.mark.parametrize("low,high", [(-2, -1), (-1, 1), (1, 2), (1, -1)])
def test_uniform(low, high):
    n = 100
    dist = dists.Uniform(low, high)
    samples = dist.sample(n, np.random.RandomState(1))
    if low < high:
        assert np.all(samples >= low)
        assert np.all(samples < high)
    else:
        assert np.all(samples <= low)
        assert np.all(samples > high)
    hist, _ = np.histogram(samples, bins=5)
    assert np.allclose(hist - np.mean(hist), 0, atol=0.1 * n)


@pytest.mark.parametrize("mean,std", [(0, 1), (0, 0), (10, 2)])
def test_gaussian(mean, std):
    n = 100
    if std <= 0:
        with pytest.raises(ValueError):
            dist = dists.Gaussian(mean, std)
    else:
        dist = dists.Gaussian(mean, std)
        samples = dist.sample(n, np.random.RandomState(1))
        assert abs(np.mean(samples) - mean) < std * 0.1
        assert abs(np.std(samples) - std) < 1


@pytest.mark.parametrize("dimensions", [0, 1, 2, 5])
def test_hypersphere(dimensions):
    n = 100 * dimensions
    if dimensions < 1:
        with pytest.raises(ValueError):
            dist = dists.UniformHypersphere(dimensions)
    else:
        dist = dists.UniformHypersphere(dimensions)
        samples = dist.sample(n, np.random.RandomState(1))
        assert samples.shape == (n, dimensions)
        assert np.allclose(
            np.mean(samples, axis=0), np.zeros(dimensions), atol=0.1)
        hist, _ = np.histogramdd(samples, bins=5)
        assert np.allclose(hist - np.mean(hist), 0, atol=0.1 * n)


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_hypersphere_surface(dimensions):
    n = 100 * dimensions
    dist = dists.UniformHypersphere(dimensions, surface=True)
    samples = dist.sample(n, np.random.RandomState(1))
    assert samples.shape == (n, dimensions)
    assert np.allclose(np.linalg.norm(samples, axis=1), 1)
    assert np.allclose(
        np.mean(samples, axis=0), np.zeros(dimensions), atol=0.1)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
