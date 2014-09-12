import numpy as np
import pytest

import nengo
import nengo.utils.distributions as dists
import nengo.utils.numpy as npext


@pytest.mark.parametrize("low,high", [(-2, -1), (-1, 1), (1, 2), (1, -1)])
def test_uniform(low, high):
    n = 100
    dist = dists.Uniform(low, high)
    samples = dist.sample(n, rng=np.random.RandomState(1))
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
        samples = dist.sample(n, rng=np.random.RandomState(1))
        assert abs(np.mean(samples) - mean) < std * 0.1
        assert abs(np.std(samples) - std) < 1


@pytest.mark.parametrize("dimensions", [0, 1, 2, 5])
def test_hypersphere(dimensions):
    n = 100 * dimensions
    if dimensions < 1:
        with pytest.raises(ValueError):
            dist = dists.UniformHypersphere().sample(1, dimensions)
    else:
        dist = dists.UniformHypersphere()
        samples = dist.sample(n, dimensions, np.random.RandomState(1))
        assert samples.shape == (n, dimensions)
        assert np.allclose(
            np.mean(samples, axis=0), np.zeros(dimensions), atol=0.1)
        hist, _ = np.histogramdd(samples, bins=5)
        assert np.allclose(hist - np.mean(hist), 0, atol=0.1 * n)


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_hypersphere_surface(dimensions):
    n = 100 * dimensions
    dist = dists.UniformHypersphere(surface=True)
    samples = dist.sample(n, dimensions, np.random.RandomState(1))
    assert samples.shape == (n, dimensions)
    assert np.allclose(npext.norm(samples, axis=1), 1)
    assert np.allclose(
        np.mean(samples, axis=0), np.zeros(dimensions), atol=0.1)


@pytest.mark.parametrize("weights", [None, [5, 1, 2, 9], [3, 2, 1, 0]])
def test_choice(weights):
    n = 2000
    choices = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    N = len(choices)

    dist = dists.Choice(choices, weights=weights)
    # If d is passed, it has to match
    with pytest.raises(ValueError):
        dist.sample(n, d=4, rng=np.random.RandomState(5))
    sample = dist.sample(n, rng=np.random.RandomState(5))
    tsample, tchoices = list(map(tuple, sample)), list(map(tuple, choices))

    # check that frequency of choices matches weights
    inds = [tchoices.index(s) for s in tsample]
    hist, bins = np.histogram(inds, bins=np.linspace(-0.5, N - 0.5, N + 1))
    p_empirical = hist / float(hist.sum())
    p = np.ones(N) / N if dist.p is None else dist.p
    sterr = 1. / np.sqrt(n)  # expected maximum standard error
    assert np.allclose(p, p_empirical, atol=sterr)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
