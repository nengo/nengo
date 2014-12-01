from __future__ import absolute_import
import numpy as np
import pytest

import nengo
from nengo.utils.distributions import Distribution
from nengo.utils.processes import (
    GaussianWhiteNoise, MarkovProcess, SampledProcess, WienerProcess)


class DistributionMock(Distribution):
    def __init__(self, retval):
        super(Distribution, self).__init__()
        self.retval = retval
        self.sample_calls = []

    def sample(self, n, d, rng=np.random):
        self.sample_calls.append((n, d, rng))
        return np.ones((n, d)) * self.retval


def test_sampled_process(rng):
    dist = DistributionMock(42)
    process = SampledProcess(dist, 2)
    samples = process.sample(0.001, 5, rng=rng)
    assert np.all(samples == 42 * np.ones((2, 5)))


def test_markov_process(rng):
    dist = DistributionMock(1)
    process = MarkovProcess(dist, dimensions=2, initial_state=[4, 5])
    samples = process.sample(0.1 * 0.1, 3, rng=rng)
    assert np.all(samples == np.array([[4.1, 4.2, 4.3], [5.1, 5.2, 5.3]]))


def test_wiener_process(rng):
    d = 5000
    t = 500
    dt = 0.001
    process = WienerProcess(d)
    samples = process.sample(dt, t, rng)

    expected_std = np.sqrt((np.arange(t) + 1) * dt)
    atol = 3. * expected_std / np.sqrt(d)
    assert np.all(np.abs(np.mean(samples, axis=0)) < atol)
    assert np.all(np.abs(np.std(samples, axis=0) - expected_std) < atol)


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_white_noise(rms, rng):
    d = 500
    t = 100

    values = GaussianWhiteNoise(rms, dimensions=d).sample(
        dt=0.001, timesteps=t, rng=rng)
    assert np.allclose(np.std(values), rms, rtol=0.02)

    assert np.allclose(
        np.std(np.abs(np.fft.rfft(values, axis=1)), axis=0) / np.sqrt(t) * 2.,
        rms, rtol=0.25)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
