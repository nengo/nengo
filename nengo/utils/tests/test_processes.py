from __future__ import absolute_import
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pytest

import nengo
from nengo.utils.distributions import Distribution
from nengo.utils.processes import (
    GaussianWhiteNoise, LimitedGaussianWhiteNoise, MarkovProcess,
    SampledProcess, WienerProcess)


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


def psd(values):
    return 2. * np.std(np.abs(np.fft.rfft(
        values, axis=1)), axis=0) / np.sqrt(np.asarray(values).shape[1])


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_white_noise(rms, rng):
    d = 500
    t = 100

    values = GaussianWhiteNoise(rms, dimensions=d).sample(
        dt=0.001, timesteps=t, rng=rng)
    assert_allclose(np.std(values), rms, rtol=0.02)
    assert_allclose(psd(values)[1:], rms, rtol=0.25)


class TestLimitedGaussianWhiteNoise(object):
    @pytest.mark.parametrize('rms', [0.5, 1, 100])
    def test_rms(self, rms, rng):
        d = 500
        t = 100
        dt = 0.001

        values = LimitedGaussianWhiteNoise(
            dt * t, d, rms=rms, dt=dt, rng=rng).sample(
            dt=dt, timesteps=t, rng=rng)
        assert_allclose(np.std(values, axis=0), rms, rtol=0.15)
        assert_allclose(psd(values)[1:], rms, rtol=0.2)

    @pytest.mark.parametrize('limit', [5, 50])
    def test_limit(self, limit, rng):
        rms = 0.5
        d = 500
        t = 1000
        dt = 0.001

        values = LimitedGaussianWhiteNoise(
            dt * t, d, limit=limit, rms=rms, dt=dt, rng=rng).sample(
            dt=dt, timesteps=t, rng=rng)
        assert_allclose(np.std(values, axis=0), rms, rtol=0.15)
        assert_almost_equal(psd(values)[np.fft.rfftfreq(t, dt) > limit], 0.)

    def test_unequal_dt(self, rng):
        d = 1
        t = 10
        dt = 0.001
        process = LimitedGaussianWhiteNoise(dt * t, d, dt=dt, rng=rng)
        with pytest.raises(AssertionError):
            process.sample(dt=1., rng=rng)

    def test_sampling_shape(self, rng):
        d = 2
        t = 10
        dt = 0.001
        process = LimitedGaussianWhiteNoise(dt * t, d, dt=dt, rng=rng)
        assert process.sample(dt, timesteps=None).shape == (2,)
        assert process.sample(dt, timesteps=1).shape == (2, 1)
        assert process.sample(dt, timesteps=5). shape == (2, 5)

    def test_sampling_out_of_signal_length(self, rng):
        d = 2
        t = 10
        dt = 0.001
        process = LimitedGaussianWhiteNoise(dt * t, d, dt=dt, rng=rng)
        values = process.sample(dt, timesteps=2 * t)
        assert_equal(values[:, :t], values[:, t:])

        values2 = np.empty((d, 2 * t))
        for i in range(2 * t):
            values2[:, i] = process.sample(dt, timesteps=None)
        assert_equal(values, values2)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
