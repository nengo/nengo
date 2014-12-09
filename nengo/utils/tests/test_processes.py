from __future__ import absolute_import
import numpy as np
import pytest

import nengo
from nengo.utils.distributions import Distribution, Gaussian
from nengo.utils.processes import (
    BrownNoise, MarkovProcess, SampledProcess, WhiteNoise)


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
    process = SampledProcess(dist)
    samples = process.sample(0.001, 5, d=2, rng=rng)
    assert np.all(samples == 42 * np.ones((5, 2)))


def test_markov_process(rng):
    dist = DistributionMock(1)
    process = MarkovProcess(dist)
    state = np.array([4., 5.])
    samples = process.sample(0.1 * 0.1, 2, state=state, d=2, rng=rng)
    assert np.all(samples == np.array([[4.1, 5.1], [4.2, 5.2]]))
    samples = process.sample(0.1 * 0.1, 1, state=state, d=2, rng=rng)
    assert np.all(samples == np.array([[4.3, 5.3]]))


def test_brown_noise(rng, plt):
    d = 5000
    t = 500
    dt = 0.001
    process = BrownNoise()
    state = process.initial_state(dt, d, rng)
    samples = process.sample(dt, t, d=d, rng=rng, **state)

    trange = np.arange(1, t + 1) * dt
    plt.title("Five Brown noise signals")
    plt.plot(trange, samples[:, :5])

    expected_std = np.sqrt((np.arange(t) + 1) * dt)
    atol = 3. * expected_std / np.sqrt(d)
    assert np.all(np.abs(np.mean(samples, axis=1)) < atol)
    assert np.all(np.abs(np.std(samples, axis=1) - expected_std) < atol)


def psd(values, dt=0.001):
    freq = np.fft.rfftfreq(values.shape[0], d=dt)
    power = 2. * np.std(np.abs(np.fft.rfft(
        values, axis=0)), axis=1) / np.sqrt(values.shape[0])
    return freq, power


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_white_noise(rms, rng, plt):
    d = 500
    t = 100
    dt = 0.001

    values = SampledProcess(Gaussian(0., rms)).sample(dt=dt, n=t, d=d, rng=rng)
    freq, val_psd = psd(values)

    trange = np.arange(1, t + 1) * dt
    plt.subplot(2, 1, 1)
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.saveas = ('utils.test_processes.test_gaussian_white_noise_rms%.1f.pdf'
                  % rms)

    assert np.allclose(np.std(values), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.2)


class TestWhiteNoise(object):
    @pytest.mark.parametrize('rms', [0.5, 1, 100])
    def test_rms(self, rms, rng, plt):
        d = 500
        t = 100
        dt = 0.001

        wn = WhiteNoise(duration=dt * t, rms=rms)
        state = wn.initial_state(dt, d, rng=rng)
        values = wn.sample(dt=dt, n=t, d=d, rng=rng, **state)
        freq, val_psd = psd(values)

        trange = np.arange(1, t + 1) * dt
        plt.subplot(2, 1, 1)
        plt.title("First two D of white noise process, rms=%.1f" % rms)
        plt.plot(trange, values[:, :2])
        plt.xlim(right=trange[-1])
        plt.subplot(2, 1, 2)
        plt.title("Power spectrum")
        plt.plot(freq, val_psd, drawstyle='steps')
        plt.saveas = ('utils.test_processes.TestWhiteNoise.test_rms_%.1f.pdf'
                      % rms)

        assert np.allclose(np.std(values), rms, rtol=0.02)
        assert np.allclose(val_psd[1:-1], rms, rtol=0.35)

    @pytest.mark.parametrize('high', [5, 50])
    def test_high(self, high, rng, plt):
        rms = 0.5
        d = 500
        t = 1000
        dt = 0.001

        process = WhiteNoise(duration=dt * t, rms=rms, high=high)
        state = process.initial_state(dt, d, rng=rng)
        values = process.sample(dt=dt, n=t, d=d, rng=rng, **state)
        freq, val_psd = psd(values)

        trange = np.arange(1, t + 1) * dt
        plt.subplot(2, 1, 1)
        plt.title("First two D of white noise process, limit=%d Hz" % high)
        plt.plot(trange, values[:, :2])
        plt.xlim(right=trange[-1])
        plt.subplot(2, 1, 2)
        plt.title("Power spectrum")
        plt.plot(freq, val_psd, drawstyle='steps')
        plt.xlim(right=high * 2.0)
        plt.saveas = ('utils.test_processes.TestWhiteNoise.test_high_%d.pdf'
                      % high)

        assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
        assert np.allclose(val_psd[np.fft.rfftfreq(t, dt) > high], 0.)

    def test_unequal_dt(self, rng):
        d = 1
        t = 10
        dt = 0.001
        process = WhiteNoise(duration=dt * t)
        state = process.initial_state(dt, d, rng=rng)
        with pytest.raises(AssertionError):
            process.sample(dt=1., n=1, rng=rng, **state)

    def test_sampling_shape(self, rng):
        d = 2
        t = 10
        dt = 0.001

        process = WhiteNoise(duration=dt * t)
        state = process.initial_state(dt, d, rng=rng)

        assert process.sample(dt, n=1, d=None, **state).shape == (1,)
        assert process.sample(dt, n=5, d=1, **state).shape == (5, 1)
        assert process.sample(dt, n=1, d=2, **state). shape == (1, 2)

    def test_sampling_out_of_signal_length(self, rng):
        d = 2
        t = 10
        dt = 0.001
        process = WhiteNoise(duration=dt * t)
        state = process.initial_state(dt=dt, d=d, rng=rng)
        values = process.sample(dt, n=2 * t, d=d, **state)
        assert np.all(values[:t] == values[t:])

        values2 = np.empty((2 * t, d))
        for i in range(2 * t):
            values2[i] = process.sample(dt, n=1, d=d, **state)[0]
        assert np.all(values == values2)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
