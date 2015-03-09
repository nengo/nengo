from __future__ import absolute_import

import numpy as np
import pytest

import nengo
from nengo.dists import Distribution, Gaussian
from nengo.processes import BrownNoise, StochasticProcess, WhiteNoise
from nengo.utils.numpy import rfftfreq


class DistributionMock(Distribution):
    def __init__(self, retval):
        super(Distribution, self).__init__()
        self.retval = retval
        self.sample_calls = []

    def sample(self, n, d, rng=np.random):
        self.sample_calls.append((n, d, rng))
        return np.ones((n, d)) * self.retval


def test_stochasticprocess(rng):
    dist = DistributionMock(42)
    process = StochasticProcess(dist)
    samples = nengo.processes.sample(5, process, d=2, rng=rng)
    assert np.all(samples == 42 * np.ones((5, 2)))
    assert nengo.processes.sample(5, process).shape == (5,)
    assert nengo.processes.sample(1, process, d=1).shape == (1, 1)
    assert nengo.processes.sample(2, process, d=3).shape == (2, 3)


def test_brownnoise(rng, plt):
    d = 5000
    t = 500
    dt = 0.001
    samples = nengo.processes.sample(t, BrownNoise(), dt=dt, d=d, rng=rng)

    trange = np.arange(1, t + 1) * dt
    expected_std = np.sqrt((np.arange(t) + 1) * dt)
    atol = 3.5 * expected_std / np.sqrt(d)

    plt.subplot(2, 1, 1)
    plt.title("Five Brown noise signals")
    plt.plot(trange, samples[:, :5])
    plt.subplot(2, 1, 2)
    plt.ylabel("Standard deviation")
    plt.plot(trange, np.abs(np.std(samples, axis=1)), label="Actual")
    plt.plot(trange, expected_std, label="Expected")
    plt.legend(loc='best')

    assert np.all(np.abs(np.mean(samples, axis=1)) < atol)
    assert np.all(np.abs(np.std(samples, axis=1) - expected_std) < atol)


def psd(values, dt=0.001):
    freq = rfftfreq(values.shape[0], d=dt)
    power = 2. * np.std(np.abs(np.fft.rfft(
        values, axis=0)), axis=1) / np.sqrt(values.shape[0])
    return freq, power


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_whitenoise(rms, rng, plt):
    d = 500
    t = 100
    dt = 0.001

    process = StochasticProcess(Gaussian(0., rms))

    values = nengo.processes.sample(t, process, dt=dt, d=d, rng=rng)
    freq, val_psd = psd(values)

    trange = np.arange(1, t + 1) * dt
    plt.subplot(2, 1, 1)
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    assert np.allclose(np.std(values), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.2)


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_whitenoise_rms(rms, rng, plt):
    d = 500
    t = 100
    dt = 0.001

    process = WhiteNoise(dt * t, rms=rms)
    values = nengo.processes.sample(t, process, dt=dt, d=d, rng=rng)
    freq, val_psd = psd(values)

    trange = np.arange(1, t + 1) * dt
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    assert np.allclose(np.std(values), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.35)


@pytest.mark.parametrize('high', [5, 50])
def test_whitenoise_high(high, rng, plt):
    rms = 0.5
    d = 500
    t = 1000
    dt = 0.001

    process = WhiteNoise(dt * t, high, rms=rms)
    values = nengo.processes.sample(t, process, dt=dt, d=d, rng=rng)
    freq, val_psd = psd(values)

    trange = np.arange(1, t + 1) * dt
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[rfftfreq(t, dt) > high] < rms * 0.5)


def test_whitenoise_dt(rng, plt):
    rms = 0.5
    high = 10
    d = 500
    t = 100
    dt = 0.01

    process = WhiteNoise(dt * t, high, rms=rms)
    values = nengo.processes.sample(t, process, dt=dt, d=d, rng=rng)
    freq, val_psd = psd(values, dt=dt)

    trange = np.arange(1, t + 1) * dt
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[rfftfreq(t, dt) > high] < rms * 0.5)


def test_sampling_shape():
    process = WhiteNoise(0.1)
    assert nengo.processes.sample(1, process).shape == (1,)
    assert nengo.processes.sample(5, process, d=1).shape == (5, 1)
    assert nengo.processes.sample(1, process, d=2). shape == (1, 2)
