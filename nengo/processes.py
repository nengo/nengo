from __future__ import absolute_import

import functools

import numpy as np

import nengo
from nengo.utils.compat import range


class StochasticProcess(object):
    """A stochastic process.

    A stochastic process is used to generate randomly varying signals.
    Unlike distributions, a stochastic process may maintain state between
    calls to ``sample``.

    Parameters
    ----------
    dist : Distribution
        The distribution from which to generate random samples.
    synapse : Synapse, optional
        Synapse object describing a filter to apply to the samples.
        If not provided, samples will be returned unfiltered.
    """

    def __init__(self, dist, synapse=None):
        self.dist = dist
        self.synapse = synapse

    def make_sample(self, dt, d=1, rng=np.random):
        """Samples the process and advances the time.

        Parameters
        ----------
        dt : float
            Timestep for each sample.
        d : int, optional
            The number of dimensions to return. Default: 1.
        rng : RandomState, optional
            Random number generator state.

        Returns
        -------
        function
            A sample function that, when called, returns a 1d array of
            length ``d``.
        """
        if self.synapse is not None:
            output = np.zeros(d)
            step = self.synapse.make_step(dt, output=output)
            return functools.partial(
                self.sample, self.dist, d=d, step=step, output=output, rng=rng)
        else:
            return functools.partial(self.sample_nostate, self.dist, d, rng)

    def f(self, dt=0.001, d=1, rng=np.random):
        """Return a function that can be passed to a Node."""
        sample_f = self.make_sample(dt=dt, d=d, rng=rng)

        def out_f(dummy_t, dummy_x=None):
            assert dummy_x is None, "Processes should not be given input."
            return sample_f()
        return out_f

    @staticmethod
    def sample_nostate(dist, d, rng=np.random):
        return dist.sample(n=1, d=d, rng=rng)[0]

    @staticmethod
    def sample(dist, d, step, output, rng=np.random):
        step(dist.sample(n=1, d=d, rng=rng)[0])
        return output


class BrownNoise(StochasticProcess):
    """A Brown noise process; i.e., a Weiner process."""

    def __init__(self):
        pass

    def make_sample(self, dt, d=1, rng=np.random):
        """Samples the process and advances the time.

        Parameters
        ----------
        dt : float
            Timestep for each sample.
        d : int, optional
            The number of dimensions to return. Default: 1.
        rng : RandomState, optional
            Random number generator state.

        Returns
        -------
        function
            A sample function that, when called, returns a 1d array of
            length ``d``.
        """
        dist = nengo.dists.Gaussian(0., 1. / np.sqrt(dt))
        output = np.zeros(d)
        step = nengo.LinearFilter([1], [1, 0]).make_step(
            dt, output, method='euler')
        return functools.partial(self.sample, dist, d, step, output, rng)


class WhiteNoise(StochasticProcess):
    """A low-pass filtered white noise process.

    Parameters
    ----------
    duration : float
        A white noise signal for this duration will be generated.
        Samples will repeat after this duration.
    high : float, optional
        The cut-off frequency of the low-pass filter, in Hz.
        If not specified, no filtering will be done.
    rms : float, optional
        The root mean square power of the filtered signal. Default: 0.5.
    """
    def __init__(self, duration, high=None, rms=0.5):
        self.duration = duration
        self.rms = rms
        self.high = high

    def make_sample(self, dt, d=1, rng=np.random):
        n_coefficients = int(np.ceil(self.duration / dt / 2.))
        shape = (d, n_coefficients + 1)
        sigma = self.rms * np.sqrt(0.5)
        coefficients = 1j * rng.normal(0., sigma, size=shape)
        coefficients += rng.normal(0., sigma, size=shape)
        coefficients[:, 0] = 0.
        coefficients[:, -1].imag = 0.
        if self.high is not None:
            set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > self.high
            coefficients[:, set_to_zero] = 0.
            power_correction = np.sqrt(
                1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
            if power_correction > 0.:
                coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)

        t = np.array([0])
        signal = np.fft.irfft(coefficients, axis=1)

        return functools.partial(self.sample, t=t, signal=signal)

    @staticmethod
    def sample(t, signal):
        t += 1
        sh = signal.shape[1] - 1
        if (t // sh) % 2 == 0:
            ix = t % sh
        else:
            ix = sh - t % sh
        return signal[:, np.atleast_1d(ix)[0]]


def sample(n, process, dt=0.001, d=None, rng=np.random):
    out = np.zeros(n) if d is None else np.zeros((n, d))
    sample_f = process.make_sample(dt=dt, d=1 if d is None else d, rng=rng)
    for i in range(n):
        out[i, ...] = sample_f()
    return out
