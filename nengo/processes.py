from __future__ import absolute_import

import numpy as np

import nengo.utils.numpy as npext
from nengo.dists import DistributionParam, Gaussian
from nengo.params import (
    BoolParam, IntParam, NumberParam, Parameter, FrozenObject)
from nengo.synapses import LinearFilter, LinearFilterParam, Lowpass
from nengo.utils.compat import range


class Process(FrozenObject):
    """A general system with input, output, and state.

    Attributes
    ----------
    default_size_out : int
        Sets the default size out for nodes running this process. Also,
        if `d` isn't specified in `run` or `run_steps`, this will be used.
        Default: 1.
    default_dt : float
        If `dt` isn't specified in `run`, `run_steps`, `ntrange`, or `trange`,
        this will be used. Default: 0.001 (1 millisecond).
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """
    default_size_out = IntParam(low=0)
    default_dt = NumberParam(low=0, low_open=True)
    seed = IntParam(low=0, high=npext.maxint, optional=True)

    def __init__(self, seed=None):
        super(Process, self).__init__()
        self.default_size_out = 1
        self.default_dt = 0.001
        self.seed = seed

    def get_rng(self, rng):
        """Get a properly seeded independent RNG for the process step"""
        seed = rng.randint(npext.maxint) if self.seed is None else self.seed
        return np.random.RandomState(seed)

    def make_step(self, size_in, size_out, dt, rng):
        raise NotImplementedError("Process must implement `make_step` method.")

    def run_steps(self, n_steps, d=None, dt=None, rng=np.random):
        # TODO: allow running with input
        d = self.default_size_out if d is None else d
        dt = self.default_dt if dt is None else dt
        rng = self.get_rng(rng)
        step = self.make_step(0, d, dt, rng)
        output = np.zeros((n_steps, d))
        for i in range(n_steps):
            output[i] = step(i * dt)
        return output

    def run(self, t, d=None, dt=None, rng=np.random):
        # TODO: allow running with input
        dt = self.default_dt if dt is None else dt
        n_steps = int(np.round(float(t) / dt))
        return self.run_steps(n_steps, d=d, dt=dt, rng=rng)

    def ntrange(self, n_steps, dt=None):
        dt = self.default_dt if dt is None else dt
        return dt * np.arange(1, n_steps + 1)

    def trange(self, t, dt=None):
        dt = self.default_dt if dt is None else dt
        n_steps = int(np.round(float(t) / dt))
        return self.ntrange(n_steps, dt=dt)


class WhiteNoise(Process):
    """Full-spectrum white noise process.

    Parameters
    ----------
    dist : Distribution, optional
        The distribution to draw samples from.
        Default: Gaussian(mean=0, std=1)
    scale : bool, optional
        Whether to scale the white noise for integration. Integrating white
        noise requires using a time constant of `sqrt(dt)` instead of `dt`
        on the noise term [1]_, to ensure the magnitude of the integrated
        noise does not change with `dt`. Defaults to True.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.

    References
    ----------
    .. [1] Gillespie, D.T. (1996) Exact numerical simulation of the Ornstein-
       Uhlenbeck process and its integral. Phys. Rev. E 54, pp. 2084-91.
    """

    dist = DistributionParam()
    scale = BoolParam()

    def __init__(self, dist=Gaussian(mean=0, std=1), scale=True, seed=None):
        super(WhiteNoise, self).__init__(seed=seed)
        self.dist = dist
        self.scale = scale

    def __repr__(self):
        return "%s(%r, scale=%r)" % (
            self.__class__.__name__, self.dist, self.scale)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        # ^ need sqrt(dt) when integrating, so divide by sqrt(dt) here,
        #   since dt / sqrt(dt) = sqrt(dt).

        def step(t):
            x = dist.sample(n=1, d=size_out, rng=rng)[0]
            return alpha * x if scale else x

        return step


class FilteredNoise(Process):
    """Filtered white noise process.

    This process takes white noise and filters it using the provided synapse.

    Parameters
    ----------
    synapse : Synapse, optional
        The synapse to use to filter the noise. Default: Lowpass(tau=0.005)
    synapse_kwargs : dict, optional
        Arguments to pass to `synapse.make_step`.
    dist : Distribution, optional
        The distribution used to generate the white noise.
        Default: Gaussian(mean=0, std=1)
    scale : bool, optional
        Whether to scale the white noise for integration, making the output
        signal invariant to `dt`. Defaults to True.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """

    synapse = LinearFilterParam()
    dist = DistributionParam()
    scale = BoolParam()

    def __init__(self, synapse=Lowpass(tau=0.005), synapse_kwargs={},
                 dist=Gaussian(mean=0, std=1), scale=True, seed=None):
        super(FilteredNoise, self).__init__(seed=seed)
        self.synapse = synapse
        self.synapse_kwargs = synapse_kwargs
        self.dist = dist
        self.scale = scale

    def __repr__(self):
        return "%s(synapse=%r, dist=%r, scale=%r)" % (
            self.__class__.__name__, self.synapse, self.dist, self.scale)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        output = np.zeros(size_out)
        filter_step = self.synapse.make_step(dt, output, **self.synapse_kwargs)

        def step(t):
            x = dist.sample(n=1, d=size_out, rng=rng)[0]
            if scale:
                x *= alpha
            filter_step(x)
            return output

        return step


class BrownNoise(FilteredNoise):
    """Brown noise process (aka Brownian noise, red noise, Wiener process).

    This process is the integral of white noise.

    Parameters
    ----------
    dist : Distribution
        The distribution used to generate the white noise.
        Default: Gaussian(mean=0, std=1)
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """
    def __init__(self, dist=Gaussian(mean=0, std=1), seed=None):
        super(BrownNoise, self).__init__(
            synapse=LinearFilter([1], [1, 0]),
            synapse_kwargs=dict(method='euler'),
            dist=dist, seed=seed)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.dist)


class WhiteSignal(Process):
    """An ideal low-pass filtered white noise process.

    This signal is created in the frequency domain, and designed to have
    exactly equal power at all frequencies below the cut-off frequency,
    and no power above the cut-off.

    The signal is naturally periodic, so it can be used beyond its period
    while still being continuous with continuous derivatives.

    Parameters
    ----------
    period : float
        A white noise signal with this period will be generated.
        Samples will repeat after this duration.
    high : float, optional
        The cut-off frequency of the low-pass filter, in Hz.
        If not specified, no filtering will be done.
    rms : float, optional
        The root mean square power of the filtered signal. Default: 0.5.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """
    period = NumberParam(low=0, low_open=True)
    high = NumberParam(low=0, low_open=True, optional=True)
    rms = NumberParam(low=0, low_open=True)

    def __init__(self, period, high=None, rms=0.5, seed=None):
        super(WhiteSignal, self).__init__(seed=seed)
        self.period = period
        self.high = high
        self.rms = rms

    def __repr__(self):
        return "%s(period=%r, high=%r, rms=%r)" % (
            self.__class__.__name__, self.period, self.high, self.rms)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        d = size_out

        n_coefficients = int(np.ceil(self.period / dt / 2.))
        shape = (n_coefficients + 1, d)
        sigma = self.rms * np.sqrt(0.5)
        coefficients = 1j * rng.normal(0., sigma, size=shape)
        coefficients += rng.normal(0., sigma, size=shape)
        coefficients[0] = 0.
        coefficients[-1].imag = 0.
        if self.high is not None:
            set_to_zero = npext.rfftfreq(2 * n_coefficients, d=dt) > self.high
            coefficients[set_to_zero] = 0.
            power_correction = np.sqrt(
                1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
            if power_correction > 0.:
                coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)
        signal = np.fft.irfft(coefficients, axis=0)

        def step(t):
            i = int(round(t / dt))
            return signal[i % signal.shape[0]]

        return step


class ProcessParam(Parameter):
    """Must be a Process."""

    def validate(self, instance, process):
        super(ProcessParam, self).validate(instance, process)
        if process is not None and not isinstance(process, Process):
            raise ValueError("Must be Process (got type '%s')" % (
                process.__class__.__name__))

        return process
