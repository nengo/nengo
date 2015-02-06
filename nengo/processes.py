from __future__ import absolute_import

import numpy as np

import nengo.utils.numpy as npext
from nengo.dists import DistributionParam, Gaussian
from nengo.params import BoolParam, IntParam, NumberParam, Parameter
from nengo.synapses import LinearFilter, LinearFilterParam, Lowpass
from nengo.utils.compat import range


class Process(object):
    """A general system with input, output, and state.

    Attributes
    ----------
    default_size_out : int
        If `d` isn't specified in `run` or `run_steps`, this will be used.
        Default: 1.
    default_dt : float
        If `dt` isn't specified in `run`, `run_steps`, `ntrange`, or `trange`,
        this will be used. Default: 0.001 (1 millisecond).
    """
    default_size_out = IntParam(low=0)
    default_dt = NumberParam(low=0, low_open=True)

    def __init__(self):
        self.default_size_out = 1
        self.default_dt = 0.001

    def make_step(self, size_in, size_out, dt, rng):
        raise NotImplementedError("Process must implement `make_step` method.")

    def run_steps(self, n_steps, d=None, dt=None, rng=np.random):
        # TODO: allow running with input
        d = self.default_size_out if d is None else d
        dt = self.default_dt if dt is None else dt
        step = self.make_step(0, d, dt, rng)
        output = np.zeros((n_steps, d))
        for i in range(n_steps):
            output[i] = step()
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

    References
    ----------
    .. [1] Gillespie, D.T. (1996) Exact numerical simulation of the Ornstein-
       Uhlenbeck process and its integral. Phys. Rev. E 54, pp. 2084-91.
    """

    dist = DistributionParam()
    scale = BoolParam()

    def __init__(self, dist=None, scale=True):
        super(WhiteNoise, self).__init__()
        self.dist = Gaussian(mean=0, std=1) if dist is None else dist
        self.scale = scale

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        # ^ need sqrt(dt) when integrating, so divide by sqrt(dt) here,
        #   since dt / sqrt(dt) = sqrt(dt).

        # separate RNG for simulation for step order independence
        sim_rng = np.random.RandomState(rng.randint(npext.maxint))

        def step():
            x = dist.sample(n=1, d=size_out, rng=sim_rng)[0]
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
    """

    synapse = LinearFilterParam()
    dist = DistributionParam()
    scale = BoolParam()

    def __init__(self, synapse=None, synapse_kwargs={}, dist=None, scale=True):
        super(FilteredNoise, self).__init__()
        self.synapse = Lowpass(tau=0.005) if synapse is None else synapse
        self.synapse_kwargs = synapse_kwargs
        self.dist = Gaussian(mean=0, std=1) if dist is None else dist
        self.scale = scale

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        output = np.zeros(size_out)
        filter_step = self.synapse.make_step(dt, output, **self.synapse_kwargs)

        # separate RNG for simulation for step order independence
        sim_rng = np.random.RandomState(rng.randint(npext.maxint))

        def step():
            x = dist.sample(n=1, d=size_out, rng=sim_rng)[0]
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
    """
    def __init__(self, dist=None):
        super(BrownNoise, self).__init__(
            synapse=LinearFilter([1], [1, 0]),
            synapse_kwargs=dict(method='euler'),
            dist=Gaussian(mean=0, std=1) if dist is None else dist)


class WhiteSignal(Process):
    """An ideal low-pass filtered white noise process.

    This signal is created in the frequency domain, and designed to have
    exactly equal power at all frequencies below the cut-off frequency,
    and no power above the cut-off.

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
        super(WhiteSignal, self).__init__()
        self.duration = duration
        self.high = high
        self.rms = rms

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        d = size_out

        n_coefficients = int(np.ceil(self.duration / dt / 2.))
        shape = (d, n_coefficients + 1)
        sigma = self.rms * np.sqrt(0.5)
        coefficients = 1j * rng.normal(0., sigma, size=shape)
        coefficients += rng.normal(0., sigma, size=shape)
        coefficients[:, 0] = 0.
        coefficients[:, -1].imag = 0.
        if self.high is not None:
            set_to_zero = npext.rfftfreq(2 * n_coefficients, d=dt) > self.high
            coefficients[:, set_to_zero] = 0.
            power_correction = np.sqrt(
                1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
            if power_correction > 0.:
                coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)

        t = np.array(0)
        signal = np.fft.irfft(coefficients, axis=1)
        sh = signal.shape[1] - 1

        def step():
            t[...] += 1
            # once t >= sh, go through signal backwards (i.e. reflect it)
            ix = t % sh if (t // sh) % 2 == 0 else sh - t % sh
            return signal[:, ix]

        return step


class ProcessParam(Parameter):
    """Must be a Process."""

    def validate(self, instance, process):
        super(ProcessParam, self).validate(instance, process)
        if process is not None and not isinstance(process, Process):
            raise ValueError("Must be Process (got type '%s')" % (
                process.__class__.__name__))

        return process
