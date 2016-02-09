import numpy as np

import nengo.utils.numpy as npext
from nengo.base import Process
from nengo.dists import DistributionParam, Gaussian
from nengo.params import BoolParam, NdarrayParam, NumberParam
from nengo.synapses import LinearFilter, LinearFilterParam, Lowpass


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

    dist = DistributionParam('dist')
    scale = BoolParam('scale')

    def __init__(self, dist=Gaussian(mean=0, std=1), scale=True, seed=None):
        super(WhiteNoise, self).__init__(seed=seed)
        self.dist = dist
        self.scale = scale

    def __repr__(self):
        return "%s(%r, scale=%r)" % (
            self.__class__.__name__, self.dist, self.scale)

    def make_step(self, shape_in, shape_out, dt, rng):
        assert shape_in == (0,)
        assert len(shape_out) == 1

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        # ^ need sqrt(dt) when integrating, so divide by sqrt(dt) here,
        #   since dt / sqrt(dt) = sqrt(dt).

        def step_whitenoise(t):
            x = dist.sample(n=1, d=shape_out[0], rng=rng)[0]
            return alpha * x if scale else x

        return step_whitenoise


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

    synapse = LinearFilterParam('synapse')
    dist = DistributionParam('dist')
    scale = BoolParam('scale')

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

    def make_step(self, shape_in, shape_out, dt, rng):
        assert shape_in == (0,)
        assert len(shape_out) == 1

        dist = self.dist
        scale = self.scale
        alpha = 1. / np.sqrt(dt)
        filter_step = self.synapse.make_step(
            shape_out, shape_out, dt, None, **self.synapse_kwargs)

        def step_filterednoise(t):
            x = dist.sample(n=1, d=shape_out[0], rng=rng)[0]
            if scale:
                x *= alpha
            return filter_step(t, x)

        return step_filterednoise


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
    period = NumberParam('period', low=0, low_open=True)
    high = NumberParam('high', low=0, low_open=True, optional=True)
    rms = NumberParam('rms', low=0, low_open=True)

    def __init__(self, period, high=None, rms=0.5, seed=None):
        super(WhiteSignal, self).__init__(seed=seed)
        self.period = period
        self.high = high
        self.rms = rms

    def __repr__(self):
        return "%s(period=%r, high=%r, rms=%r)" % (
            self.__class__.__name__, self.period, self.high, self.rms)

    def make_step(self, shape_in, shape_out, dt, rng):
        assert shape_in == (0,)

        n_coefficients = int(np.ceil(self.period / dt / 2.))
        shape = (n_coefficients + 1,) + shape_out
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

        def step_whitesignal(t):
            i = int(round(t / dt))
            return signal[i % signal.shape[0]]

        return step_whitesignal


class PresentInput(Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for `presentation_time` seconds.
    """
    inputs = NdarrayParam('inputs', shape=('...',))
    presentation_time = NumberParam('presentation_time', low=0, low_open=True)

    def __init__(self, inputs, presentation_time):
        self.inputs = inputs
        self.presentation_time = presentation_time
        super(PresentInput, self).__init__(
            default_size_out=self.inputs[0].size)

    def make_step(self, shape_in, shape_out, dt, rng):
        assert shape_in == (0,)
        assert shape_out == (self.inputs[0].size,)

        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)

        def step_presentinput(t):
            i = int((t-dt) / presentation_time + 1e-7)
            return inputs[i % n]

        return step_presentinput
