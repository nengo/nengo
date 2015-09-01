from __future__ import absolute_import

import numpy as np

import nengo.utils.numpy as npext
from nengo.dists import DistributionParam, Gaussian
from nengo.params import (
    BoolParam, EnumParam, IntParam, NdarrayParam, NumberParam, TupleParam,
    Parameter, FrozenObject)
from nengo.synapses import LinearFilter, LinearFilterParam, Lowpass
from nengo.utils.compat import range


class Process(FrozenObject):
    """A general system with input, output, and state.

    Attributes
    ----------
    default_size_in : int
        Sets the default size in for nodes using this process. Default: 0.
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
    default_size_in = IntParam(low=0)
    default_size_out = IntParam(low=0)
    default_dt = NumberParam(low=0, low_open=True)
    seed = IntParam(low=0, high=npext.maxint, optional=True)

    def __init__(self, default_size_in=0, default_size_out=1, seed=None):
        super(Process, self).__init__()
        self.default_size_in = default_size_in
        self.default_size_out = default_size_out
        self.default_dt = 0.001
        self.seed = seed

    def get_rng(self, rng):
        """Get a properly seeded independent RNG for the process step"""
        seed = rng.randint(npext.maxint) if self.seed is None else self.seed
        return np.random.RandomState(seed)

    def make_step(self, size_in, size_out, dt, rng):
        raise NotImplementedError("Process must implement `make_step` method.")

    def run_steps(self, n_steps, d=None, dt=None, rng=np.random):
        d = self.default_size_out if d is None else d
        dt = self.default_dt if dt is None else dt
        rng = self.get_rng(rng)
        step = self.make_step(0, d, dt, rng)
        output = np.zeros((n_steps, d))
        for i in range(n_steps):
            output[i] = step(i * dt)
        return output

    def run(self, t, d=None, dt=None, rng=np.random):
        dt = self.default_dt if dt is None else dt
        n_steps = int(np.round(float(t) / dt))
        return self.run_steps(n_steps, d=d, dt=dt, rng=rng)

    def run_input(self, x, d=None, dt=None, rng=np.random):
        n_steps = len(x)
        size_in = np.asarray(x[0]).size
        size_out = self.default_size_out if d is None else d
        dt = self.default_dt if dt is None else dt
        rng = self.get_rng(rng)
        step = self.make_step(size_in, size_out, dt, rng)
        output = np.zeros((n_steps, size_out))
        for i, xi in enumerate(x):
            output[i] = step(i * dt, xi)
        return output

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


class PresentInput(Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for `presentation_time` seconds.
    """
    inputs = NdarrayParam(shape=('...',))
    presentation_time = NumberParam(low=0, low_open=True)

    def __init__(self, inputs, presentation_time):
        self.inputs = inputs
        self.presentation_time = presentation_time
        super(PresentInput, self).__init__(
            default_size_out=self.inputs[0].size)

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == 0
        assert size_out == self.inputs[0].size

        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)

        def step_image_input(t):
            i = int(t / presentation_time + 1e-7)
            return inputs[i % n]

        return step_image_input


class Conv2d(Process):
    """Perform 2-D (image) convolution on an input.

    Parameters
    ----------
    filters : array_like (n_filters, n_channels, f_height, f_width)
        Static filters to convolve with the input. Shape is number of filters,
        number of input channels, filter height, and filter width. Shape can
        also be (n_filters, height, width, n_channels, f_height, f_width)
        to apply different filters at each point in the image, where 'height'
        and 'width' are the input image height and width.
    shape_in : 3-tuple (n_channels, height, width)
        Shape of the input images: channels, height, width.
    """

    shape_in = TupleParam(length=3)
    shape_out = TupleParam(length=3)
    filters = NdarrayParam(shape=('...',))
    biases = NdarrayParam(shape=('...',), optional=True)

    def __init__(self, shape_in, filters, biases=None):
        self.shape_in = tuple(shape_in)
        if len(self.shape_in) != 3:
            raise ValueError("`shape_in` must have three dimensions "
                             "(channels, height, width)")

        self.filters = filters
        self.shape_out = (self.filters.shape[0],) + self.shape_in[1:]
        if len(self.filters.shape) not in [4, 6]:
            raise ValueError(
                "`filters` must have four or six dimensions "
                "(filters, [height, width,] channels, f_height, f_width)")
        if self.filters.shape[-3] != self.shape_in[0]:
            raise ValueError(
                "Filter channels (%d) and input channels (%d) must match"
                % (self.filters.shape[-3], self.shape_in[0]))

        self.biases = biases if biases is not None else None
        if self.biases is not None:
            if self.biases.size == 1:
                self.biases.shape = (1, 1, 1)
            elif self.biases.size == np.prod(self.shape_out):
                self.biases.shape = self.shape_out
            elif self.biases.size == self.shape_out[0]:
                self.biases.shape = (self.shape_out[0], 1, 1)
            elif self.biases.size == np.prod(self.shape_out[1:]):
                self.biases.shape = (1,) + self.shape_out[1:]
            else:
                raise ValueError(
                    "Biases size (%d) does not match output shape %s"
                    % (self.biases.size, self.shape_out))

        super(Conv2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)

        filters = self.filters
        local_filters = filters.ndim == 6
        biases = self.biases
        shape_in = self.shape_in
        shape_out = self.shape_out

        def step_conv2d(t, x):
            x = x.reshape(shape_in)
            ni, nj = shape_in[-2:]
            f = filters.shape[0]
            si, sj = filters.shape[-2:]
            si2 = (si - 1) / 2
            sj2 = (sj - 1) / 2

            y = np.zeros(shape_out)
            for i in range(ni):
                for j in range(nj):
                    i0, i1 = i - si2, i + si2 + 1
                    j0, j1 = j - sj2, j + sj2 + 1
                    sli = slice(max(-i0, 0), min(ni + si - i1, si))
                    slj = slice(max(-j0, 0), min(nj + sj - j1, sj))
                    w = (filters[:, i, j, :, sli, slj] if local_filters else
                         filters[:, :, sli, slj])
                    xij = x[:, max(i0, 0):min(i1, ni), max(j0, 0):min(j1, nj)]
                    y[:, i, j] = np.dot(xij.ravel(), w.reshape(f, -1).T)

            if biases is not None:
                y += biases

            return y.ravel()

        return step_conv2d


class Pool2d(Process):
    """Perform 2-D (image) pooling on an input."""
    shape_in = TupleParam(length=3)
    shape_out = TupleParam(length=3)
    size = IntParam(low=1)
    stride = IntParam(low=1)
    kind = EnumParam(values=('avg', 'max'))

    def __init__(self, shape_in, size, stride=None, kind='avg'):
        self.shape_in = shape_in
        self.size = size
        self.stride = stride if stride is not None else size
        self.kind = kind
        if self.stride > self.size:
            raise ValueError("Stride (%d) must be <= size (%d)" %
                             (self.stride, self.size))

        c, nxi, nxj = self.shape_in
        nyi = int(np.floor(float(nxi - 1) / self.stride)) + 1
        nyj = int(np.floor(float(nxj - 1) / self.stride)) + 1
        self.shape_out = (c, nyi, nyj)

        super(Pool2d, self).__init__(
            default_size_in=np.prod(self.shape_in),
            default_size_out=np.prod(self.shape_out))

    def make_step(self, size_in, size_out, dt, rng):
        assert size_in == np.prod(self.shape_in)
        assert size_out == np.prod(self.shape_out)
        c, nxi, nxj = self.shape_in
        c, nyi, nyj = self.shape_out
        s = self.size
        st = self.stride
        kind = self.kind

        def step_pool2d(t, x):
            x = x.reshape(c, nxi, nxj)
            y = np.zeros_like(x[:, ::st, ::st])
            n = np.zeros((nyi, nyj))
            assert y.shape[-2:] == (nyi, nyj)

            for i in range(s):
                for j in range(s):
                    xij = x[:, i::st, j::st]
                    ni, nj = xij.shape[-2:]
                    if kind == 'max':
                        y[:, :ni, :nj] = np.maximum(y[:, :ni, :nj], xij)
                    elif kind == 'avg':
                        y[:, :ni, :nj] += xij
                        n[:ni, :nj] += 1
                    else:
                        raise NotImplementedError(kind)

            if kind == 'avg':
                y /= n

            return y.ravel()

        return step_pool2d


class ProcessParam(Parameter):
    """Must be a Process."""

    def validate(self, instance, process):
        super(ProcessParam, self).validate(instance, process)
        if process is not None and not isinstance(process, Process):
            raise ValueError("Must be Process (got type '%s')" % (
                process.__class__.__name__))

        return process
