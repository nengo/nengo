import warnings

import numpy as np

import nengo.utils.numpy as npext
from nengo.base import Process
from nengo.dists import DistributionParam, Gaussian
from nengo.exceptions import ValidationError
from nengo.params import BoolParam, DictParam, EnumParam, NdarrayParam, NumberParam
from nengo.synapses import LinearFilter, Lowpass, SynapseParam
from nengo.utils.numpy import is_number


class WhiteNoise(Process):
    """Full-spectrum white noise process.

    Parameters
    ----------
    dist : Distribution, optional
        The distribution from which to draw samples.
    scale : bool, optional
        Whether to scale the white noise for integration. Integrating white
        noise requires using a time constant of ``sqrt(dt)`` instead of ``dt``
        on the noise term [1]_, to ensure the magnitude of the integrated
        noise does not change with ``dt``.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.

    References
    ----------
    .. [1] Gillespie, D.T. (1996) Exact numerical simulation of the Ornstein-
       Uhlenbeck process and its integral. Phys. Rev. E 54, pp. 2084-91.
    """

    dist = DistributionParam("dist")
    scale = BoolParam("scale")

    def __init__(self, dist=Gaussian(mean=0, std=1), scale=True, **kwargs):
        super().__init__(default_size_in=0, **kwargs)
        self.dist = dist
        self.scale = scale

    def __repr__(self):
        return "%s(%r, scale=%r)" % (type(self).__name__, self.dist, self.scale)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert len(shape_out) == 1

        dist = self.dist
        scale = self.scale
        alpha = 1.0 / np.sqrt(dt)
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
        The synapse to use to filter the noise.
    dist : Distribution, optional
        The distribution used to generate the white noise.
    scale : bool, optional
        Whether to scale the white noise for integration, making the output
        signal invariant to ``dt``.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """

    synapse = SynapseParam("synapse")
    dist = DistributionParam("dist")
    scale = BoolParam("scale")

    def __init__(
        self,
        synapse=Lowpass(tau=0.005),
        dist=Gaussian(mean=0, std=1),
        scale=True,
        # fmt: off
        **kwargs
        # fmt: on
    ):
        super().__init__(default_size_in=0, **kwargs)
        self.synapse = synapse
        self.dist = dist
        self.scale = scale

    def __repr__(self):
        return "%s(synapse=%r, dist=%r, scale=%r)" % (
            type(self).__name__,
            self.synapse,
            self.dist,
            self.scale,
        )

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        return self.synapse.make_state(shape_out, shape_out, dt, dtype=dtype)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert len(shape_out) == 1

        dist = self.dist
        scale = self.scale
        alpha = 1.0 / np.sqrt(dt)
        filter_step = self.synapse.make_step(shape_out, shape_out, dt, rng, state)

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
    dist : Distribution, optional
        The distribution used to generate the white noise.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """

    def __init__(self, dist=Gaussian(mean=0, std=1), **kwargs):
        super().__init__(
            synapse=LinearFilter([1], [1, 0], method="euler"), dist=dist, **kwargs
        )

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.dist)


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
    high : float
        The cut-off frequency of the low-pass filter, in Hz.
        Must not exceed the Nyquist frequency for the simulation
        timestep, which is ``0.5 / dt``.
    rms : float, optional
        The root mean square power of the filtered signal
    y0 : float, optional
        Align the phase of each output dimension to begin at the value
        that is closest (in absolute value) to y0.
    seed : int, optional
        Random number seed. Ensures noise will be the same each run.
    """

    period = NumberParam("period", low=0, low_open=True)
    high = NumberParam("high", low=0, low_open=True)
    rms = NumberParam("rms", low=0, low_open=True)
    y0 = NumberParam("y0", optional=True)

    def __init__(self, period, high, rms=0.5, y0=None, **kwargs):
        super().__init__(default_size_in=0, **kwargs)
        self.period = period
        self.high = high
        self.rms = rms
        self.y0 = y0

        if self.high is not None and self.high < 1.0 / self.period:
            raise ValidationError(
                "Make ``high >= 1. / period`` to produce a non-zero signal",
                attr="high",
                obj=self,
            )

    def __repr__(self):
        return "%s(period=%r, high=%r, rms=%r)" % (
            type(self).__name__,
            self.period,
            self.high,
            self.rms,
        )

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)

        nyquist_cutoff = 0.5 / dt
        if self.high > nyquist_cutoff:
            raise ValidationError(
                "High must not exceed the Nyquist frequency "
                "for the given dt (%0.3f)" % nyquist_cutoff,
                attr="high",
                obj=self,
            )

        n_coefficients = int(np.ceil(self.period / dt / 2.0))
        shape = (n_coefficients + 1,) + shape_out
        sigma = self.rms * np.sqrt(0.5)
        coefficients = 1j * rng.normal(0.0, sigma, size=shape)
        coefficients += rng.normal(0.0, sigma, size=shape)
        coefficients[0] = 0.0
        coefficients[-1].imag = 0.0

        set_to_zero = npext.rfftfreq(2 * n_coefficients, d=dt) > self.high
        coefficients[set_to_zero] = 0.0
        power_correction = np.sqrt(
            1.0 - np.sum(set_to_zero, dtype=float) / n_coefficients
        )
        if power_correction > 0.0:
            coefficients /= power_correction
        coefficients *= np.sqrt(2 * n_coefficients)
        signal = np.fft.irfft(coefficients, axis=0)

        if self.y0 is not None:
            # Starts each dimension off where it is closest to y0
            def shift(x):
                offset = np.argmin(abs(self.y0 - x))
                return np.roll(x, -offset + 1)  # +1 since t starts at dt

            signal = np.apply_along_axis(shift, 0, signal)

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
        Show each input for this amount of time (in seconds).
    """

    inputs = NdarrayParam("inputs", shape=("...",))
    presentation_time = NumberParam("presentation_time", low=0, low_open=True)

    def __init__(self, inputs, presentation_time, **kwargs):
        self.inputs = inputs
        self.presentation_time = presentation_time
        super().__init__(
            default_size_in=0, default_size_out=self.inputs[0].size, **kwargs
        )

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert shape_out == (self.inputs[0].size,)

        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)

        def step_presentinput(t):
            i = int((t - dt) / presentation_time + 1e-7)
            return inputs[i % n]

        return step_presentinput


class PiecewiseDataParam(DictParam):
    """Piecewise-specific validation for the data dictionary.

    In the `.Piecewise` data dict, the keys are points in time (float) and
    values are numerical constants or callables of the same dimensionality.
    """

    def coerce(self, instance, data):
        data = super().coerce(instance, data)

        size_out = None
        for time, value in data.items():
            if not is_number(time):
                raise ValidationError(
                    "Keys must be times (floats or ints), "
                    "not %r" % type(time).__name__,
                    attr="data",
                    obj=instance,
                )

            # figure out the length of this item
            if callable(value):
                try:
                    value = np.ravel(value(time))
                except Exception:
                    raise ValidationError(
                        "callable object for time step %.3f "
                        "should return a numerical constant" % time,
                        attr="data",
                        obj=instance,
                    )
            else:
                value = np.ravel(value)
                data[time] = value
            size = value.size

            # make sure this is the same size as previous items
            if size != size_out and size_out is not None:
                raise ValidationError(
                    "time %g has size %d instead of %d" % (time, size, size_out),
                    attr="data",
                    obj=instance,
                )
            size_out = size

        return data


class Piecewise(Process):
    """A piecewise function with different options for interpolation.

    Given an input dictionary of ``{0: 0, 0.5: -1, 0.75: 0.5, 1: 0}``,
    this process  will emit the numerical values (0, -1, 0.5, 0)
    starting at the corresponding time points (0, 0.5, 0.75, 1).

    The keys in the input dictionary must be times (float or int).
    The values in the dictionary can be floats, lists of floats,
    or numpy arrays. All lists or numpy arrays must be of the same length,
    as the output shape of the process will be determined by the shape
    of the values.

    Interpolation on the data points using `scipy.interpolate` is also
    supported. The default interpolation is 'zero', which creates a
    piecewise function whose values change at the specified time points.
    So the above example would be shortcut for::

        def function(t):
            if t < 0.5:
                return 0
            elif t < 0.75
                return -1
            elif t < 1:
                return 0.5
            else:
                return 0

    For times before the first specified time, an array of zeros (of
    the correct length) will be emitted.
    This means that the above can be simplified to::

        Piecewise({0.5: -1, 0.75: 0.5, 1: 0})

    Parameters
    ----------
    data : dict
        A dictionary mapping times to the values that should be emitted
        at those times. Times must be numbers (ints or floats), while values
        can be numbers, lists of numbers, numpy arrays of numbers,
        or callables that return any of those options.
    interpolation : str, optional
        One of 'linear', 'nearest', 'slinear', 'quadratic', 'cubic', or 'zero'.
        Specifies how to interpolate between times with specified value.
        'zero' creates a plain piecewise function whose values begin at
        corresponding time points, while all other options interpolate
        as described in `scipy.interpolate`.

    Attributes
    ----------
    data : dict
        A dictionary mapping times to the values that should be emitted
        at those times. Times are numbers (ints or floats), while values
        can be numbers, lists of numbers, numpy arrays of numbers,
        or callables that return any of those options.
    interpolation : str
        One of 'linear', 'nearest', 'slinear', 'quadratic', 'cubic', or 'zero'.
        Specifies how to interpolate between times with specified value.
        'zero' creates a plain piecewise function whose values change at
        corresponding time points, while all other options interpolate
        as described in `scipy.interpolate`.

    Examples
    --------

    >>> from nengo.processes import Piecewise
    >>> process = Piecewise({0.5: 1, 0.75: -1, 1: 0})
    >>> with nengo.Network() as model:
    ...     u = nengo.Node(process, size_out=process.default_size_out)
    ...     up = nengo.Probe(u)
    >>> with nengo.Simulator(model) as sim:
    ...     sim.run(1.5)
    >>> f = sim.data[up]
    >>> t = sim.trange()
    >>> f[t == 0.2]
    array([[ 0.]])
    >>> f[t == 0.58]
    array([[ 1.]])
    """

    data = PiecewiseDataParam("data", readonly=True)
    interpolation = EnumParam(
        "interpolation",
        values=("zero", "linear", "nearest", "slinear", "quadratic", "cubic"),
    )

    def __init__(self, data, interpolation="zero", **kwargs):
        self.data = data

        needs_scipy = ("linear", "nearest", "slinear", "quadratic", "cubic")
        if interpolation in needs_scipy:
            self.sp_interpolate = None
            if any(callable(val) for val in self.data.values()):
                warnings.warn(
                    "%r interpolation cannot be applied because "
                    "a callable was supplied for some piece of the "
                    "function. Using 'zero' interpolation instead." % (interpolation,)
                )
                interpolation = "zero"
            else:
                try:
                    import scipy.interpolate

                    self.sp_interpolate = scipy.interpolate
                except ImportError:
                    warnings.warn(
                        "%r interpolation cannot be applied because "
                        "scipy is not installed. Using 'zero' "
                        "interpolation instead." % (interpolation,)
                    )
                    interpolation = "zero"
        self.interpolation = interpolation

        super().__init__(default_size_in=0, default_size_out=self.size_out, **kwargs)

    @property
    def size_out(self):
        time, value = next(iter(self.data.items()))
        value = np.ravel(value(time)) if callable(value) else value
        return value.size

    def make_step(self, shape_in, shape_out, dt, rng, state):
        tp, yp = zip(*sorted(self.data.items()))
        assert shape_in == (0,)
        assert shape_out == (self.size_out,)

        if self.interpolation == "zero":

            def step_piecewise(t):
                ti = (np.searchsorted(tp, t + 0.5 * dt) - 1).clip(-1, len(yp) - 1)
                if ti == -1:
                    return np.zeros(shape_out)
                else:
                    return np.ravel(yp[ti](t)) if callable(yp[ti]) else yp[ti]

        else:
            assert self.sp_interpolate is not None

            if self.interpolation == "cubic" and 0 not in tp:
                warnings.warn(
                    "'cubic' interpolation may fail if data not " "specified for t=0.0"
                )

            f = self.sp_interpolate.interp1d(
                tp,
                yp,
                axis=0,
                kind=self.interpolation,
                bounds_error=False,
                fill_value=0.0,
            )

            def step_piecewise(t):
                return np.ravel(f(t))

        return step_piecewise
