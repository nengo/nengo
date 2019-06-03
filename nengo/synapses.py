import warnings

import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam, NdarrayParam, NumberParam, Parameter, Unconfigurable)
from nengo.utils.filter_design import cont2discrete, tf2ss
from nengo.utils.numpy import as_shape, is_number


class Synapse(Process):
    """Abstract base class for synapse models.

    Conceptually, a synapse model emulates a biological synapse, taking in
    input in the form of released neurotransmitter and opening ion channels
    to allow more or less current to flow into the neuron.

    In Nengo, the implementation of a synapse is as a specific case of a
    `.Process` in which the input and output shapes are the same.
    The input is the current across the synapse, and the output is the current
    that will be induced in the postsynaptic neuron.

    Synapses also contain the `.Synapse.filt` and `.Synapse.filtfilt` methods,
    which make it easy to use Nengo's synapse models outside of Nengo
    simulations.

    Parameters
    ----------
    default_size_in : int, optional (Default: 1)
        The size_in used if not specified.
    default_size_out : int (Default: None)
        The size_out used if not specified.
        If None, will be the same as default_size_in.
    default_dt : float (Default: 0.001 (1 millisecond))
        The simulation timestep used if not specified.
    seed : int, optional (Default: None)
        Random number seed. Ensures random factors will be the same each run.

    Attributes
    ----------
    default_dt : float (Default: 0.001 (1 millisecond))
        The simulation timestep used if not specified.
    default_size_in : int (Default: 0)
        The size_in used if not specified.
    default_size_out : int (Default: 1)
        The size_out used if not specified.
    seed : int, optional (Default: None)
        Random number seed. Ensures random factors will be the same each run.
    """

    def __init__(self, default_size_in=1, default_size_out=None,
                 default_dt=0.001, seed=None):
        if default_size_out is None:
            default_size_out = default_size_in
        super().__init__(default_size_in=default_size_in,
                         default_size_out=default_size_out,
                         default_dt=default_dt,
                         seed=seed)

    def allocate(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}  # pragma: no cover  Implement if the synapse has a state

    def filt(self, x, dt=None, axis=0, y0=None, copy=True, filtfilt=False):
        """Filter ``x`` with this synapse model.

        Parameters
        ----------
        x : array_like
            The signal to filter.
        dt : float, optional (Default: None)
            The timestep of the input signal.
            If None, ``default_dt`` will be used.
        axis : int, optional (Default: 0)
            The axis along which to filter.
        y0 : array_like, optional (Default: None)
            The starting state of the filter output. If None, the initial
            value of the input signal along the axis filtered will be used.
        copy : bool, optional (Default: True)
            Whether to copy the input data, or simply work in-place.
        filtfilt : bool, optional (Default: False)
            If True, runs the process forward then backward on the signal,
            for zero-phase filtering (like Matlab's ``filtfilt``).
        """
        # This function is very similar to `Process.apply`, but allows for
        # a) filtering along any axis, and b) zero-phase filtering (filtfilt).
        dt = self.default_dt if dt is None else dt
        filtered = np.array(x, copy=copy)
        filt_view = np.rollaxis(filtered, axis=axis)  # rolled view on filtered

        if y0 is None:
            y0 = filt_view[0]

        shape_in = shape_out = as_shape(filt_view[0].shape, min_dim=1)
        state = self.allocate(shape_in, shape_out, dt, dtype=filtered.dtype,
                              y0=y0)
        step = self.make_step(shape_in, shape_out, dt, rng=None, state=state)

        for i, signal_in in enumerate(filt_view):
            filt_view[i] = step(i * dt, signal_in)

        if filtfilt:  # Flip the filt_view and filter again
            n = len(filt_view) - 1
            filt_view = filt_view[::-1]
            for i, signal_in in enumerate(filt_view):
                filt_view[i] = step((n - i) * dt, signal_in)

        return filtered

    def filtfilt(self, x, **kwargs):
        """Zero-phase filtering of ``x`` using this filter.

        Equivalent to `filt(x, filtfilt=True, **kwargs) <.Synapse.filt>`.
        """
        return self.filt(x, filtfilt=True, **kwargs)


class LinearFilter(Synapse):
    """General linear time-invariant (LTI) system synapse.

    This class can be used to implement any linear filter, given the
    filter's transfer function. [1]_

    Parameters
    ----------
    num : array_like
        Numerator coefficients of transfer function.
    den : array_like
        Denominator coefficients of transfer function.
    analog : boolean, optional (Default: True)
        Whether the synapse coefficients are analog (i.e. continuous-time),
        or discrete. Analog coefficients will be converted to discrete for
        simulation using the simulator ``dt``.

    Attributes
    ----------
    analog : boolean
        Whether the synapse coefficients are analog (i.e. continuous-time),
        or discrete. Analog coefficients will be converted to discrete for
        simulation using the simulator ``dt``.
    den : ndarray
        Denominator coefficients of transfer function.
    num : ndarray
        Numerator coefficients of transfer function.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Filter_%28signal_processing%29
    """

    num = NdarrayParam('num', shape='*')
    den = NdarrayParam('den', shape='*')
    analog = BoolParam('analog')

    def __init__(self, num, den, analog=True, **kwargs):
        super().__init__(**kwargs)
        self.num = num
        self.den = den
        self.analog = analog

    def combine(self, obj):
        """Combine in series with another LinearFilter."""
        if not isinstance(obj, LinearFilter):
            raise ValidationError(
                "Can only combine with other LinearFilters", attr='obj')
        if self.analog != obj.analog:
            raise ValidationError(
                "Cannot combine analog and digital filters", attr='obj')
        num = np.polymul(self.num, obj.num)
        den = np.polymul(self.den, obj.den)
        return LinearFilter(num, den,
                            analog=self.analog,
                            default_size_in=self.default_size_in,
                            default_size_out=self.default_size_out,
                            default_dt=self.default_dt,
                            seed=self.seed)

    def evaluate(self, frequencies):
        """Evaluate the transfer function at the given frequencies.

        Examples
        --------

        Using the ``evaluate`` function to make a Bode plot::

            synapse = nengo.synapses.LinearFilter([1], [0.02, 1])
            f = numpy.logspace(-1, 3, 100)
            y = synapse.evaluate(f)
            plt.subplot(211); plt.semilogx(f, 20*np.log10(np.abs(y)))
            plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')
            plt.subplot(212); plt.semilogx(f, np.angle(y))
            plt.xlabel('frequency [Hz]'); plt.ylabel('phase [radians]')
        """
        frequencies = 2.j*np.pi*frequencies
        w = frequencies if self.analog else np.exp(frequencies)
        y = np.polyval(self.num, w) / np.polyval(self.den, w)
        return y

    def _get_ss(self, dt, method='zoh'):
        A, B, C, D = tf2ss(self.num, self.den)

        # discretize (if len(A) == 0, filter is stateless and already discrete)
        if self.analog and len(A) > 0:
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt, method=method)

        if B.ndim == 1:
            B = B[:, None]
        if D.ndim == 1:
            D = D[:, None]

        return A, B, C, D

    def allocate(self, shape_in, shape_out, dt,
                 y0=None, dtype=np.float64, method='zoh'):
        assert shape_in == shape_out

        A, B, C, D = self._get_ss(dt, method=method)

        # create state memory variable X
        X = np.zeros((A.shape[0],) + shape_out, dtype=dtype)

        return dict(X=X)

    def make_step(self, shape_in, shape_out, dt, rng, state=None,
                  method='zoh'):
        """Returns a `.Step` instance that implements the linear filter."""
        assert shape_in == shape_out
        assert state is not None

        A, B, C, D = self._get_ss(dt, method=method)
        X = state['X']

        def step_linearfilter(t, signal):
            output = np.dot(C, X) + D * signal
            X[:] = np.dot(A, X) + B * signal
            return output

        return step_linearfilter


class Lowpass(LinearFilter):
    """Standard first-order lowpass filter synapse.

    The impulse-response function is given by::

        f(t) = (1 / tau) * exp(-t / tau)

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.

    Attributes
    ----------
    tau : float
        The time constant of the filter in seconds.
    """
    tau = NumberParam('tau', low=0)

    def __init__(self, tau, **kwargs):
        super().__init__([1], [tau, 1], **kwargs)
        self.tau = tau


class Alpha(LinearFilter):
    """Alpha-function filter synapse.

    The impulse-response function is given by::

        alpha(t) = (t / tau**2) * exp(-t / tau)

    and was found by [1]_ to be a good basic model for synapses.

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.

    Attributes
    ----------
    tau : float
        The time constant of the filter in seconds.

    References
    ----------
    .. [1] Mainen, Z.F. and Sejnowski, T.J. (1995). Reliability of spike timing
       in neocortical neurons. Science (New York, NY), 268(5216):1503-6.
    """

    tau = NumberParam('tau', low=0)

    def __init__(self, tau, **kwargs):
        super().__init__([1], [tau**2, 2*tau, 1], **kwargs)
        self.tau = tau


class Delay(LinearFilter):
    """A single time-step (pure) delay. """

    def __init__(self, **kwargs):
        super().__init__([1], [1, 0], analog=False, **kwargs)


class Triangle(Synapse):
    """Triangular finite impulse response (FIR) synapse.

    This synapse has a triangular and finite impulse response. The length of
    the triangle is ``t`` seconds; thus the digital filter will have
    ``t / dt + 1`` taps.

    Parameters
    ----------
    t : float
        Length of the triangle, in seconds.

    Attributes
    ----------
    t : float
        Length of the triangle, in seconds.
    """

    t = NumberParam('t', low=0)

    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def _get_n(self, dt):
        n_taps = int(np.round(self.t / float(dt))) + 1
        num = np.arange(n_taps, 0, -1)
        num = num / num.sum()

        # Minimal multiply implementation finds the difference between
        # coefficients and subtracts a scaled signal at each time step.
        n0, ndiff = num[0], num[-1]

        return n_taps, n0, ndiff

    def allocate(self, shape_in, shape_out, dt, y0=None, dtype=np.float64):
        assert shape_in == shape_out

        n_taps, _, ndiff = self._get_n(dt)
        Y = np.zeros(shape_out, dtype=dtype)
        X = np.zeros((n_taps,) + shape_out, dtype=dtype)
        Xi = np.zeros(1, dtype=dtype)  # counter for X position

        if y0 is not None and len(X) > 0:
            y0 = np.array(y0, copy=False, ndmin=1)
            X[:] = ndiff * y0[None, ...]
            Y[:] = y0

        return dict(Y=Y, X=X, Xi=Xi)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        assert shape_in == shape_out
        assert state is not None

        Y, X, Xi = state['Y'], state['X'], state['Xi']
        n_taps, n0, ndiff = self._get_n(dt)
        n0 = n0.astype(Y.dtype)
        ndiff = ndiff.astype(X.dtype)
        assert len(X) == n_taps

        def step_triangle(t, signal):
            Y[...] += n0 * signal
            Y[...] -= X.sum(axis=0)
            Xi[:] = (Xi + 1) % len(X)
            X[int(Xi.item())] = ndiff * signal
            return Y

        return step_triangle


def filt(signal, synapse, dt, axis=0, x0=None, copy=True):
    """Filter ``signal`` with ``synapse``.

    .. note:: Deprecated in Nengo 2.1.0.
              Use `.Synapse.filt` method instead.
    """
    warnings.warn("Use ``synapse.filt`` instead", DeprecationWarning)
    return synapse.filt(signal, dt=dt, axis=axis, y0=x0, copy=copy)


def filtfilt(signal, synapse, dt, axis=0, x0=None, copy=True):
    """Zero-phase filtering of ``signal`` using the ``synapse`` filter.

    .. note:: Deprecated in Nengo 2.1.0.
              Use `.Synapse.filtfilt` method instead.
    """
    warnings.warn("Use ``synapse.filtfilt`` instead", DeprecationWarning)
    return synapse.filtfilt(signal, dt=dt, axis=axis, y0=x0, copy=copy)


class SynapseParam(Parameter):
    equatable = True

    def __init__(self, name,
                 default=Unconfigurable, optional=True, readonly=None):
        super().__init__(name, default, optional, readonly)

    def coerce(self, instance, synapse):
        synapse = Lowpass(synapse) if is_number(synapse) else synapse
        self.check_type(instance, synapse, Synapse)
        return super().coerce(instance, synapse)
