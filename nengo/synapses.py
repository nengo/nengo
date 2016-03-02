import collections
import warnings

import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.params import (BoolParam, NdarrayParam, NumberParam, Parameter,
                          Unconfigurable)
from nengo.utils.compat import is_number
from nengo.utils.filter_design import cont2discrete
from nengo.utils.numpy import as_shape


class Synapse(Process):
    """Abstract base class for synapse objects"""

    def __init__(self, analog=True):
        super(Synapse, self).__init__()
        self.analog = analog

    def filt(self, x, dt=1., axis=0, y0=None, copy=True, filtfilt=False):
        """Filter ``x`` with this synapse.

        Parameters
        ----------
        x : array_like
            The signal to filter.
        dt : float, optional (default: 1)
            The time-step of the input signal for analog synapses (default: 1).
        axis : integer, optional (default: 0)
            The axis along which to filter.
        y0 : array_like, optional (default: x0)
            The starting state of the filter output. Defaults to the initial
            value of the input signal along the axis filtered.
        copy : boolean, optional (default: True)
            Whether to copy the input data, or simply work in-place.
        filtfilt : boolean, optional (default: False)
            If true, runs the process forwards then backwards on the signal,
            for zero-phase filtering (like MATLAB's ``filtfilt``).
        """
        # This function is very similar to `Process.apply`, but allows for
        # a) filtering along any axis, and b) zero-phase filtering (filtfilt).

        if self.analog and dt is None:
            raise ValueError("`dt` must be provided for analog synapses.")

        filtered = np.array(x, copy=copy)
        filt_view = np.rollaxis(filtered, axis=axis)  # rolled view on filtered

        if y0 is None:
            y0 = filt_view[0]

        shape_in = shape_out = as_shape(filt_view[0].shape, min_dim=1)
        step = self.make_step(shape_in, shape_out, dt, None, y0=y0)

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

        Equivalent to ``filt(x, filtfilt=True, **kwargs)``.
        """
        return self.filt(x, filtfilt=True, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, y0=None):
        raise NotImplementedError("Synapses should implement make_step.")


class LinearFilter(Synapse):
    """General linear time-invariant (LTI) system synapse.

    This class can be used to implement any linear filter, given the
    filter's transfer function. [1]_


    Parameters
    ----------
    num : array_like
        Numerator coefficients of continuous-time transfer function.
    den : array_like
        Denominator coefficients of continuous-time transfer function.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Filter_%28signal_processing%29
    """

    num = NdarrayParam('num', shape='*')
    den = NdarrayParam('den', shape='*')
    analog = BoolParam('analog')

    def __init__(self, num, den, analog=True):
        super(LinearFilter, self).__init__(analog=analog)
        self.num = num
        self.den = den

    def __repr__(self):
        return "%s(%s, %s, analog=%r)" % (
            self.__class__.__name__, self.num, self.den, self.analog)

    def evaluate(self, frequencies):
        """Evaluate transfer function at given frequencies.

        Example
        -------
        Using the ``evaluate`` function to make a Bode plot::
        >>> synapse = nengo.synapses.LinearFilter([1], [0.02, 1])
        >>> f = numpy.logspace(-1, 3, 100)
        >>> y = synapse.evaluate(f)
        >>> plt.subplot(211); plt.semilogx(f, 20*np.log10(np.abs(y)))
        >>> plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')
        >>> plt.subplot(212); plt.semilogx(f, np.angle(y))
        >>> plt.xlabel('frequency [Hz]'); plt.ylabel('phase [radians]')
        """
        frequencies = 2.j*np.pi*frequencies
        w = frequencies if self.analog else np.exp(frequencies)
        y = np.polyval(self.num, w) / np.polyval(self.den, w)
        return y

    def make_step(self, shape_in, shape_out, dt, rng, y0=None, method='zoh'):
        assert shape_in == shape_out

        num, den = self.num, self.den
        if self.analog:
            num, den, _ = cont2discrete((num, den), dt, method=method)
            num = num.flatten()

        if den[0] != 1.:
            raise ValidationError("First element of the denominator must be 1",
                                  attr='den', obj=self)
        num = num[1:] if num[0] == 0 else num
        den = den[1:]  # drop first element (equal to 1)

        output = np.zeros(shape_out)
        if len(num) == 1 and len(den) == 0:
            return LinearFilter.NoDen(num, den, output)
        elif len(num) == 1 and len(den) == 1:
            return LinearFilter.Simple(num, den, output, y0=y0)
        return LinearFilter.General(num, den, output, y0=y0)

    @staticmethod
    def _make_zero_step(shape_in, shape_out, dt, rng, y0=None):
        output = np.zeros(shape_out)
        if y0 is not None:
            output[:] = y0

        return LinearFilter.NoDen(np.array([1.]), np.array([]), output)

    class Step(object):
        """Abstract base class for LTI filtering step functions."""
        def __init__(self, num, den, output):
            self.num = num
            self.den = den
            self.output = output

        def __call__(self, t, signal):
            raise NotImplementedError("Step functions must implement __call__")

    class NoDen(Step):
        """An LTI step function for transfer functions with no denominator.

        This step function should be much faster than the equivalent general
        step function.
        """
        def __init__(self, num, den, output):
            if len(den) > 0:
                raise ValidationError("'den' must be empty (got length %d)"
                                      % len(den), attr='den', obj=self)
            super(LinearFilter.NoDen, self).__init__(num, den, output)
            self.b = num[0]

        def __call__(self, t, signal):
            self.output[...] = self.b * signal
            return self.output

    class Simple(Step):
        """An LTI step function for transfer functions with one num and den.

        This step function should be much faster than the equivalent general
        step function.
        """
        def __init__(self, num, den, output, y0=None):
            if len(num) != 1:
                raise ValidationError("'num' must be length 1 (got %d)"
                                      % len(num), attr='num', obj=self)
            if len(den) != 1:
                raise ValidationError("'den' must be length 1 (got %d)"
                                      % len(den), attr='den', obj=self)

            super(LinearFilter.Simple, self).__init__(num, den, output)
            self.b = num[0]
            self.a = den[0]
            if y0 is not None:
                self.output[...] = y0

        def __call__(self, t, signal):
            self.output *= -self.a
            self.output += self.b * signal
            return self.output

    class General(Step):
        """An LTI step function for any given transfer function.

        Implements a discrete-time LTI system using the difference equation
        [1]_ for the given transfer function (num, den).

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Digital_filter#Difference_equation
        """
        def __init__(self, num, den, output, y0=None):
            super(LinearFilter.General, self).__init__(num, den, output)
            self.x = collections.deque(maxlen=len(num))
            self.y = collections.deque(maxlen=len(den))
            if y0 is not None:
                self.output[...] = y0
                for _ in num:
                    self.x.appendleft(np.array(self.output))
                for _ in den:
                    self.y.appendleft(np.array(self.output))

        def __call__(self, t, signal):
            self.output[...] = 0

            self.x.appendleft(np.array(signal))
            for k, xk in enumerate(self.x):
                self.output += self.num[k] * xk
            for k, yk in enumerate(self.y):
                self.output -= self.den[k] * yk
            self.y.appendleft(np.array(self.output))

            return self.output


class Lowpass(LinearFilter):
    """Standard first-order lowpass filter synapse.

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.
    """
    tau = NumberParam('tau', low=0)

    def __init__(self, tau):
        super(Lowpass, self).__init__([1], [tau, 1])
        self.tau = tau

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.tau)

    def make_step(self, shape_in, shape_out, dt, rng, y0=None, **kwargs):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return self._make_zero_step(shape_in, shape_out, dt, rng, y0=y0)
        return super(Lowpass, self).make_step(
            shape_in, shape_out, dt, rng, y0=y0, **kwargs)


class Alpha(LinearFilter):
    """Alpha-function filter synapse.

    The impulse-response function is given by

        alpha(t) = (t / tau) * exp(-t / tau)

    and was found by [1]_ to be a good basic model for synapses.

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.

    References
    ----------
    .. [1] Mainen, Z.F. and Sejnowski, T.J. (1995). Reliability of spike timing
       in neocortical neurons. Science (New York, NY), 268(5216):1503-6.
    """
    tau = NumberParam('tau', low=0)

    def __init__(self, tau):
        super(Alpha, self).__init__([1], [tau**2, 2*tau, 1])
        self.tau = tau

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.tau)

    def make_step(self, shape_in, shape_out, dt, rng, y0=None, **kwargs):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return self._make_zero_step(shape_in, shape_out, dt, rng, y0=y0)
        return super(Alpha, self).make_step(
            shape_in, shape_out, dt, rng, y0=y0, **kwargs)


class Triangle(Synapse):
    """Triangular FIR synapse.

    This synapse has a triangular and finite impulse response. The length of
    the triangle is `t` seconds, thus the digital filter will have `t / dt + 1`
    taps.
    """
    t = NumberParam('t', low=0)

    def __init__(self, t):
        super(Triangle, self).__init__(analog=True)
        self.t = t

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.t)

    def make_step(self, shape_in, shape_out, dt, rng, y0=None):
        assert shape_in == shape_out

        n_taps = int(np.round(self.t / float(dt))) + 1
        num = np.arange(n_taps, 0, -1, dtype=float)
        num /= num.sum()

        # Minimal multiply implementation finds the difference between
        # coefficients and subtracts a scaled signal at each time step.
        n0, ndiff = num[0], num[-1]
        x = collections.deque(maxlen=n_taps)

        output = np.zeros(shape_out)
        if y0 is not None:
            output[:] = y0

        def step_triangle(t, signal):
            output[...] += n0 * signal
            for xk in x:
                output[...] -= xk
            x.appendleft(ndiff * signal)
            return output

        return step_triangle


def filt(signal, synapse, dt, axis=0, x0=None, copy=True):
    """Filter ``signal`` with ``synapse``.

    Deprecated: use ``synapse.filt`` instead.
    """
    warnings.warn("Use ``synapse.filt`` instead", DeprecationWarning)
    synapse.filt(signal, dt=dt, axis=axis, y0=x0, copy=copy)


def filtfilt(signal, synapse, dt, axis=0, x0=None, copy=True):
    """Zero-phase filtering of ``signal`` using the ``syanpse`` filter.

    Deprecated: use ``synapse.filtfilt`` instead.
    """
    warnings.warn("Use ``synapse.filtfilt`` instead", DeprecationWarning)
    synapse.filtfilt(signal, dt=dt, axis=axis, y0=x0, copy=copy)


class SynapseParam(Parameter):
    equatable = True

    def __init__(self, name,
                 default=Unconfigurable, optional=True, readonly=None):
        super(SynapseParam, self).__init__(name, default, optional, readonly)

    def __set__(self, instance, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        super(SynapseParam, self).__set__(instance, synapse)

    def validate(self, instance, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValidationError("'%s' is not a synapse type" % synapse,
                                  attr=self.name, obj=instance)
        super(SynapseParam, self).validate(instance, synapse)


class LinearFilterParam(SynapseParam):
    def validate(self, instance, synapse):
        if synapse is not None and not isinstance(synapse, LinearFilter):
            raise ValidationError("'%s' is not a LinearFilter" % synapse,
                                  attr=self.name, obj=instance)
        super(SynapseParam, self).validate(instance, synapse)
