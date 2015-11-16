import collections
import warnings

import numpy as np

from nengo.exceptions import ValidationError
from nengo.params import (BoolParam, NdarrayParam, NumberParam, Parameter,
                          FrozenObject, Unconfigurable)
from nengo.utils.compat import is_number
from nengo.utils.filter_design import cont2discrete


class Synapse(FrozenObject):
    """Abstract base class for synapse objects"""

    def __init__(self, analog=True):
        super(Synapse, self).__init__()
        self.analog = analog

    def filt(self, signal, dt=None, axis=0, y0=None, copy=True,
             filtfilt=False):
        """Filter ``signal`` with this synapse.

        Parameters
        ----------
        signal : array_like
            The signal to filter.
        dt : float
            The time-step of the input signal, for analog synapses.
        axis : integer, optional (default: 0)
            The axis along which to filter.
        y0 : array_like, optional (default: np.zeros(d))
            The starting state of the filter output.
        copy : boolean, optional (default: True)
            Whether to copy the input data, or simply work in-place.
        filtfilt : boolean, optional (default: False)
            If true, runs the process forwards then backwards on the signal,
            for zero-phase filtering (like MATLAB's ``filtfilt``).
        """
        if self.analog and dt is None:
            raise ValueError("`dt` must be provided for analog synapses.")

        filtered = np.array(signal, copy=copy)
        filt_view = np.rollaxis(filtered, axis=axis)  # rolled view on filtered

        # --- buffer method
        if y0 is not None:
            if y0.shape != filt_view[0].shape:
                raise ValidationError(
                    "'y0' with shape %s must have shape %s" %
                    (y0.shape, filt_view[0].shape), attr='y0')
            signal_out = np.array(y0)
        else:
            # signal_out is our buffer for the current filter state
            signal_out = np.zeros_like(filt_view[0])

        step = self.make_step(dt, signal_out)

        for i, signal_in in enumerate(filt_view):
            step(signal_in)
            filt_view[i] = signal_out

        if filtfilt:
            # Flip the filt_view and filter again
            filt_view = filt_view[::-1]
            for i, signal_in in enumerate(filt_view):
                step(signal_in)
                filt_view[i] = signal_out

        return filtered

    def filtfilt(self, *args, **kwargs):
        """Zero-phase filtering of ``signal`` using this filter.

        Equivalent to ``filt(*args, **kwargs, filtfilt=True)``.
        """
        return self.filt(*args, filtfilt=True, **kwargs)

    def make_step(self, dt, output):
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

    def make_step(self, dt, output, method='zoh'):
        num, den = self.num, self.den
        if self.analog:
            num, den, _ = cont2discrete((num, den), dt, method=method)
            num = num.flatten()

        if den[0] != 1.:
            raise ValidationError("First element of the denominator must be 1",
                                  attr='den', obj=self)
        num = num[1:] if num[0] == 0 else num
        den = den[1:]  # drop first element (equal to 1)

        if len(num) == 1 and len(den) == 0:
            return LinearFilter.NoDen(num, den, output)
        elif len(num) == 1 and len(den) == 1:
            return LinearFilter.Simple(num, den, output)
        return LinearFilter.General(num, den, output)

    class Step(object):
        """Abstract base class for LTI filtering step functions."""
        def __init__(self, num, den, output):
            self.num = num
            self.den = den
            self.output = output

        def __call__(self, signal):
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

        def __call__(self, signal):
            self.output[...] = self.b * signal

    class Simple(Step):
        """An LTI step function for transfer functions with one num and den.

        This step function should be much faster than the equivalent general
        step function.
        """
        def __init__(self, num, den, output):
            if len(num) != 1:
                raise ValidationError("'num' must be length 1 (got %d)"
                                      % len(num), attr='num', obj=self)
            if len(den) != 1:
                raise ValidationError("'den' must be length 1 (got %d)"
                                      % len(den), attr='den', obj=self)

            super(LinearFilter.Simple, self).__init__(num, den, output)
            self.b = num[0]
            self.a = den[0]

        def __call__(self, signal):
            self.output *= -self.a
            self.output += self.b * signal

    class General(Step):
        """An LTI step function for any given transfer function.

        Implements a discrete-time LTI system using the difference equation
        [1]_ for the given transfer function (num, den).

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Digital_filter#Difference_equation
        """
        def __init__(self, num, den, output):
            super(LinearFilter.General, self).__init__(num, den, output)
            self.x = collections.deque(maxlen=len(num))
            self.y = collections.deque(maxlen=len(den))

        def __call__(self, signal):
            self.output[...] = 0

            self.x.appendleft(np.array(signal))
            for k, xk in enumerate(self.x):
                self.output += self.num[k] * xk
            for k, yk in enumerate(self.y):
                self.output -= self.den[k] * yk
            self.y.appendleft(np.array(self.output))


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

    def make_step(self, dt, output):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return LinearFilter.NoDen(np.array([1.]), np.array([]), output)
        return super(Lowpass, self).make_step(dt, output)


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

    def make_step(self, dt, output):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return LinearFilter.NoDen(np.array([1.]), np.array([]), output)
        return super(Alpha, self).make_step(dt, output)


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

    def make_step(self, dt, output):
        n_taps = int(np.round(self.t / float(dt))) + 1
        num = np.arange(n_taps, 0, -1, dtype=output.dtype)
        num /= num.sum()

        # Minimal multiply implementation finds the difference between
        # coefficients and subtracts a scaled signal at each time step.
        n0, ndiff = num[0], num[-1]
        x = collections.deque(maxlen=n_taps)

        def step_triangle(signal):
            output[...] += n0 * signal
            for xk in x:
                output[...] -= xk
            x.appendleft(ndiff * signal)

        return step_triangle


def filt(signal, synapse, dt, axis=0, x0=None, copy=True):
    """Filter ``signal`` with ``synapse``.

    Deprecated: use ``synapse.filt`` instead.

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    syanpse : float, Synapse
        The synapse model with which to filter the signal.
        If a float is passed in, it will be interpreted as the ``tau``
        parameter of a lowpass filter.
    dt : float
        The time-step of the input signal, for analog synapses.
    axis : integer, optional
        The axis along which to filter. Default: 0.
    x0 : array_like, optional
        The starting state of the filter output.
    copy : boolean, optional
        Whether to copy the input data, or simply work in-place. Default: True.
    """
    warnings.warn("Use ``synapse.filt`` instead", DeprecationWarning)
    synapse.filt(signal, dt=dt, axis=axis, y0=x0, copy=copy)


def filtfilt(signal, synapse, dt, axis=0, copy=True):
    """Zero-phase filtering of ``signal`` using the ``syanpse`` filter.

    This is done by filtering the input in forward and reverse directions.

    Equivalent to scipy and Matlab's filtfilt function using the filter
    defined by the synapse object passed in.

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    synapse : float, Synapse
        The synapse model with which to filter the signal.
        If a float is passed in, it will be interpreted as the ``tau``
        parameter of a lowpass filter.
    dt : float
        The time-step of the input signal, for analog synapses.
    axis : integer, optional
        The axis along which to filter. Default: 0.
    copy : boolean, optional
        Whether to copy the input data, or simply work in-place. Default: True.
    """
    warnings.warn("Use ``synapse.filtfilt`` instead", DeprecationWarning)
    synapse.filtfilt(signal, dt=dt, axis=axis, copy=copy)


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
