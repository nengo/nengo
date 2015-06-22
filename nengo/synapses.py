import collections

import numpy as np

from nengo.params import (BoolParam, NdarrayParam, NumberParam, Parameter,
                          FrozenObject, Unconfigurable)
from nengo.utils.compat import is_number
from nengo.utils.filter_design import cont2discrete


class Synapse(FrozenObject):
    """Abstract base class for synapse objects"""

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

    num = NdarrayParam(shape='*')
    den = NdarrayParam(shape='*')
    analog = BoolParam()

    def __init__(self, num, den, analog=True):
        super(LinearFilter, self).__init__()
        self.num = num
        self.den = den
        self.analog = analog

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.num, self.den)

    def make_step(self, dt, output, method='zoh'):
        num, den = self.num, self.den
        if self.analog:
            num, den, _ = cont2discrete((num, den), dt, method=method)
            num = num.flatten()

        if den[0] != 1.:
            raise ValueError("First element of the denominator must be 1")
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
            raise NotImplementedError

    class NoDen(Step):
        """An LTI step function for transfer functions with no denominator.

        This step function should be much faster than the equivalent general
        step function.
        """
        def __init__(self, num, den, output):
            if len(den) > 0:
                raise ValueError("`den` must be empty (got length %d)"
                                 % (len(den)))
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
            if len(num) != 1 or len(den) != 1:
                raise ValueError("`num` and `den` must both be length 1 "
                                 "(got %d and %d)" % (len(num), len(den)))
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
    tau = NumberParam(low=0)

    def __init__(self, tau):
        super(Lowpass, self).__init__([1], [tau, 1])
        self.tau = tau

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.tau)

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
    tau = NumberParam(low=0)

    def __init__(self, tau):
        super(Alpha, self).__init__([1], [tau**2, 2*tau, 1])
        self.tau = tau

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.tau)

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
    t = NumberParam(low=0)

    def __init__(self, t):
        super(Triangle, self).__init__()
        self.t = t

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.t)

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

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    syanpse : float, Synapse
        The synapse model with which to filter the signal.
        If a float is passed in, it will be interpreted as the ``tau``
        parameter of a lowpass filter.
    axis : integer, optional
        The axis along which to filter. Default: 0.
    x0 : array_like, optional
        The starting state of the filter output.
    copy : boolean, optional
        Whether to copy the input data, or simply work in-place. Default: True.
    """
    if is_number(synapse):
        synapse = Lowpass(synapse)

    filtered = np.array(signal, copy=copy)
    filt_view = np.rollaxis(filtered, axis=axis)  # rolled view on filtered

    # --- buffer method
    if x0 is not None:
        if x0.shape != filt_view[0].shape:
            raise ValueError("'x0' with shape %s must have shape %s" %
                             (x0.shape, filt_view[0].shape))
        signal_out = np.array(x0)
    else:
        # signal_out is our buffer for the current filter state
        signal_out = np.zeros_like(filt_view[0])

    step = synapse.make_step(dt, signal_out)

    for i, signal_in in enumerate(filt_view):
        step(signal_in)
        filt_view[i] = signal_out

    return filtered


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
    axis : integer, optional
        The axis along which to filter. Default: 0.
    copy : boolean, optional
        Whether to copy the input data, or simply work in-place. Default: True.
    """
    if is_number(synapse):
        synapse = Lowpass(synapse)

    filtered = np.array(signal, copy=copy)
    filt_view = np.rollaxis(filtered, axis=axis)
    signal_out = np.zeros_like(filt_view[0])
    step = synapse.make_step(dt, signal_out)

    for i, signal_in in enumerate(filt_view):
        step(signal_in)
        filt_view[i] = signal_out

    # Flip the filt_view and filter again
    filt_view = filt_view[::-1]
    for i, signal_in in enumerate(filt_view):
        step(signal_in)
        filt_view[i] = signal_out

    return filtered


class SynapseParam(Parameter):
    equatable = True

    def __init__(self, default=Unconfigurable, optional=True, readonly=None):
        super(SynapseParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        super(SynapseParam, self).__set__(instance, synapse)

    def validate(self, instance, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValueError("'%s' is not a synapse type" % synapse)
        super(SynapseParam, self).validate(instance, synapse)


class LinearFilterParam(SynapseParam):
    def validate(self, instance, synapse):
        if synapse is not None and not isinstance(synapse, LinearFilter):
            raise ValueError("'%s' is not a LinearFilter" % synapse)
        super(SynapseParam, self).validate(instance, synapse)
