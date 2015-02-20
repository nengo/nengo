import collections
import functools

import numpy as np

from nengo.params import Parameter
from nengo.utils.compat import is_number
from nengo.utils.filter_design import cont2discrete


class Synapse(object):
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

    def __init__(self, num, den):
        self.num = num
        self.den = den

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.num, self.den)

    @staticmethod
    def no_den_step(signal, output, b):
        output[...] = b * signal

    @staticmethod
    def simple_step(signal, output, a, b):
        output *= -a
        output += b * signal

    @staticmethod
    def general_step(signal, output, x, y, num, den):
        """Filter an LTI system with the given transfer function.

        Implements a discrete-time LTI system using the difference equation
        [1]_ for the given transfer function (num, den).

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Digital_filter#Difference_equation
        """

        output[...] = 0

        x.appendleft(np.array(signal))
        for k, xk in enumerate(x):
            output += num[k] * xk
        for k, yk in enumerate(y):
            output -= den[k] * yk
        y.appendleft(np.array(output))

    def make_step(self, dt, output, method='zoh'):
        num, den, _ = cont2discrete((self.num, self.den), dt, method=method)
        num = num.flatten()
        num = num[1:] if num[0] == 0 else num
        den = den[1:]  # drop first element (equal to 1)

        if len(num) == 1 and len(den) == 0:
            return functools.partial(
                LinearFilter.no_den_step, output=output, b=num[0])
        elif len(num) == 1 and len(den) == 1:
            return functools.partial(
                LinearFilter.simple_step, output=output, a=den[0], b=num[0])
        else:
            x = collections.deque(maxlen=len(num))
            y = collections.deque(maxlen=len(den))
            return functools.partial(LinearFilter.general_step,
                                     output=output, x=x, y=y, num=num, den=den)


class Lowpass(LinearFilter):
    """Standard first-order lowpass filter synapse.

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.
    """
    def __init__(self, tau):
        self.tau = tau
        super(Lowpass, self).__init__([1], [tau, 1])

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.tau)

    def make_step(self, dt, output):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return functools.partial(
                LinearFilter.no_den_step, output=output, b=1.)
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
    def __init__(self, tau):
        self.tau = tau
        super(Alpha, self).__init__([1], [tau**2, 2*tau, 1])

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.tau)

    def make_step(self, dt, output):
        # if tau < 0.03 * dt, exp(-dt / tau) < 1e-14, so just make it zero
        if self.tau <= .03 * dt:
            return functools.partial(
                LinearFilter.no_den_step, output=output, b=1.)
        return super(Alpha, self).make_step(dt, output)


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
    def __init__(self, default, optional=True, readonly=False):
        assert optional  # None has meaning (no filtering)
        super(SynapseParam, self).__init__(
            default, optional, readonly)

    def __set__(self, conn, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        self.validate(conn, synapse)
        self.data[conn] = synapse

    def validate(self, conn, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValueError("'%s' is not a synapse type" % synapse)
        super(SynapseParam, self).validate(conn, synapse)
