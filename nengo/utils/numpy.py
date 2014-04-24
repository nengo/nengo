"""
Extra functions to extend the capabilities of Numpy.
"""
from __future__ import absolute_import
import numpy as np

maxint = np.iinfo(np.int32).max


def array(x, dims=None, min_dims=0, **kwargs):
    y = np.array(x, **kwargs)
    dims = max(min_dims, y.ndim) if dims is None else dims

    if y.ndim < dims:
        shape = np.ones(dims, dtype='int')
        shape[:y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValueError(
            "Input cannot be cast to array with %d dimensions" % dims)

    return y


def filt(x, tau, axis=0, copy=True):
    """First-order causal lowpass filter.

    This performs standard first-order lowpass filtering with transfer function
                         1
        T(s) = ----------------------
               tau_in_seconds * s + 1
    discretized using the zero-order hold method.

    Parameters
    ----------
    x : array_like
        The signal to filter.
    tau : float
        The dimensionless filter time constant (tau = tau_in_seconds / dt).
    axis : integer
        The axis along which to filter.
    copy : boolean
        Whether to copy the input data, or simply work in-place.
    """
    x = np.array(x, copy=copy)
    y = np.rollaxis(x, axis=axis)  # y is rolled view on x

    # --- buffer method
    d = -np.expm1(-1. / tau)
    yy = np.zeros_like(y[0])  # yy is our buffer for the current filter state
    for i, yi in enumerate(y):
        yy += d * (yi - yy)
        y[i] = yy

    return x


def filtfilt(x, tau, axis=0, copy=True):
    """Zero-phase second-order non-causal lowpass filter, implemented by
    filtering the input in forward and reverse directions.

    This function is equivalent to scipy's or Matlab's filtfilt function
    with the first-order lowpass filter
                         1
        T(s) = ----------------------
               tau_in_seconds * s + 1
    as the filter. The resulting equivalent filter has zero phase distortion
    and a transfer function magnitude equal to the square of T(s),
    discretized using the zero-order hold method.

    Parameters
    ----------
    x : array_like
        The signal to filter.
    tau : float
        The dimensionless filter time constant (tau = tau_in_seconds / dt).
    axis : integer
        The axis along which to filter.
    copy : boolean
        Whether to copy the input data, or simply work in-place.
    """
    x = np.array(x, copy=copy)
    y = np.rollaxis(x, axis=axis)  # y is rolled view on x

    # --- buffer method
    d = -np.expm1(-1. / tau)

    # filter forwards
    yy = np.zeros_like(y[0])  # yy is our buffer for the current filter state
    for i, yi in enumerate(y):
        yy += d * (yi - yy)
        y[i] = yy

    # filter backwards
    z = y[::-1]  # z is a flipped view on y
    for i, zi in enumerate(z):
        yy += d * (zi - yy)
        z[i] = yy

    return x


def norm(x, axis=None, keepdims=False):
    """Euclidean norm

    Parameters
    ----------
    x : array_like
        Array to compute the norm over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.sum(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def rms(x, axis=None, keepdims=False):
    """Root-mean-square amplitude

    Parameters
    ----------
    x : array_like
        Array to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    y = np.sqrt(np.mean(x**2, axis=axis))
    return np.expand_dims(y, axis=axis) if keepdims else y


def rmse(x, y, axis=None, keepdims=False):
    """Root-mean-square error amplitude

    Equivalent to rms(x - y, axis=axis, keepdims=keepdims).

    Parameters
    ----------
    x, y : array_like
        Arrays to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. `None` sums all axes. See `np.sum`.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See `np.sum` in
        newer versions of Numpy (>= 1.7).
    """
    return rms(x - y, axis=axis, keepdims=keepdims)
