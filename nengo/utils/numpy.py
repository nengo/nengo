"""
Extra functions to extend the capabilities of Numpy.
"""
from __future__ import absolute_import
import numpy as np


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


def rmse(a, b):
    return rms(a - b)
