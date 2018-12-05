"""
Extra functions to extend the capabilities of Numpy.
"""
from __future__ import absolute_import

import numpy as np

from .compat import PY2, is_integer, is_iterable
from ..exceptions import ValidationError

maxint = np.iinfo(np.int32).max


def compare(a, b):
    return 0 if a == b else 1 if a > b else -1 if a < b else None


def as_shape(x, min_dim=0):
    """Return a tuple if ``x`` is iterable or ``(x,)`` if ``x`` is integer."""
    if is_iterable(x):
        shape = tuple(x)
    elif is_integer(x):
        shape = (x,)
    else:
        raise ValueError("%r cannot be safely converted to a shape" % x)

    if len(shape) < min_dim:
        shape = tuple([1] * (min_dim - len(shape))) + shape

    return shape


def broadcast_shape(shape, length):
    """Pad a shape with ones following standard Numpy broadcasting."""
    n = len(shape)
    if n < length:
        return tuple([1] * (length - n) + list(shape))
    else:
        return shape


def array(x, dims=None, min_dims=0, readonly=False, **kwargs):
    y = np.array(x, **kwargs)
    dims = max(min_dims, y.ndim) if dims is None else dims

    if y.ndim < dims:
        shape = np.ones(dims, dtype='int')
        shape[:y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValidationError("Input cannot be cast to array with "
                              "%d dimensions" % dims, attr='dims')

    if readonly:
        y.flags.writeable = False

    return y


def array_hash(a, n=100):
    """Simple fast array hash function.

    For arrays with size larger than ``n``, pick ``n`` elements at random
    to hash. This strategy should work well for dense matrices, but for
    sparse ones it is more likely to give hash collisions.
    """
    if not isinstance(a, np.ndarray):
        return hash(a)

    if a.size < n:
        # hash all elements
        v = np.array(a)  # copy for numpy<1.13 (issue #1122)
        v.setflags(write=False)
        return hash(v.data if PY2 else v.data.tobytes())
    else:
        # pick random elements to hash
        rng = np.random.RandomState(a.size)
        inds = [rng.randint(0, a.shape[i], size=n) for i in range(a.ndim)]
        v = a[inds]
        v.setflags(write=False)
        return hash(v.data if PY2 else v.data.tobytes())


def array_offset(x):
    """Get offset of array data from base data in bytes."""
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__['data'][0]
    start = x.__array_interface__['data'][0]
    return start - base_start


def expm(A, n_factors=None, normalize=False):
    """Simple matrix exponential to replace Scipy's matrix exponential

    This just uses a recursive (factored) version of the Taylor series,
    and is not as good as Scipy (which uses Pade approximants). The hard
    part with this implementation is choosing the length of the Taylor
    series. A longer series is generally needed for a matrix with a larger
    eigenvalues, but I'm not exactly sure how these relate. I'm using
    a heuristic based on the matrix norm, since this is kind-of related
    to the size of eigenvalues, though for larger norms the function
    becomes inaccurate no matter the length of the series.

    This function is mostly intended for use in `filter_design`, where
    the matrices should be small, both in dimensions and norm.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValidationError("'A' must be a square matrix", attr='A')

    a = np.linalg.norm(A)
    if normalize:
        a = int(a)
        A = A / float(a)

    if n_factors is None:
        n_factors = 20 if normalize else min(max(int(a), 20), 1000)

    Y = np.zeros_like(A)
    for i in range(n_factors, 0, -1):
        Y = np.dot(A / float(i), Y)
        np.fill_diagonal(Y, Y.diagonal() + 1)  # add identity matrix

    return np.linalg.matrix_power(Y, a) if normalize else Y


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
    return np.sqrt(np.sum(x**2, axis=axis, keepdims=keepdims))


def meshgrid_nd(*args):
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(*(
        a.reshape(s[:i] + (-1,) + s[i + 1:]) for i, a in enumerate(args)))


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
    return np.sqrt(np.mean(x**2, axis=axis, keepdims=keepdims))


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


if hasattr(np.fft, 'rfftfreq'):
    rfftfreq = np.fft.rfftfreq
else:
    def rfftfreq(n, d=1.0):
        return np.abs(np.fft.fftfreq(n=n, d=d)[:n // 2 + 1])
