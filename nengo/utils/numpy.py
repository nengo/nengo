"""
Extra functions to extend the capabilities of Numpy.
"""
import collections
import logging

import numpy as np

from ..exceptions import ValidationError


logger = logging.getLogger(__name__)
try:
    import scipy.sparse as scipy_sparse

    def is_spmatrix(obj):
        return isinstance(obj, scipy_sparse.spmatrix)


except ImportError as e:
    logger.info("Could not import scipy.sparse:\n%s", str(e))
    scipy_sparse = None

    def is_spmatrix(obj):
        return False


maxint = np.iinfo(np.int32).max


def is_integer(obj):
    return isinstance(obj, (int, np.integer))


def is_iterable(obj):
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0  # 0-d arrays give error if iterated over
    else:
        return isinstance(obj, collections.abc.Iterable)


def is_number(obj, check_complex=False):
    types = (float, complex, np.number) if check_complex else (float, np.floating)
    return is_integer(obj) or isinstance(obj, types)


def is_array(obj):
    # np.generic allows us to return true for scalars as well as true arrays
    return isinstance(obj, (np.ndarray, np.generic))


def is_array_like(obj):
    # While it's possible that there are some iterables other than list/tuple
    # that can be made into arrays, it's very likely that those arrays
    # will have dtype=object, which is likely to cause unexpected issues.
    return is_array(obj) or is_number(obj) or isinstance(obj, (list, tuple))


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
        shape = np.ones(dims, dtype="int")
        shape[: y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValidationError(
            "Input cannot be cast to array with " "%d dimensions" % dims, attr="dims"
        )

    if readonly:
        y.flags.writeable = False

    return y


def _array_hash(a, n=100):
    if not isinstance(a, np.ndarray):
        return hash(a)

    if a.size < n:
        # hash all elements
        v = a.view()
        v.setflags(write=False)
        return hash(v.data.tobytes())
    else:
        # pick random elements to hash
        rng = np.random.RandomState(a.size)
        inds = tuple(rng.randint(0, a.shape[i], size=n) for i in range(a.ndim))
        v = a[inds]
        v.setflags(write=False)
        return hash(v.data.tobytes())


def array_hash(a, n=100):
    """Simple fast array hash function.

    For arrays with size larger than ``n``, pick ``n`` elements at random
    to hash. This strategy should work well for dense arrays, but for
    sparse arrays (those with few non-zero elements) it is more likely to
    give hash collisions.

    For sparse matrices, we apply the same logic on the underlying elements
    (data, indices) of the sparse matrix.
    """
    if scipy_sparse and isinstance(a, scipy_sparse.spmatrix):
        if isinstance(
            a,
            (scipy_sparse.csr_matrix, scipy_sparse.csc_matrix, scipy_sparse.bsr_matrix),
        ):
            return hash(
                (
                    _array_hash(a.data, n=n),
                    _array_hash(a.indices, n=n),
                    _array_hash(a.indptr, n=n),
                )
            )
        else:
            if not isinstance(a, scipy_sparse.coo_matrix):
                a = a.tocoo()
            return hash(
                (
                    _array_hash(a.data, n=n),
                    _array_hash(a.row, n=n),
                    _array_hash(a.col, n=n),
                )
            )

    return _array_hash(a, n=n)


def array_offset(x):
    """Get offset of array data from base data in bytes."""
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__["data"][0]
    start = x.__array_interface__["data"][0]
    return start - base_start


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
    return np.sqrt(np.sum(x ** 2, axis=axis, keepdims=keepdims))


def meshgrid_nd(*args):
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(
        *(a.reshape(s[:i] + (-1,) + s[i + 1 :]) for i, a in enumerate(args))
    )


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
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))


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


if hasattr(np.fft, "rfftfreq"):
    rfftfreq = np.fft.rfftfreq
else:

    def rfftfreq(n, d=1.0):
        return np.abs(np.fft.fftfreq(n=n, d=d)[: n // 2 + 1])
