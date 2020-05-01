"""
Extra functions to extend the capabilities of Numpy.
"""
from collections import OrderedDict
from collections.abc import Iterable
import logging
import os
import warnings

import numpy as np

from ..exceptions import ValidationError


logger = logging.getLogger(__name__)
try:
    import scipy.sparse as scipy_sparse

    def is_spmatrix(obj):
        """Check if ``obj`` is a sparse matrix."""
        return isinstance(obj, scipy_sparse.spmatrix)


except ImportError as e:
    logger.info("Could not import scipy.sparse:\n%s", str(e))
    scipy_sparse = None

    def is_spmatrix(obj):
        """Check if ``obj`` is a sparse matrix."""
        return False


maxseed = np.iinfo(np.uint32).max
maxint = np.iinfo(np.int32).max


# numpy 1.17 introduced a slowdown to clip, so
# use nengo.utils.numpy.clip instead of np.clip
# This has persisted through 1.19 at least
clip = (
    np.core.umath.clip
    if tuple(int(st) for st in np.__version__.split(".")) >= (1, 17, 0)
    else np.clip
)


def is_integer(obj):
    """Check if ``obj`` is an integer type."""
    return isinstance(obj, (int, np.integer))


def is_iterable(obj):
    """Check if ``obj`` is an iterable."""
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0  # 0-d arrays give error if iterated over
    else:
        return isinstance(obj, Iterable)


def is_number(obj, check_complex=False):
    """Check if ``obj`` is a numeric type."""
    types = (float, complex, np.number) if check_complex else (float, np.floating)
    return is_integer(obj) or isinstance(obj, types)


def is_array(obj):
    """Check if ``obj`` is a numpy array."""
    # np.generic allows us to return true for scalars as well as true arrays
    return isinstance(obj, (np.ndarray, np.generic))


def is_array_like(obj):
    """Check if ``obj`` is an array like object."""
    # While it's possible that there are some iterables other than list/tuple
    # that can be made into arrays, it's very likely that those arrays
    # will have dtype=object, which is likely to cause unexpected issues.
    return is_array(obj) or is_number(obj) or isinstance(obj, (list, tuple))


def compare(a, b):
    """Return -1/0/1 if a is less/equal/greater than b."""

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
    """Create numpy array with some extra configuration.

    This is a wrapper around ``np.array``.

    Unlike ``np.array``, the additional single-dimensional indices added by
    ``dims`` or ``min_dims`` will appear at the *end* of the shape (for example,
    ``array([1, 2, 3], dims=4).shape == (3, 1, 1, 1)``).

    Parameters
    ----------
    dims : int or None
        If not ``None``, force the output array to have exactly this many indices.
        If the input has more than this number of indices, this throws an error.
    min_dims : int
        Force the output array to have at least this many indices
        (ignored if ``dims is not None``).
    readonly : bool
        Make the output array read-only.
    **kwargs
        Additional keyword arguments to pass to ``np.array``.
    """

    y = np.array(x, **kwargs)
    dims = max(min_dims, y.ndim) if dims is None else dims

    if y.ndim < dims:
        shape = np.ones(dims, dtype="int")
        shape[: y.ndim] = y.shape
        y.shape = shape
    elif y.ndim > dims:
        raise ValidationError(
            "Input cannot be cast to array with %d dimensions" % dims, attr="dims"
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
    """Compute the Euclidean norm.

    Parameters
    ----------
    x : array_like
        Array to compute the norm over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. ``None`` sums all axes. See ``np.sum``.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See ``np.sum`` in
        newer versions of Numpy (>= 1.7).
    """
    x = np.asarray(x)
    return np.sqrt(np.sum(x ** 2, axis=axis, keepdims=keepdims))


def meshgrid_nd(*args):
    """Multidimensional meshgrid."""
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(
        *(a.reshape(s[:i] + (-1,) + s[i + 1 :]) for i, a in enumerate(args))
    )


def rms(x, axis=None, keepdims=False):
    """Compute the root-mean-square amplitude.

    Parameters
    ----------
    x : array_like
        Array to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. ``None`` sums all axes. See ``np.sum``.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See ``np.sum`` in
        newer versions of Numpy (>= 1.7).
    """
    x = np.asarray(x)
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=keepdims))


def rmse(x, y, axis=None, keepdims=False):  # pragma: no cover
    """Compute the root-mean-square error.

    Equivalent to rms(x - y, axis=axis, keepdims=keepdims).

    Parameters
    ----------
    x, y : array_like
        Arrays to compute RMS amplitude over.
    axis : None or int or tuple of ints, optional
        Axis or axes to sum across. ``None`` sums all axes. See ``np.sum``.
    keepdims : bool, optional
        If True, the reduced axes are left in the result. See ``np.sum`` in
        newer versions of Numpy (>= 1.7).
    """
    warnings.warn(
        "The 'rmse' function is deprecated and will be removed in a future "
        "version. Please use ``rms(x - y)`` instead.",
        DeprecationWarning,
    )
    x, y = np.asarray(x), np.asarray(y)
    return rms(x - y, axis=axis, keepdims=keepdims)


if hasattr(np.fft, "rfftfreq"):
    rfftfreq = np.fft.rfftfreq
else:

    def rfftfreq(n, d=1.0):
        return np.abs(np.fft.fftfreq(n=n, d=d)[: n // 2 + 1])


_betaincinv22_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "betaincinv22_table.npz"
)
_betaincinv22_table = None


def _make_betaincinv22_table(filename=_betaincinv22_file, n_interp=200, n_dims=50):
    """Save lookup table for betaincinv22_lookup. """
    import scipy.special  # pylint: disable=import-outside-toplevel

    rng = np.random.RandomState(0)

    n_dims_log = int(0.5 * n_dims)
    dims_lin = np.arange(1, n_dims - n_dims_log + 1)
    dims_log = np.round(np.logspace(np.log10(dims_lin[-1] + 1), 3, n_dims_log)).astype(
        dims_lin.dtype
    )
    dims = np.unique(np.concatenate([dims_lin, dims_log]))

    x_table = []
    y_table = []
    for dim in dims:
        n_range = int(0.8 * n_interp)
        x0 = np.linspace(0, 1, n_interp - n_range)  # samples in the domain
        y0 = np.linspace(1e-16, 1 - 1e-7, n_range)  # samples in the range
        x1 = scipy.special.betainc(dim / 2.0, 0.5, y0)
        interp_x = np.unique(np.concatenate([x0, x1]))
        while len(interp_x) < n_interp:
            # add random points until we achieve the length
            interp_x = np.unique(
                np.concatenate([interp_x, rng.uniform(size=n_interp - len(interp_x))])
            )

        interp_x.sort()
        assert interp_x.size == n_interp

        interp_y = scipy.special.betaincinv(dim / 2.0, 0.5, interp_x)
        x_table.append(interp_x)
        y_table.append(interp_y)

    x_table = np.asarray(x_table)
    y_table = np.asarray(y_table)

    np.savez(filename, dims=dims, x=x_table, y=y_table)


def betaincinv22_lookup(dims, x, filename=_betaincinv22_file):
    """Look up values for betaincinv(dims / 2, 0.5, x). """
    global _betaincinv22_table

    if not is_integer(dims) or dims < 1:
        raise ValueError("`dims` must be an integer >= 1")

    if _betaincinv22_table is None:
        data = np.load(filename)
        _betaincinv22_table = OrderedDict(
            (d, (x, y)) for d, x, y in zip(data["dims"], data["x"], data["y"])
        )
        assert (np.diff(list(_betaincinv22_table)) >= 1).all()

    if dims in _betaincinv22_table:
        xp, yp = _betaincinv22_table[dims]
    else:
        known_dims = np.array(list(_betaincinv22_table))
        i = np.searchsorted(known_dims, dims)
        assert i > 0
        if i >= len(known_dims):
            # dims is larger than any known dimension we have, so just use the largest
            xp, yp = _betaincinv22_table[known_dims[-1]]
        else:
            # take average of two curves
            dims0, dims1 = known_dims[i - 1], known_dims[i]
            xp0, yp0 = _betaincinv22_table[dims0]
            xp1, yp1 = _betaincinv22_table[dims1]
            assert dims0 < dims < dims1
            ratio0 = (dims1 - dims) / (dims1 - dims0)
            ratio1 = 1 - ratio0
            xp = (ratio0 * xp0 + ratio1 * xp1) if len(xp0) == len(xp1) else xp0
            yp = ratio0 * np.interp(xp, xp0, yp0) + ratio1 * np.interp(xp, xp1, yp1)

    return np.interp(x, xp, yp)
