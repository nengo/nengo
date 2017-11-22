"""Matrix exponential function ``expm``.

This function is used in ``utils.filter_design``. We use the Pade approximant
computation from Scipy.

----

These are borrowed from SciPy and used under their license:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2012 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np


def _comb(n, k):
    """Compute n choose k"""
    if k > n or k < 0:
        return 0
    if k == 0:
        return 1
    if k >= n // 2 + 1:
        return _comb(n, n-k)

    prod = lambda a, b: a * b
    num = reduce(prod, range(n-k+1, n+1))
    den = reduce(prod, range(1, k+1))
    return num // den


# -----------------------------------------------------------------------------
# From scipy/sparse/sputils.py v0.18.0
def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    if issequence(x):
        return False
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isshape(x):
    """Is x a valid 2-tuple of dimensions?
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                return True
        return False


def issequence(t):
    return ((isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))


def ismatrix(t):
    return ((isinstance(t, (list, tuple)) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))


def isdense(x):
    return isinstance(x, np.ndarray)


# -----------------------------------------------------------------------------
# From scipy/sparse/linalg/interface.py v0.18.0
class LinearOperator(object):
    """Common interface for performing matrix vector products

    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.

    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.

    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M,N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    dtype : dtype
        Data type of the matrix.

    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])

    """
    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return _CustomLinearOperator(*args, **kwargs)
        else:
            obj = super(LinearOperator, cls).__new__(cls)

            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                raise TypeError("LinearOperator subclass should implement"
                                " at least one of _matvec and _matmat.")

            return obj

    def __init__(self, dtype, shape):
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        if not isshape(shape):
            raise ValueError("invalid shape %r (must be 2-d)" % shape)

        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Called from subclasses at the end of the __init__ routine.
        """
        if self.dtype is None:
            v = np.zeros(self.shape[-1])
            self.dtype = np.asarray(self.matvec(v)).dtype

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """
        return np.hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])

    def _matvec(self, x):
        """Default matrix-vector multiplication handler.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        return self.matmat(x.reshape(-1, 1))

    def matvec(self, x):
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)

        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = np.asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)

        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        if isinstance(x, np.matrix):
            y = np.asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, x):
        """Default implementation of _rmatvec; defers to adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            raise NotImplementedError
        else:
            return self.H.matvec(x)

    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """

        X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        M, N = self.shape

        if X.shape[0] != N:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)

        if isinstance(Y, np.matrix):
            Y = np.asmatrix(Y)

        return Y

    def __call__(self, x):
        return self*x

    def __mul__(self, x):
        return self.dot(x)

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            x = np.asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if np.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        """Hermitian adjoint.

        Returns the Hermitian adjoint of self, aka the Hermitian
        conjugate or Hermitian transpose. For a complex matrix, the
        Hermitian adjoint is equal to the conjugate transpose.

        Can be abbreviated self.H instead of self.adjoint().

        Returns
        -------
        A_H : LinearOperator
            Hermitian adjoint of self.
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transpose this linear operator.

        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        shape = (self.shape[1], self.shape[0])
        return _CustomLinearOperator(shape, matvec=self.rmatvec,
                                     rmatvec=self.matvec,
                                     dtype=self.dtype)


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None):
        super(_CustomLinearOperator, self).__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__matmat_impl = matmat

        self._init_dtype()

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._matmat(X)

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplemented("rmatvec is not defined")
        return self.__rmatvec_impl(x)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     dtype=self.dtype)


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return np.find_common_type(dtypes, [])


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape != B.shape:
            raise ValueError('cannot add %r and %r: shape mismatch'
                             % (A, B))
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(_get_dtype([A, B]), A.shape)

    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _matmat(self, x):
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    def _adjoint(self):
        A, B = self.args
        return A.H + B.H


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch'
                             % (A, B))
        super(_ProductLinearOperator, self).__init__(_get_dtype([A, B]),
                                                     (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _matmat(self, x):
        return self.args[0].matmat(self.args[1].matmat(x))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not np.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        dtype = _get_dtype([A], [type(alpha)])
        super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return np.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * alpha


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if A.shape[0] != A.shape[1]:
            raise ValueError('square LinearOperator expected, got %r' % A)
        if not isintlike(p):
            raise ValueError('integer expected as p')

        super(_PowerLinearOperator, self).__init__(_get_dtype([A]), A.shape)
        self.args = (A, p)

    def _power(self, fun, x):
        res = np.array(x, copy=True)
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)

    def _matmat(self, x):
        return self._power(self.args[0].matmat, x)

    def _adjoint(self):
        A, p = self.args
        return A.H ** p


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj


class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = adjoint.shape[1], adjoint.shape[0]

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=None):
        super(IdentityOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self


def aslinearoperator(A):
    """Return A as a LinearOperator.

    'A' may be any of the following types:
     - ndarray
     - matrix
     - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
     - LinearOperator
     - An object with .shape and .matvec attributes

    See the LinearOperator documentation for additional information.

    Examples
    --------
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinearoperator(M)
    <2x3 MatrixLinearOperator with dtype=int32>

    """
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = np.atleast_2d(np.asarray(A))
        return MatrixLinearOperator(A)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = None
            dtype = None

            if hasattr(A, 'rmatvec'):
                rmatvec = A.rmatvec
            if hasattr(A, 'dtype'):
                dtype = A.dtype
            return LinearOperator(A.shape, A.matvec,
                                  rmatvec=rmatvec, dtype=dtype)

        else:
            raise TypeError('type not understood')


# -----------------------------------------------------------------------------
# From scipy/sparse/linalg/expm.py v0.18.0
UPPER_TRIANGULAR = 'upper_triangular'


def _onenorm_matrix_power_nnm(A, p):
    """
    Compute the 1-norm of a non-negative integer power of a non-negative matrix.

    Parameters
    ----------
    A : a square ndarray or matrix or sparse matrix
        Input matrix with non-negative entries.
    p : non-negative integer
        The power to which the matrix is to be raised.

    Returns
    -------
    out : float
        The 1-norm of the matrix power p of A.

    """
    # check input
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # Explicitly make a column vector so that this works when A is a
    # numpy matrix (in addition to ndarray and sparse matrix).
    v = np.ones((A.shape[0], 1), dtype=float)
    M = A.T
    for i in range(p):
        v = M.dot(v)
    return max(v)


def _onenorm(A):
    return np.linalg.norm(A, 1)


def _ident_like(A):
    return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)


def _count_nonzero(A):
    return np.count_nonzero(A)


def _is_upper_triangular(A):
    return _count_nonzero(np.tril(A, -1)) == 0


def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.

    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.

    """
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    f = None
    # if structure == UPPER_TRIANGULAR:
    #     f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
    if f is not None:
        if alpha is None:
            alpha = 1.
        out = f(alpha, A, B)
    else:
        if alpha is None:
            out = A.dot(B)
        else:
            out = alpha * A.dot(B)
    return out


class MatrixPowerOperator(LinearOperator):

    def __init__(self, A, p, structure=None):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0:
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self._structure = structure
        self.dtype = A.dtype
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x)
        return x

    def _rmatvec(self, x):
        A_T = self._A.T
        x = x.ravel()
        for i in range(self._p):
            x = A_T.dot(x)
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = _smart_matrix_product(self._A, X, structure=self._structure)
        return X

    @property
    def T(self):
        return MatrixPowerOperator(self._A.T, self._p)


class ProductOperator(LinearOperator):
    """
    For now, this is limited to products of multiple square matrices.
    """

    def __init__(self, *args, **kwargs):
        self._structure = kwargs.get('structure', None)
        for A in args:
            if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(
                    'For now, the ProductOperator implementation is '
                    'limited to the product of multiple square matrices.')
        if args:
            n = args[0].shape[0]
            for A in args:
                for d in A.shape:
                    if d != n:
                        raise ValueError(
                            'The square matrices of the ProductOperator '
                            'must all have the same shape.')
            self.shape = (n, n)
            self.ndim = len(self.shape)
        self.dtype = np.find_common_type([x.dtype for x in args], [])
        self._operator_sequence = args

    def _matvec(self, x):
        for A in reversed(self._operator_sequence):
            x = A.dot(x)
        return x

    def _rmatvec(self, x):
        x = x.ravel()
        for A in self._operator_sequence:
            x = A.T.dot(x)
        return x

    def _matmat(self, X):
        for A in reversed(self._operator_sequence):
            X = _smart_matrix_product(A, X, structure=self._structure)
        return X

    @property
    def T(self):
        T_args = [A.T for A in reversed(self._operator_sequence)]
        return ProductOperator(*T_args)


def _onenormest_matrix_power(
        A, p, t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return onenormest(MatrixPowerOperator(A, p, structure=structure))


def _onenormest_product(
        operator_seq, t=2, itmax=5, compute_v=False, compute_w=False,
        structure=None):
    """
    Efficiently estimate the 1-norm of the matrix product of the args.

    Parameters
    ----------
    operator_seq : linear operator sequence
        Matrices whose 1-norm of product is to be computed.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.
    structure : str, optional
        A string describing the structure of all operators.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return onenormest(ProductOperator(*operator_seq, structure=structure))


class _ExpmPadeHelper(object):
    """
    Help lazily evaluate a matrix exponential.

    The idea is to not do more work than we need for high expm precision,
    so we lazily compute matrix powers and store or precompute
    other properties of the matrix.

    """
    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and products
            will be used. Otherwise, the one-norm of powers and products
            may initially be estimated.
        """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = _ident_like(A)
        self.structure = structure
        self.use_exact_onenorm = use_exact_onenorm

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _smart_matrix_product(
                self.A, self.A, structure=self.structure)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _smart_matrix_product(
                self.A2, self.A2, structure=self.structure)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _smart_matrix_product(
                self.A4, self.A2, structure=self.structure)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _smart_matrix_product(
                self.A6, self.A2, structure=self.structure)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _smart_matrix_product(
                self.A4, self.A6, structure=self.structure)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4)**(1/4.)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6)**(1/6.)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8)**(1/8.)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1/10.)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self.use_exact_onenorm:
            return self.d4_tight
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = _onenormest_matrix_power(
                    self.A2, 2, structure=self.structure)**(1/4.)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self.use_exact_onenorm:
            return self.d6_tight
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = _onenormest_matrix_power(
                    self.A2, 3, structure=self.structure)**(1/6.)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self.use_exact_onenorm:
            return self.d8_tight
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = _onenormest_matrix_power(
                    self.A4, 2, structure=self.structure)**(1/8.)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self.use_exact_onenorm:
            return self.d10_tight
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = _onenormest_product(
                    (self.A4, self.A6), structure=self.structure)**(1/10.)
            return self._d10_approx

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _smart_matrix_product(
            self.A,
            b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
        V = b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _smart_matrix_product(
            self.A,
            b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
        V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _smart_matrix_product(
            self.A, b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
        V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.)
        U = _smart_matrix_product(
            self.A,
            (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
             b[3]*self.A2 + b[1]*self.ident),
            structure=self.structure)
        V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
             b[2]*self.A2 + b[0]*self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)
        U2 = _smart_matrix_product(
            B6,
            b[13]*B6 + b[11]*B4 + b[9]*B2,
            structure=self.structure)
        U = _smart_matrix_product(
            B,
            (U2 + b[7]*B6 + b[5]*B4 + b[3]*B2 + b[1]*self.ident),
            structure=self.structure)
        V2 = _smart_matrix_product(
            B6,
            b[12]*B6 + b[10]*B4 + b[8]*B2,
            structure=self.structure)
        V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
        return U, V


def expm(A):
    """
    Compute the matrix exponential using Pade approximation.

    Parameters
    ----------
    A : (M,M) array_like or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated

    Returns
    -------
    expA : (M,M) ndarray
        Matrix exponential of `A`

    Notes
    -----
    This is algorithm (6.1) which is a simplification of algorithm (5.1).

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    """
    return _expm(A, use_exact_onenorm='auto')


def _expm(A, use_exact_onenorm):  # noqa: C901
    # Core of expm, separated to allow testing exact and approximate
    # algorithms.

    # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
    if isinstance(A, (list, tuple)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # Trivial case
    if A.shape == (1, 1):
        out = [[np.exp(A[0, 0])]]
        return np.array(out)

    # Detect upper triangularity.
    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None

    if use_exact_onenorm == "auto":
        # Hardcode a matrix order threshold for exact vs. estimated one-norms.
        use_exact_onenorm = A.shape[0] < 200

    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(
        A, structure=structure, use_exact_onenorm=use_exact_onenorm)

    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        # Invoke Code Fragment 2.1.
        X = _fragment_2_1(X, h.A, s)
    else:
        # X = r_13(A)^(2^s) by repeated squaring.
        for i in range(s):
            X = X.dot(X)
    return X


def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `U` and `V`.
        Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    P = U + V
    Q = -U + V
    return np.linalg.solve(Q, P)


def _sinch(x):
    """
    Stably evaluate sinch.

    Notes
    -----
    The strategy of falling back to a sixth order Taylor expansion
    was suggested by the Spallation Neutron Source docs
    which was found on the internet by google search.
    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
    The details of the cutoff point and the Horner-like evaluation
    was picked without reference to anything in particular.

    Note that sinch is not currently implemented in scipy.special,
    whereas the "engineer's" definition of sinc is implemented.
    The implementation of sinc involves a scaling factor of pi
    that distinguishes it from the "mathematician's" version of sinc.

    """

    # If x is small then use sixth order Taylor expansion.
    # How small is small? I am using the point where the relative error
    # of the approximation is less than 1e-14.
    # If x is large then directly evaluate sinh(x) / x.
    x2 = x*x
    if abs(x) < 0.0135:
        return 1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.)))
    else:
        return np.sinh(x) / x


def _eq_10_42(lam_1, lam_2, t_12):
    """
    Equation (10.42) of Functions of Matrices: Theory and Computation.

    Notes
    -----
    This is a helper function for _fragment_2_1 of expm_2009.
    Equation (10.42) is on page 251 in the section on Schur algorithms.
    In particular, section 10.4.3 explains the Schur-Parlett algorithm.
    expm([[lam_1, t_12], [0, lam_1])
    =
    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
    [0, exp(lam_2)]
    """

    # The plain formula t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
    # apparently suffers from cancellation, according to Higham's textbook.
    # A nice implementation of sinch, defined as sinh(x)/x,
    # will apparently work around the cancellation.
    a = 0.5 * (lam_1 + lam_2)
    b = 0.5 * (lam_1 - lam_2)
    return t_12 * np.exp(a) * _sinch(b)


def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    The argument X is modified in-place, but this modification is not the same
    as the returned value of the function.
    This function also takes pains to do things in ways that are compatible
    with sparse matrices, for example by avoiding fancy indexing
    and by using methods of the matrices whenever possible instead of
    using functions of the numpy or scipy libraries themselves.

    """
    # Form X = r_m(2^-s T)
    # Replace diag(X) by exp(2^-s diag(T)).
    n = X.shape[0]
    diag_T = np.ravel(T.diagonal().copy())

    # Replace diag(X) by exp(2^-s diag(T)).
    scale = 2 ** -s
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]

    for i in range(s-1, -1, -1):
        X = X.dot(X)

        # Replace diag(X) by exp(2^-i diag(T)).
        scale = 2 ** -i
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]

        # Replace (first) superdiagonal of X by explicit formula
        # for superdiagonal of exp(2^-i T) from Eq (10.42) of
        # the author's 2008 textbook
        # Functions of Matrices: Theory and Computation.
        for k in range(n-1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k+1]
            t_12 = scale * T[k, k+1]
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k+1] = value

    # Return the updated X matrix.
    return X


def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    p = 2*m + 1

    # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
    # They are coefficients of terms of a generating function series expansion.
    choose_2p_p = _comb(2*p, p)
    abs_c_recip = float(choose_2p_p * math.factorial(2*p + 1))

    # This is explained after Eq. (1.2) of the 2009 expm paper.
    # It is the "unit roundoff" of IEEE double precision arithmetic.
    u = 2**-53

    # Compute the one-norm of matrix power p of abs(A).
    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), p)

    # Treat zero norm as a special case.
    if not A_abs_onenorm:
        return 0

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha/u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)


# -----------------------------------------------------------------------------
# From scipy/sparse/linalg/_onenormest.py v0.18.0
def onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Compute a lower bound of the 1-norm of a sparse matrix.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can be transposed and that can
        produce matrix products.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    Notes
    -----
    This is algorithm 2.4 of [1].

    In [2] it is described as follows.
    "This algorithm typically requires the evaluation of
    about 4t matrix-vector products and almost invariably
    produces a norm estimate (which is, in fact, a lower
    bound on the norm) correct to within a factor 3."

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Nicholas J. Higham and Francoise Tisseur (2000),
           "A Block Algorithm for Matrix 1-Norm Estimation,
           with an Application to 1-Norm Pseudospectra."
           SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
           "A new scaling and squaring algorithm for the matrix exponential."
           SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.

    """

    # Check the input.
    A = aslinearoperator(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected the operator to act like a square matrix')

    # If the operator size is small compared to t,
    # then it is easier to compute the exact norm.
    # Otherwise estimate the norm.
    n = A.shape[1]
    if t >= n:
        A_explicit = np.asarray(aslinearoperator(A).matmat(np.identity(n)))
        if A_explicit.shape != (n, n):
            raise Exception('internal error: ',
                            'unexpected shape ' + str(A_explicit.shape))
        col_abs_sums = abs(A_explicit).sum(axis=0)
        if col_abs_sums.shape != (n, ):
            raise Exception('internal error: ',
                            'unexpected shape ' + str(col_abs_sums.shape))
        argmax_j = np.argmax(col_abs_sums)
        v = elementary_vector(n, argmax_j)
        w = A_explicit[:, argmax_j]
        est = col_abs_sums[argmax_j]
    else:
        est, v, w, nmults, nresamples = _onenormest_core(A, A.H, t, itmax)

    # Report the norm estimate along with some certificates of the estimate.
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        return est


def _blocked_elementwise(func):
    """
    Decorator for an elementwise function, to apply it blockwise along
    first dimension, to avoid excessive memory usage in temporaries.
    """
    block_size = 2**20

    def wrapper(x):
        if x.shape[0] < block_size:
            return func(x)
        else:
            y0 = func(x[:block_size])
            y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
            y[:block_size] = y0
            del y0
            for j in range(block_size, x.shape[0], block_size):
                y[j:j+block_size] = func(x[j:j+block_size])
            return y
    return wrapper


@_blocked_elementwise
def sign_round_up(X):
    """
    This should do the right thing for both real and complex matrices.

    From Higham and Tisseur:
    "Everything in this section remains valid for complex matrices
    provided that sign(A) is redefined as the matrix (aij / |aij|)
    (and sign(0) = 1) transposes are replaced by conjugate transposes."

    """
    Y = X.copy()
    Y[Y == 0] = 1
    Y /= np.abs(Y)
    return Y


@_blocked_elementwise
def _max_abs_axis1(X):
    return np.max(np.abs(X), axis=1)


def _sum_abs_axis0(X):
    block_size = 2**20
    r = None
    for j in range(0, X.shape[0], block_size):
        y = np.sum(np.abs(X[j:j+block_size]), axis=0)
        if r is None:
            r = y
        else:
            r += y
    return r


def elementary_vector(n, i):
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v


def vectors_are_parallel(v, w):
    # Columns are considered parallel when they are equal or negative.
    # Entries are required to be in {-1, 1},
    # which guarantees that the magnitudes of the vectors are identical.
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    return np.dot(v, w) == n


def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    for v in X.T:
        if not any(vectors_are_parallel(v, w) for w in Y.T):
            return False
    return True


def column_needs_resampling(i, X, Y=None):
    # column i of X needs resampling if either
    # it is parallel to a previous column of X or
    # it is parallel to a column of Y
    n, t = X.shape
    v = X[:, i]
    if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
        return True
    if Y is not None:
        if any(vectors_are_parallel(v, w) for w in Y.T):
            return True
    return False


def resample_column(i, X):
    X[:, i] = np.random.randint(0, 2, size=X.shape[0])*2 - 1


def less_than_or_close(a, b):
    return np.allclose(a, b) or (a < b)


def _algorithm_2_2(A, AT, t):  # noqa: C901
    """
    This is Algorithm 2.2.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.

    Returns
    -------
    g : sequence
        A non-negative decreasing vector
        such that g[j] is a lower bound for the 1-norm
        of the column of A of jth largest 1-norm.
        The first entry of this vector is therefore a lower bound
        on the 1-norm of the linear operator A.
        This sequence has length t.
    ind : sequence
        The ith entry of ind is the index of the column A whose 1-norm
        is given by g[i].
        This sequence of indices has length t, and its entries are
        chosen from range(n), possibly with repetition,
        where n is the order of the operator A.

    Notes
    -----
    This algorithm is mainly for testing.
    It uses the 'ind' array in a way that is similar to
    its usage in algorithm 2.4.  This algorithm 2.2 may be easier to test,
    so it gives a chance of uncovering bugs related to indexing
    which could have propagated less noticeably to algorithm 2.4.

    """
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    n = A_linear_operator.shape[0]

    # Initialize the X block with columns of unit 1-norm.
    X = np.ones((n, t))
    if t > 1:
        X[:, 1:] = np.random.randint(0, 2, size=(n, t-1))*2 - 1
    X /= float(n)

    # Iteratively improve the lower bounds.
    # Track extra things, to assert invariants for debugging.
    g_prev = None
    h_prev = None
    k = 1
    ind = range(t)
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        g = _sum_abs_axis0(Y)
        best_j = np.argmax(g)
        g.sort()
        g = g[::-1]
        S = sign_round_up(Y)
        Z = np.asarray(AT_linear_operator.matmat(S))
        h = _max_abs_axis1(Z)

        # If this algorithm runs for fewer than two iterations,
        # then its return values do not have the properties indicated
        # in the description of the algorithm.
        # In particular, the entries of g are not 1-norms of any
        # column of A until the second iteration.
        # Therefore we will require the algorithm to run for at least
        # two iterations, even though this requirement is not stated
        # in the description of the algorithm.
        if k >= 2:
            if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
                break
        ind = np.argsort(h)[::-1][:t]
        h = h[ind]
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        # Check invariant (2.2).
        if k >= 2:
            if not less_than_or_close(g_prev[0], h_prev[0]):
                raise Exception('invariant (2.2) is violated')
            if not less_than_or_close(h_prev[0], g[0]):
                raise Exception('invariant (2.2) is violated')

        # Check invariant (2.3).
        if k >= 3:
            for j in range(t):
                if not less_than_or_close(g[j], g_prev[j]):
                    raise Exception('invariant (2.3) is violated')

        # Update for the next iteration.
        g_prev = g
        h_prev = h
        k += 1

    # Return the lower bounds and the corresponding column indices.
    return g, ind


def _onenormest_core(A, AT, t, itmax):  # noqa: C901
    """
    Compute a lower bound of the 1-norm of a sparse matrix.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
    itmax : int, optional
        Use at most this many iterations.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.
    nmults : int, optional
        The number of matrix products that were computed.
    nresamples : int, optional
        The number of times a parallel column was observed,
        necessitating a re-randomization of the column.

    Notes
    -----
    This is algorithm 2.4.

    """
    # This function is a more or less direct translation
    # of Algorithm 2.4 from the Higham and Tisseur (2000) paper.
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    if t < 1:
        raise ValueError('at least one column is required')
    n = A.shape[0]
    if t >= n:
        raise ValueError('t should be smaller than the order of A')
    # Track the number of big*small matrix multiplications
    # and the number of resamplings.
    nmults = 0
    nresamples = 0
    # "We now explain our choice of starting matrix.  We take the first
    # column of X to be the vector of 1s [...] This has the advantage that
    # for a matrix with nonnegative elements the algorithm converges
    # with an exact estimate on the second iteration, and such matrices
    # arise in applications [...]"
    X = np.ones((n, t), dtype=float)
    # "The remaining columns are chosen as rand{-1,1},
    # with a check for and correction of parallel columns,
    # exactly as for S in the body of the algorithm."
    if t > 1:
        for i in range(1, t):
            # These are technically initial samples, not resamples,
            # so the resampling count is not incremented.
            resample_column(i, X)
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)
                nresamples += 1
    # "Choose starting matrix X with columns of unit 1-norm."
    X /= float(n)
    # "indices of used unit vectors e_j"
    ind_hist = np.zeros(0, dtype=np.intp)
    est_old = 0
    S = np.zeros((n, t), dtype=float)
    k = 1
    ind = None
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        nmults += 1
        mags = _sum_abs_axis0(Y)
        est = np.max(mags)
        best_j = np.argmax(mags)
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j]
        # (1)
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = sign_round_up(Y)
        del Y
        # (2)
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        if t > 1:
            # "Ensure that no column of S is parallel to another column of S
            # or to a column of S_old by replacing columns of S by rand{-1,1}."
            for i in range(t):
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    nresamples += 1
        del S_old
        # (3)
        Z = np.asarray(AT_linear_operator.matmat(S))
        nmults += 1
        h = _max_abs_axis1(Z)
        del Z
        # (4)
        if k >= 2 and max(h) == h[ind_best]:
            break
        # "Sort h so that h_first >= ... >= h_last
        # and re-order ind correspondingly."
        #
        # Later on, we will need at most t+len(ind_hist) largest
        # entries, so drop the rest
        ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy()
        del h
        if t > 1:
            # (5)
            # Break if the most promising t vectors have been visited already.
            if np.in1d(ind[:t], ind_hist).all():
                break
            # Put the most promising unvisited vectors at the front of the list
            # and put the visited vectors at the end of the list.
            # Preserve the order of the indices induced by the ordering of h.
            seen = np.in1d(ind, ind_hist)
            ind = np.concatenate((ind[~seen], ind[seen]))
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        new_ind = ind[:t][~np.in1d(ind[:t], ind_hist)]
        ind_hist = np.concatenate((ind_hist, new_ind))
        k += 1
    v = elementary_vector(n, ind_best)
    return est, v, w, nmults, nresamples
