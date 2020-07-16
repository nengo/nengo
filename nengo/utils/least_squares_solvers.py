"""These solvers are to be passed as arguments to `~.Solver` objects.

For example:

.. testcode::

   from nengo.solvers import LstsqL2
   from nengo.utils.least_squares_solvers import SVD

   with nengo.Network():
       ens_a = nengo.Ensemble(10, 1)
       ens_b = nengo.Ensemble(10, 1)
       nengo.Connection(ens_a, ens_b, solver=LstsqL2(solver=SVD()))

"""

import numpy as np

import nengo.utils.numpy as npext
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam,
    FrozenObject,
    IntParam,
    NdarrayParam,
    NumberParam,
    Parameter,
)


def format_system(A, Y):
    """Extract data from A/Y matrices."""

    assert Y.ndim > 0
    m, n = A.shape
    matrix_in = Y.ndim > 1
    d = Y.shape[1] if matrix_in else 1
    Y = Y if matrix_in else Y[:, None]
    return Y, m, n, d, matrix_in


def rmses(A, X, Y):
    """Returns the root-mean-squared error (RMSE) of the solution X."""
    return npext.rms(Y - np.dot(A, X), axis=0)


class LeastSquaresSolver(FrozenObject):
    """Linear least squares system solver."""

    def __call__(self, A, Y, sigma, rng=None):
        raise NotImplementedError("LeastSquaresSolver must implement call")


class Cholesky(LeastSquaresSolver):
    """Solve a least-squares system using the Cholesky decomposition."""

    transpose = BoolParam("transpose", optional=True)

    def __init__(self, transpose=None):
        super().__init__()
        self.transpose = transpose

    def __call__(self, A, Y, sigma, rng=None):
        m, n = A.shape
        transpose = self.transpose
        if transpose is None:
            # transpose if matrix is fat, but not if sigmas for each neuron
            transpose = m < n and sigma.size == 1

        if transpose:
            # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
            G = np.dot(A, A.T)
            b = Y
        else:
            # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
            G = np.dot(A.T, A)
            b = np.dot(A.T, Y)

        # add L2 regularization term 'lambda' = m * sigma**2
        np.fill_diagonal(G, G.diagonal() + m * sigma ** 2)

        try:
            import scipy.linalg  # pylint: disable=import-outside-toplevel

            factor = scipy.linalg.cho_factor(G, overwrite_a=True)
            X = scipy.linalg.cho_solve(factor, b)
        except ImportError:
            L = np.linalg.cholesky(G)
            L = np.linalg.inv(L.T)
            X = np.dot(L, np.dot(L.T, b))

        X = np.dot(A.T, X) if transpose else X
        info = {"rmses": rmses(A, X, Y)}
        return X, info


class ConjgradScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's conjugate gradient.

    Parameters
    ----------
    tol : float
        Relative tolerance of the CG solver (see [1]_ for details).
    atol : float
        Absolute tolerance of the CG solver (see [1]_ for details).

    References
    ----------
    .. [1] scipy.sparse.linalg.cg documentation,
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
    """

    tol = NumberParam("tol", low=0)
    atol = NumberParam("atol", low=0)

    def __init__(self, tol=1e-4, atol=1e-8):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        assert scipy.sparse.linalg

        super().__init__()
        self.tol = tol
        self.atol = atol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = m * sigma ** 2
        calcAA = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        G = scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=calcAA, matmat=calcAA, dtype=A.dtype
        )
        B = np.dot(A.T, Y)

        X = np.zeros((n, d), dtype=B.dtype)
        infos = np.zeros(d, dtype="int")
        itns = np.zeros(d, dtype="int")
        for i in range(d):
            # use the callback to count the number of iterations
            def callback(x, i=i):
                itns[i] += 1

            try:
                X[:, i], infos[i] = scipy.sparse.linalg.cg(
                    G, B[:, i], tol=self.tol, callback=callback, atol=self.atol
                )
            except TypeError as e:  # pragma: no cover
                # no atol parameter in Scipy < 1.1.0
                if "atol" not in str(e):
                    raise e
                X[:, i], infos[i] = scipy.sparse.linalg.cg(
                    G, B[:, i], tol=self.tol, callback=callback
                )

        info = {"rmses": rmses(A, X, Y), "iterations": itns, "info": infos}
        return X if matrix_in else X.ravel(), info


class LSMRScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's LSMR."""

    tol = NumberParam("tol", low=0)

    def __init__(self, tol=1e-4):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        assert scipy.sparse.linalg

        super().__init__()
        self.tol = tol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = sigma * np.sqrt(m)
        X = np.zeros((n, d), dtype=Y.dtype)
        itns = np.zeros(d, dtype="int")
        for i in range(d):
            X[:, i], _, itns[i], _, _, _, _, _ = scipy.sparse.linalg.lsmr(
                A, Y[:, i], damp=damp, atol=self.tol, btol=self.tol
            )

        info = {"rmses": rmses(A, X, Y), "iterations": itns}
        return X if matrix_in else X.ravel(), info


class Conjgrad(LeastSquaresSolver):
    """Solve a least-squares system using conjugate gradient."""

    tol = NumberParam("tol", low=0)
    maxiters = IntParam("maxiters", low=1, optional=True)
    X0 = NdarrayParam("X0", shape=("*", "*"), optional=True)

    def __init__(self, tol=1e-2, maxiters=None, X0=None):
        super().__init__()
        self.tol = tol
        self.maxiters = maxiters
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)
        X = np.zeros((n, d)) if self.X0 is None else np.array(self.X0)
        if X.shape != (n, d):
            raise ValidationError(
                "Must be shape %s, got %s" % ((n, d), X.shape), attr="X0", obj=self
            )

        damp = m * sigma ** 2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        iters = -np.ones(d, dtype="int")
        for i in range(d):
            X[:, i], iters[i] = self._conjgrad_iters(
                G, B[:, i], X[:, i], maxiters=self.maxiters, rtol=rtol
            )

        info = {"rmses": rmses(A, X, Y), "iterations": iters}
        return X if matrix_in else X.ravel(), info

    @staticmethod
    def _conjgrad_iters(calcAx, b, x, maxiters=None, rtol=1e-6):
        """Solve a single-RHS linear system using conjugate gradient."""

        if maxiters is None:
            maxiters = b.shape[0]

        r = b - calcAx(x)
        p = r.copy()
        rsold = np.dot(r, r)

        for i in range(maxiters):
            Ap = calcAx(p)
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = np.dot(r, r)
            beta = rsnew / rsold

            if np.sqrt(rsnew) < rtol:
                break

            if beta < 1e-12:  # no perceptible change in p
                break

            # p = r + beta*p
            p *= beta
            p += r
            rsold = rsnew

        return x, i + 1


class BlockConjgrad(LeastSquaresSolver):
    """Solve a multiple-RHS least-squares system using block conj. gradient."""

    tol = NumberParam("tol", low=0)
    X0 = NdarrayParam("X0", shape=("*", "*"), optional=True)

    def __init__(self, tol=1e-2, X0=None):
        super().__init__()
        self.tol = tol
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)
        sigma = np.asarray(sigma, dtype="float")
        sigma = sigma.reshape(sigma.size, 1)

        X = np.zeros((n, d)) if self.X0 is None else np.array(self.X0)
        if X.shape != (n, d):
            raise ValidationError(
                "Must be shape %s, got %s" % ((n, d), X.shape), attr="X0", obj=self
            )

        damp = m * sigma ** 2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        # --- conjugate gradient
        R = B - G(X)
        P = np.array(R)
        Rsold = np.dot(R.T, R)
        AP = np.zeros((n, d))

        maxiters = int(n / d) + 1
        for i in range(maxiters):
            AP = G(P)
            alpha = np.linalg.solve(np.dot(P.T, AP), Rsold)
            X += np.dot(P, alpha)
            R -= np.dot(AP, alpha)

            Rsnew = np.dot(R.T, R)
            if (np.diag(Rsnew) < rtol ** 2).all():
                break

            beta = np.linalg.solve(Rsold, Rsnew)
            P = R + np.dot(P, beta)
            Rsold = Rsnew

        info = {"rmses": rmses(A, X, Y), "iterations": i + 1}
        return X if matrix_in else X.ravel(), info


class SVD(LeastSquaresSolver):
    """Solve a least-squares system using full SVD."""

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, _, _, matrix_in = format_system(A, Y)
        U, s, V = np.linalg.svd(A, full_matrices=0)
        si = s / (s ** 2 + m * sigma ** 2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {"rmses": npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.ravel(), info


class RandomizedSVD(LeastSquaresSolver):
    """Solve a least-squares system using a randomized (partial) SVD.

    Useful for solving large matrices quickly, but non-optimally.

    Parameters
    ----------
    n_components : int, optional
        The number of SVD components to compute. A small survey of activity
        matrices suggests that the first 60 components capture almost all
        the variance.
    n_oversamples : int, optional
        The number of additional samples on the range of A.
    n_iter : int, optional
        The number of power iterations to perform (can help with noisy data).

    See also
    --------
    sklearn.utils.extmath.randomized_svd : Function used by this class
    """

    n_components = IntParam("n_components", low=1)
    n_oversamples = IntParam("n_oversamples", low=0)
    n_iter = IntParam("n_iter", low=0)

    def __init__(self, n_components=60, n_oversamples=10, n_iter=0):
        from sklearn.utils.extmath import (  # pylint: disable=import-outside-toplevel
            randomized_svd,
        )

        assert randomized_svd
        super().__init__()
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter

    def __call__(self, A, Y, sigma, rng=np.random):
        from sklearn.utils.extmath import (  # pylint: disable=import-outside-toplevel
            randomized_svd,
        )

        Y, m, n, _, matrix_in = format_system(A, Y)
        if min(m, n) <= self.n_components + self.n_oversamples:
            # more efficient to do a full SVD
            return SVD()(A, Y, sigma, rng=rng)

        U, s, V = randomized_svd(
            A,
            self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=rng,
        )
        si = s / (s ** 2 + m * sigma ** 2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {"rmses": npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.ravel(), info


class LeastSquaresSolverParam(Parameter):
    def coerce(self, instance, solver):
        self.check_type(instance, solver, LeastSquaresSolver)
        return super().coerce(instance, solver)
