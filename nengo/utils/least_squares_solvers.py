from __future__ import absolute_import

import numpy as np

import nengo.utils.numpy as npext
from nengo.exceptions import ValidationError
from nengo.params import Parameter


def format_system(A, Y):
    m, n = A.shape
    matrix_in = Y.ndim > 1
    d = Y.shape[1] if matrix_in else 1
    Y = Y.reshape((Y.shape[0], d))
    return Y, m, n, d, matrix_in


def rmses(A, X, Y):
    """Returns the root-mean-squared error (RMSE) of the solution X."""
    return npext.rms(Y - np.dot(A, X), axis=0)


class LeastSquaresSolver(object):
    """Linear least squares system solver."""

    def __call__(self, A, y, sigma, rng=None):
        raise NotImplementedError("LeastSquaresSolver must implement call")


class Cholesky(LeastSquaresSolver):
    """Solve a least-squares system using the Cholesky decomposition."""

    def __init__(self, transpose=None):
        self.transpose = transpose

    def __call__(self, A, y, sigma, rng=None):
        m, n = A.shape
        transpose = self.transpose
        if transpose is None:
            # transpose if matrix is fat, but not if sigmas for each neuron
            transpose = m < n and sigma.size == 1

        if transpose:
            # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
            G = np.dot(A, A.T)
            b = y
        else:
            # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
            G = np.dot(A.T, A)
            b = np.dot(A.T, y)

        # add L2 regularization term 'lambda' = m * sigma**2
        np.fill_diagonal(G, G.diagonal() + m * sigma**2)

        try:
            import scipy.linalg
            factor = scipy.linalg.cho_factor(G, overwrite_a=True)
            x = scipy.linalg.cho_solve(factor, b)
        except ImportError:
            L = np.linalg.cholesky(G)
            L = np.linalg.inv(L.T)
            x = np.dot(L, np.dot(L.T, b))

        x = np.dot(A.T, x) if transpose else x
        info = {'rmses': rmses(A, x, y)}
        return x, info


class ConjgradScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's conjugate gradient."""

    def __init__(self, tol=1e-4):
        self.tol = tol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg
        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = m * sigma**2
        calcAA = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        G = scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=calcAA, matmat=calcAA, dtype=A.dtype)
        B = np.dot(A.T, Y)

        X = np.zeros((n, d), dtype=B.dtype)
        infos = np.zeros(d, dtype='int')
        itns = np.zeros(d, dtype='int')
        for i in range(d):
            # use the callback to count the number of iterations
            def callback(x):
                itns[i] += 1

            X[:, i], infos[i] = scipy.sparse.linalg.cg(
                G, B[:, i], tol=self.tol, callback=callback)

        info = {'rmses': rmses(A, X, Y), 'iterations': itns, 'info': infos}
        return X if matrix_in else X.flatten(), info


class LSMRScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's LSMR."""

    def __init__(self, tol=1e-4):
        self.tol = tol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg
        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = sigma * np.sqrt(m)
        X = np.zeros((n, d), dtype=Y.dtype)
        itns = np.zeros(d, dtype='int')
        for i in range(d):
            X[:, i], _, itns[i], _, _, _, _, _ = scipy.sparse.linalg.lsmr(
                A, Y[:, i], damp=damp, atol=self.tol, btol=self.tol)

        info = {'rmses': rmses(A, X, Y), 'iterations': itns}
        return X if matrix_in else X.flatten(), info


class Conjgrad(LeastSquaresSolver):
    """Solve a least-squares system using conjugate gradient."""

    def __init__(self, tol=1e-2, maxiters=None, X0=None):
        self.tol = tol
        self.maxiters = maxiters
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = m * sigma**2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        X = (np.zeros((n, d)) if self.X0 is None else
             np.array(self.X0).reshape((n, d)))
        iters = -np.ones(d, dtype='int')
        for i in range(d):
            X[:, i], iters[i] = self._conjgrad_iters(
                G, B[:, i], X[:, i], maxiters=self.maxiters, rtol=rtol)

        info = {'rmses': rmses(A, X, Y), 'iterations': iters}
        return X if matrix_in else X.flatten(), info

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

        return x, i+1


class BlockConjgrad(LeastSquaresSolver):
    """Solve a multiple-RHS least-squares system using block conj. gradient."""

    def __init__(self, tol=1e-2, X0=None):
        self.tol = tol
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)
        sigma = np.asarray(sigma, dtype='float')
        sigma = sigma.reshape(sigma.size, 1)

        damp = m * sigma**2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        # --- conjugate gradient
        X = (np.zeros((n, d)) if self.X0 is None else
             np.array(self.X0).reshape((n, d)))
        R = B - G(X)
        P = np.array(R)
        Rsold = np.dot(R.T, R)
        AP = np.zeros((n, d))

        maxiters = int(n / d)
        for i in range(maxiters):
            AP = G(P)
            alpha = np.linalg.solve(np.dot(P.T, AP), Rsold)
            X += np.dot(P, alpha)
            R -= np.dot(AP, alpha)

            Rsnew = np.dot(R.T, R)
            if (np.diag(Rsnew) < rtol**2).all():
                break

            beta = np.linalg.solve(Rsold, Rsnew)
            P = R + np.dot(P, beta)
            Rsold = Rsnew

        info = {'rmses': rmses(A, X, Y), 'iterations': i + 1}
        return X if matrix_in else X.flatten(), info


class SVD(LeastSquaresSolver):
    """Solve a least-squares system using full SVD."""

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, _, _, matrix_in = format_system(A, Y)
        U, s, V = np.linalg.svd(A, full_matrices=0)
        si = s / (s**2 + m * sigma**2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.flatten(), info


class RandomizedSVD(LeastSquaresSolver):
    """Solve a least-squares system using a randomized (partial) SVD.

    Parameters
    ----------
    n_components : int (default is 60)
        The number of SVD components to compute. A small survey of activity
        matrices suggests that the first 60 components capture almost all
        the variance.
    n_oversamples: int (default is 10)
        The number of additional samples on the range of A.
    n_iter : int (default is 0)
        The number of power iterations to perform (can help with noisy data).

    See also
    --------
    ``sklearn.utils.extmath.randomized_svd`` for details about the parameters.
    """

    def __init__(self, n_components=60, n_oversamples=10, **kwargs):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.kwargs = kwargs

    def __call__(self, A, Y, sigma, rng=np.random):
        from sklearn.utils.extmath import randomized_svd

        Y, m, n, _, matrix_in = format_system(A, Y)
        if min(m, n) <= self.n_components + self.n_oversamples:
            # more efficient to do a full SVD
            return SVD()(A, Y, sigma, rng=rng)

        U, s, V = randomized_svd(
            A, self.n_components, n_oversamples=self.n_oversamples,
            random_state=rng, **self.kwargs)
        si = s / (s**2 + m * sigma**2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.flatten(), info


class LeastSquaresSolverParam(Parameter):
    def validate(self, instance, solver):
        if solver is not None and not isinstance(solver, LeastSquaresSolver):
            raise ValidationError(
                "'%s' is not a least-squares subsolver "
                "(see ``nengo.solvers.lstsq`` for options)" % solver,
                attr=self.name, obj=instance)
        super(LeastSquaresSolverParam, self).validate(instance, solver)
