"""
Functions concerned with solving for decoders or full weight matrices.

Many of the solvers in this file can solve for decoders or weight matrices,
depending on whether the post-population encoders `E` are provided (see below).
Solvers that are only intended to solve for either decoders or weights can
remove the `E` parameter or make it manditory as they see fit.
"""
import collections
import logging

import numpy as np

from nengo.params import Parameter
import nengo.utils.numpy as npext
from nengo.utils.compat import range, with_metaclass, iteritems
from nengo.utils.magic import DocstringInheritor

logger = logging.getLogger(__name__)


def cholesky(A, y, sigma, transpose=None):
    """Solve the least-squares system using the Cholesky decomposition."""
    m, n = A.shape
    if transpose is None:
        # transpose if matrix is fat, but not if we have sigmas for each neuron
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
    info = {'rmses': npext.rms(y - np.dot(A, x), axis=0)}
    return x, info


def conjgrad_scipy(A, Y, sigma, tol=1e-4):
    """Solve the least-squares system using Scipy's conjugate gradient."""
    import scipy.sparse.linalg
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = m * sigma**2
    calcAA = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    G = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=calcAA, matmat=calcAA, dtype=A.dtype)
    B = np.dot(A.T, Y)

    X = np.zeros((n, d), dtype=B.dtype)
    infos = np.zeros(d, dtype='int')
    itns = np.zeros(d, dtype='int')
    for i in range(d):
        def callback(x):
            itns[i] += 1  # use the callback to count the number of iterations

        X[:, i], infos[i] = scipy.sparse.linalg.cg(
            G, B[:, i], tol=tol, callback=callback)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': itns,
            'info': infos}
    return X if matrix_in else X.flatten(), info


def lsmr_scipy(A, Y, sigma, tol=1e-4):
    """Solve the least-squares system using Scipy's LSMR."""
    import scipy.sparse.linalg
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = sigma * np.sqrt(m)
    X = np.zeros((n, d), dtype=Y.dtype)
    itns = np.zeros(d, dtype='int')
    for i in range(d):
        X[:, i], _, itns[i], _, _, _, _, _ = scipy.sparse.linalg.lsmr(
            A, Y[:, i], damp=damp, atol=tol, btol=tol)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': itns}
    return X if matrix_in else X.flatten(), info


def _conjgrad_iters(calcAx, b, x, maxiters=None, rtol=1e-6):
    """Solve the single-RHS linear system using conjugate gradient."""

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


def conjgrad(A, Y, sigma, X0=None, maxiters=None, tol=1e-2):
    """Solve the least-squares system using conjugate gradient."""
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = m * sigma**2
    rtol = tol * np.sqrt(m)
    G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    B = np.dot(A.T, Y)

    X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
    iters = -np.ones(d, dtype='int')
    for i in range(d):
        X[:, i], iters[i] = _conjgrad_iters(
            G, B[:, i], X[:, i], maxiters=maxiters, rtol=rtol)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': iters}
    return X if matrix_in else X.flatten(), info


def block_conjgrad(A, Y, sigma, X0=None, tol=1e-2):
    """Solve a multiple-RHS least-squares system using block conjuate gradient.
    """
    Y, m, n, d, matrix_in = _format_system(A, Y)
    sigma = np.asarray(sigma, dtype='float')
    sigma = sigma.reshape(sigma.size, 1)

    damp = m * sigma**2
    rtol = tol * np.sqrt(m)
    G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    B = np.dot(A.T, Y)

    # --- conjugate gradient
    X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
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

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': i + 1}
    return X if matrix_in else X.flatten(), info


def _format_system(A, Y):
    m, n = A.shape
    matrix_in = Y.ndim > 1
    d = Y.shape[1] if matrix_in else 1
    Y = Y.reshape((Y.shape[0], d))
    return Y, m, n, d, matrix_in


class Solver(with_metaclass(DocstringInheritor)):
    """
    Decoder or weight solver.
    """

    def __call__(self, A, Y, rng=None, E=None):
        """Call the solver.

        Parameters
        ----------
        A : array_like (M, N)
            Matrix of the N neurons' activities at the M evaluation points
        Y : array_like (M, D)
            Matrix of the target decoded values for each of the D dimensions,
            at each of the M evaluation points.
        rng : numpy.RandomState, optional
            A random number generator to use as required. If none is provided,
            numpy.random will be used.
        E : array_like (D, N2), optional
            Array of post-population encoders. Providing this tells the solver
            to return an array of connection weights rather than decoders.

        Returns
        -------
        X : np.ndarray (N, D) or (N, N2)
            (N, D) array of decoders (if solver.weights == False) or
            (N, N2) array of weights (if solver.weights == True).
        info : dict
            A dictionary of information about the solve. All dictionaries have
            an 'rmses' key that contains RMS errors of the solve. Other keys
            are unique to particular solvers.
        """
        raise NotImplementedError("Solvers must implement '__call__'")

    def mul_encoders(self, Y, E):
        if self.weights:
            if E is None:
                raise ValueError("Encoders must be provided for weight solver")
            return np.dot(Y, E)
        else:
            if E is not None:
                raise ValueError("Encoders must be 'None' for decoder solver")
            return Y

    def __hash__(self):
        items = list(self.__dict__.items())
        items.sort(key=lambda item: item[0])

        hashes = []
        for k, v in items:
            if isinstance(v, np.ndarray):
                if v.size < 1e5:
                    a = v[:]
                    a.setflags(write=False)
                    hashes.append(hash(a))
                else:
                    raise ValueError("array is too large to hash")
            elif isinstance(v, collections.Iterable):
                hashes.append(hash(tuple(v)))
            elif isinstance(v, collections.Callable):
                hashes.append(hash(v.__code__))
            else:
                hashes.append(hash(v))

        return hash(tuple(hashes))

    def __str__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ', '.join("%s=%s" % (k, v) for k, v in iteritems(self.__dict__)))


class Lstsq(Solver):
    """Unregularized least-squares"""

    def __init__(self, weights=False, rcond=0.01):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        rcond : float, optional
            Cut-off ratio for small singular values (see `numpy.linalg.lstsq`).
        """
        self.rcond = rcond
        self.weights = weights

    def __call__(self, A, Y, rng=None, E=None):
        Y = self.mul_encoders(Y, E)
        X, residuals2, rank, s = np.linalg.lstsq(A, Y, rcond=self.rcond)
        return X, {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
                   'residuals': np.sqrt(residuals2),
                   'rank': rank,
                   'singular_values': s}


class _LstsqNoiseSolver(Solver):
    """Base for least-squares solvers with noise"""

    def __init__(self, weights=False, noise=0.1, solver=cholesky, **kwargs):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        noise : float, optional
            Amount of noise, as a fraction of the neuron activity.
        solver : callable, optional
            Subsolver to use for solving the least-squares problem.
        kwargs
            Additional arguments passed to `solver`.
        """
        self.weights = weights
        self.noise = noise
        self.solver = solver
        self.kwargs = kwargs


class LstsqNoise(_LstsqNoiseSolver):
    """Least-squares with additive Gaussian white noise."""

    def __call__(self, A, Y, rng=None, E=None):
        rng = np.random if rng is None else rng
        sigma = self.noise * A.max()
        A = A + rng.normal(scale=sigma, size=A.shape)
        X, info = self.solver(A, Y, 0, **self.kwargs)
        return self.mul_encoders(X, E), info


class LstsqMultNoise(_LstsqNoiseSolver):
    """Least-squares with multiplicative white noise."""

    def __call__(self, A, Y, rng=None, E=None):
        rng = np.random if rng is None else rng
        A = A + rng.normal(scale=self.noise, size=A.shape) * A
        X, info = self.solver(A, Y, 0, **self.kwargs)
        return self.mul_encoders(X, E), info


class _LstsqL2Solver(Solver):
    """Base for L2-regularized least-squares solvers"""

    def __init__(self, weights=False, reg=0.1, solver=cholesky, **kwargs):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        reg : float, optional
            Amount of regularization, as a fraction of the neuron activity.
        solver : callable, optional
            Subsolver to use for solving the least-squares problem.
        kwargs
            Additional arguments passed to `solver`.
        """
        self.weights = weights
        self.reg = reg
        self.solver = solver
        self.kwargs = kwargs


class LstsqL2(_LstsqL2Solver):
    """Least-squares with L2 regularization."""

    def __call__(self, A, Y, rng=None, E=None):
        sigma = self.reg * A.max()
        X, info = self.solver(A, Y, sigma, **self.kwargs)
        return self.mul_encoders(X, E), info


class LstsqL2nz(_LstsqL2Solver):
    """Least-squares with L2 regularization on non-zero components."""

    def __call__(self, A, Y, rng=None, E=None):
        # Compute the equivalent noise standard deviation. This equals the
        # base amplitude (noise_amp times the overall max activation) times
        # the square-root of the fraction of non-zero components.
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))

        # sigma == 0 means the neuron is never active, so won't be used, but
        # we have to make sigma != 0 for numeric reasons.
        sigma[sigma == 0] = sigma.max()

        X, info = self.solver(A, Y, sigma, **self.kwargs)
        return self.mul_encoders(X, E), info


class LstsqL1(Solver):
    """Least-squares with L1 and L2 regularization (elastic net).

    This method is well suited for creating sparse decoders or weight matrices.
    """
    def __init__(self, weights=False, l1=1e-4, l2=1e-6):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        l1 : float, optional
            Amount of L1 regularization.
        l2 : float, optional
            Amount of L2 regularization.
        """
        import sklearn.linear_model  # noqa F401, import to check existence
        assert sklearn.linear_model
        self.weights = weights
        self.l1 = l1
        self.l2 = l2

    def __call__(self, A, Y, rng=None, E=None):
        import sklearn.linear_model
        Y = self.mul_encoders(Y, E)

        # TODO: play around with regularization constants (I just guessed).
        #   Do we need to scale regularization by number of neurons, to get
        #   same level of sparsity? esp. with weights? Currently, setting
        #   l1=1e-3 works well with weights when connecting 1D populations
        #   with 100 neurons each.
        a = self.l1 * A.max()      # L1 regularization
        b = self.l2 * A.max()**2   # L2 regularization
        alpha = a + b
        l1_ratio = a / (a + b)

        # --- solve least-squares A * X = Y
        model = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=1000)
        model.fit(A, Y)
        X = model.coef_.T
        X.shape = (A.shape[1], Y.shape[1]) if Y.ndim > 1 else (A.shape[1],)
        infos = {'rmses': npext.rms(Y - np.dot(A, X), axis=0)}
        return X, infos


class LstsqDrop(Solver):
    """Find sparser decoders/weights by dropping small values.

    This solver first solves for coefficients (decoders/weights) with
    L2 regularization, drops those nearest to zero, and retrains remaining.
    """

    def __init__(self, weights=False, drop=0.25,
                 solver1=LstsqL2nz(reg=0.1), solver2=LstsqL2nz(reg=0.01)):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        drop : float, optional
            Fraction of decoders or weights to set to zero.
        solver1 : Solver, optional
            Solver for finding the initial decoders.
        solver2 : Solver, optional
            Used for re-solving for the decoders after dropout.
        """
        self.weights = weights
        self.drop = drop
        self.solver1 = solver1
        self.solver2 = solver2

    def __call__(self, A, Y, rng=None, E=None):
        Y, m, n, d, matrix_in = _format_system(A, Y)

        # solve for coefficients using standard solver
        X, info0 = self.solver1(A, Y, rng=rng)
        X = self.mul_encoders(X, E)

        # drop weights close to zero, based on `drop` ratio
        Xabs = np.sort(np.abs(X.flat))
        threshold = Xabs[int(np.round(self.drop * Xabs.size))]
        X[np.abs(X) < threshold] = 0

        # retrain nonzero weights
        Y = self.mul_encoders(Y, E)
        for i in range(X.shape[1]):
            nonzero = X[:, i] != 0
            if nonzero.sum() > 0:
                X[nonzero, i], info1 = self.solver2(
                    A[:, nonzero], Y[:, i], rng=rng)

        info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
                'info0': info0, 'info1': info1}
        return X if matrix_in else X.flatten(), info


class Nnls(Solver):
    """Non-negative least-squares without regularization.

    Similar to `lstsq`, except the output values are non-negative.
    """
    def __init__(self, weights=False):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        """
        import scipy.optimize  # import here too to throw error early
        assert scipy.optimize
        self.weights = weights

    def __call__(self, A, Y, rng=None, E=None):
        import scipy.optimize

        Y, m, n, d, matrix_in = _format_system(A, Y)
        Y = self.mul_encoders(Y, E)

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for i in range(d):
            X[:, i], residuals[i] = scipy.optimize.nnls(A, Y[:, i])

        info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
                'residuals': residuals}
        return X if matrix_in else X.flatten(), info


class NnlsL2(Nnls):
    """Non-negative least-squares with L2 regularization.

    Similar to `lstsq_L2`, except the output values are non-negative.
    """
    def __init__(self, weights=False, reg=0.1):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        reg : float, optional
            Amount of regularization, as a fraction of the neuron activity.
        """
        super(NnlsL2, self).__init__(weights)
        self.reg = reg

    def __call__(self, A, Y, rng=None, E=None):
        # form Gram matrix so we can add regularization
        sigma = self.reg * A.max()
        G = np.dot(A.T, A)
        Y = np.dot(A.T, Y)
        np.fill_diagonal(G, G.diagonal() + sigma)
        return super(NnlsL2, self).__call__(G, Y, rng=rng, E=E)


class NnlsL2nz(Nnls):
    """Non-negative least-squares with L2 regularization on nonzero components.

    Similar to `lstsq_L2nz`, except the output values are non-negative.
    """
    def __init__(self, weights=False, reg=0.1):
        """
        weights : boolean, optional
            If false solve for decoders (default), otherwise solve for weights.
        reg : float, optional
            Amount of regularization, as a fraction of the neuron activity.
        """
        super(NnlsL2nz, self).__init__(weights)
        self.reg = reg

    def __call__(self, A, Y, rng=None, E=None):
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))
        sigma[sigma == 0] = 1

        # form Gram matrix so we can add regularization
        G = np.dot(A.T, A)
        Y = np.dot(A.T, Y)
        np.fill_diagonal(G, G.diagonal() + sigma)
        return super(NnlsL2nz, self).__call__(G, Y, rng=rng, E=E)


class SolverParam(Parameter):
    def validate(self, instance, solver):
        if solver is not None and not isinstance(solver, Solver):
            raise ValueError("'%s' is not a solver" % solver)
        super(SolverParam, self).validate(instance, solver)
