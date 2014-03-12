"""
Functions concerned with solving for decoders or full weight matrices.

Many of the solvers in this file can solve for decoders or weight matrices,
depending on whether the post-population encoders `E` are provided (see below).
Solvers that are only intended to solve for either decoders or weights can
remove the `E` parameter or make it manditory as they see fit.

All solvers take following arguments:
  A : array_like (M, N)
    Matrix of the N neurons' activities at the M evaluation points
  Y : array_like (M, D)
    Matrix of the target decoded values for each of the D dimensions,
    at each of the M evaluation points.

All solvers have the following optional keyword parameters:
  rng : numpy.RandomState
    A random number generator to use as required. If none is provided,
    numpy.random will be used.
  E : array_like (D, N2)
    Array of post-population encoders. Providing this tells the solver
    to return an array of connection weights rather than decoders.

All solvers return the following:
  X : np.ndarray (N, D) or (N, N2)
    (N, D) array of decoders if E is none, or (N, N2) array of weights
    if E is not none.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import scipy.linalg
    import scipy.optimize
except ImportError:
    logger.info("Failed to import 'scipy'")
    scipy = None

try:
    import sklearn.linear_model
except ImportError:
    logger.info("Failed to import 'sklearn'")
    sklearn = None


def lstsq(A, Y, rng=np.random, E=None, rcond=0.01):
    """Unregularized least-squares."""
    Y = np.dot(Y, E) if E is not None else Y
    X, res, rank, s = np.linalg.lstsq(A, Y, rcond=rcond)
    return X


def lstsq_noise(A, Y, rng=np.random, E=None, noise_amp=0.1):
    """Least-squares with additive white noise."""
    sigma = noise_amp * A.max()
    A = A + rng.normal(scale=sigma, size=A.shape)
    Y = np.dot(Y, E) if E is not None else Y
    return _cholesky(A, Y, 0)


def lstsq_multnoise(A, Y, rng=np.random, E=None, noise_amp=0.1):
    """Least-squares with multiplicative white noise."""
    A = A + rng.normal(scale=noise_amp, size=A.shape) * A
    Y = np.dot(Y, E) if E is not None else Y
    return _cholesky(A, Y, 0)


def lstsq_L2(A, Y, rng=np.random, E=None, noise_amp=0.1):
    """Least-squares with L2 regularization."""
    Y = np.dot(Y, E) if E is not None else Y
    sigma = noise_amp * A.max()
    return _cholesky(A, Y, sigma)


def lstsq_L2nz(A, Y, rng=np.random, E=None, noise_amp=0.1):
    """Least-squares with L2 regularization on non-zero components."""
    Y = np.dot(Y, E) if E is not None else Y

    # Compute the equivalent noise standard deviation. This equals the
    # base amplitude (noise_amp times the overall max activation) times
    # the square-root of the fraction of non-zero components.
    m, n = A.shape
    sigma = (noise_amp * A.max()) * np.sqrt(
        (A > 0).mean(axis=1 if m < n else 0))

    # sigma == 0 means the neuron is never active, so won't be used, but
    # we have to make sigma != 0 for numeric reasons.
    sigma[sigma == 0] = 1

    # Solve the LS problem using the Cholesky decomposition
    return _cholesky(A, Y, sigma)


def lstsq_L1(A, Y, rng=np.random, E=None, l1=1e-4, l2=1e-6):
    """Least-squares with L1 and L2 regularization (elastic net).

    This method is well suited for creating sparse decoders or weight matrices.
    """
    if sklearn is None:
        raise RuntimeError(
            "'lstsq_L1' requires the 'sklearn' package to be installed")

    # TODO: play around with these regularization constants (I just guessed).
    #   Do we need to scale regularization by number of neurons, to get same
    #   level of sparsity? esp. with weights? Currently, setting l1=1e-3 works
    #   well with weights when connecting 1D populations with 100 neurons each.
    a = l1 * A.max()      # L1 regularization
    b = l2 * A.max()**2   # L2 regularization
    alpha = a + b
    l1_ratio = a / (a + b)

    ### solve least-squares A * X = Y
    if E is not None:
        Y = np.dot(Y, E)

    model = sklearn.linear_model.ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=1000)
    model.fit(A, Y)
    X = model.coef_.T
    return X


# def lstsq_L1b(A, Y, rng=np.random, E=None, l1=1e-4):
#     """Least-squares with L1 regularization."""
#     if scipy is None:
#         raise RuntimeError(
#             "'lstsq_L1' requires the 'scipy' package to be installed")

#     a = l1 * A.max()      # L1 regularization

#     ### solve least-squares A * X = Y
#     if E is not None:
#         Y = np.dot(Y, E)

#     X, iters = _nonlinear_cg_L1(A, Y, a)
#     print "iters", iters
#     return X


def lstsq_drop(A, Y, rng, E=None, noise_amp=0.1, drop=0.25, solver=lstsq_L2nz):
    """Find sparser decoders/weights by dropping small values.

    This solver first solves for coefficients (decoders/weights) with
    L2 regularization, drops those nearest to zero, and retrains remaining.
    """

    # solve for coefficients using standard solver
    X = solver(A, Y, rng=rng, noise_amp=noise_amp)
    if E is not None:
        X = np.dot(X, E)

    # drop weights close to zero, based on `drop` ratio
    Xabs = np.sort(np.abs(X.flat))
    threshold = Xabs[int(np.round(drop * Xabs.size))]
    X[np.abs(X) < threshold] = 0

    # retrain nonzero weights
    if E is not None:
        Y = np.dot(Y, E)

    for i in xrange(X.shape[1]):
        nonzero = X[:, i] != 0
        if nonzero.sum() > 0:
            X[nonzero, i] = solver(A[:, nonzero], Y[:, i],
                                   rng=rng, noise_amp=0.1 * noise_amp)

    return X


def _cholesky(A, b, sigma, transpose=None):
    """Solve the least-squares system using the Cholesky decomposition."""
    m, n = A.shape
    transpose = m < n if transpose is None else transpose
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A)
        b = np.dot(A.T, b)

    reglambda = sigma ** 2 * m  # regularization parameter lambda
    np.fill_diagonal(G, G.diagonal() + reglambda)

    if scipy is not None:
        factor = scipy.linalg.cho_factor(G, overwrite_a=True)
        x = scipy.linalg.cho_solve(factor, b)
    else:
        L = np.linalg.cholesky(G)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, b))

    return np.dot(A.T, x) if transpose else x


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


def _conjgrad(A, B, sigma, X0=None, maxiters=None, tol=1e-2):
    """Solve the least-squares system using conjugate gradient."""

    m, n = A.shape
    damp = m * sigma**2

    G = lambda x: np.dot(A.T, np.dot(A, x)) + damp*x
    B = np.dot(A.T, B)

    rtol = tol * np.sqrt(m)

    if B.ndim == 1:
        x = np.zeros(n) if X0 is None else np.array(X0).reshape(n)
        x, iters = _conjgrad_iters(G, B, x, maxiters=maxiters, rtol=rtol)
        return x, iters
    elif B.ndim == 2:
        d = B.shape[1]
        X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
        iters = -np.ones(d, dtype='int')

        for i in range(d):
            X[:, i], iters[i] = _conjgrad_iters(
                G, B[:, i], X[:, i], maxiters=maxiters, rtol=rtol)

        return X, iters
    else:
        raise ValueError("B must be vector or matrix")


def _block_conjgrad(A, B, sigma, X0=None, tol=1e-2):
    """Solve a least-squares system with multiple-RHS."""

    m, n = A.shape
    matrix_in = B.ndim > 1
    d = B.shape[1] if matrix_in else 1
    # if d == 1:
    #     return conjgrad(A, B, sigma, X0)

    if not matrix_in:
        B = B.reshape((-1, 1))

    # G = lambda X: (np.dot(A.T, np.dot(A, X)) + (m * sigma**2) * X)
    G = lambda X: (np.dot(np.dot(A, X).T, A).T + (m * sigma**2) * X)  # faster
    B = np.dot(A.T, B)

    rtol = tol * np.sqrt(m)

    ### conjugate gradient
    X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
    R = B - G(X)
    P = np.array(R)
    Rsold = np.dot(R.T, R)
    AP = np.zeros((n, d))

    maxiters = int(n / d)
    for i in range(maxiters):
        # AP = G(P)
        for j in range(d):  # why is this loop faster than matrix-multiply?
            AP[:, j] = G(P[:, j])

        alpha = np.linalg.solve(np.dot(P.T, AP), Rsold)
        X += np.dot(P, alpha)
        R -= np.dot(AP, alpha)

        Rsnew = np.dot(R.T, R)
        if (np.diag(Rsnew) < rtol**2).all():
            break

        beta = np.linalg.solve(Rsold, Rsnew)
        P = R + np.dot(P, beta)
        Rsold = Rsnew.copy()

    if matrix_in:
        return X, i+1
    else:
        return X.flatten(), i+1


# def _nonlinear_cg_L1(A, B, sigma, x0=None):

#     m, n = A.shape
#     matrix_in = B.ndim > 1
#     d = B.shape[1] if matrix_in else 1

#     damp = np.sqrt(m) * sigma

#     X = np.zeros((n, d)) if x0 is None else np.array(x0)
#     X.shape = (n, d)
#     iters = np.zeros(d, dtype='int')

#     for i in xrange(d):
#         X[:, i], iters[i] = _nonlinear_cg_L1_iters(A, B[:, i], X[:, i], damp)

#     if matrix_in:
#         return X, i+1
#     else:
#         return X.flatten(), i+1


# def _nonlinear_cg_L1_iters(A, b, x, damp, maxiters=1000):

#     def vnorm(v):
#         return np.sqrt(np.dot(v, v))

#     p = None
#     calcAAx = lambda x: np.dot(A.T, np.dot(A, x))
#     bA = np.dot(b, A)
#     bs = np.dot(b, b)

#     for i in xrange(maxiters):

#         AAx = calcAAx(x)
#         r = bA - AAx
#         # print np.dot(r, r)


#         ### L1 norm
#         # dx = 2*np.dot(b - np.dot(A,x),A) + np.sqrt(m)*sigma*np.sign(x)
#         # dx = 2*(bA - np.dot(G,x)) - damp * np.sign(x)
#         # dx = 2 * (bA - calcAAx(x)) - damp * np.sign(x)
#         dx = 2 * (bA - AAx) - damp * np.sign(x)

#         if p is None:
#             p = np.array(dx)
#         else:
#             beta = np.dot(dx, dx - dx0) / np.dot(dx0, dx0)
#             beta = max(beta, 0)
#             p = dx + beta * p

#         # perform line search to find alpha (using secant method)
#         bAp = np.dot(bA, p)
#         AAp = calcAAx(p)
#         xAAp = np.dot(x, AAp)
#         pAAp = np.dot(p, AAp)
#         xp = np.dot(x, p)
#         pp = np.dot(p, p)

#         ### L1 norm
#         def fa(alpha):
#             return 2*pAAp*alpha + 2*(xAAp - bAp) + damp * np.dot(
#                 np.sign(x + alpha*p), p)


#         alpha0 = -1e-3 / vnorm(p)
#         alpha1 = 1e-3 / vnorm(p)
#         f0 = fa(alpha0)
#         f1 = fa(alpha1)

#         j = 0
#         while f0*f1 > 0:
#             # alpha0 *= 2
#             alpha1 *= 2
#             # f0, f1 = fa(alpha0), fa(alpha1)
#             f1 = fa(alpha1)
#             if j > 99:
#                 raise Exception('could not find starting point')
#             j += 1

#         alpha = scipy.optimize.brentq(fa, alpha0, alpha1)

#         x += alpha*p
#         dx0 = dx.copy()
#         # dx0 = dx

#         if vnorm(alpha*p) / vnorm(x) < 1e-8:
#             break

#     return x, i+1
