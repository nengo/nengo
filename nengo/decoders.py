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


def lstsq_L2nz(A, Y, rng, E=None, noise_amp=0.1):
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


def lstsq_L1(A, Y, rng, E=None, l1=1e-4, l2=1e-6):
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
    """
    Find the least-squares solution of the given linear system(s)
    using the Cholesky decomposition.
    """
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
