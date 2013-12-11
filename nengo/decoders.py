"""
This file contains functions concerned with solving for decoders
or other types of weight matrices.

The idea is that as we develop new methods to find decoders either more
quickly or with different constraints (e.g., L1-norm regularization),
the associated functions will be placed here.
"""

import numpy as np


def sample_hypersphere(dimensions, n_samples, rng, surface=False):
    """Generate sample points from a hypersphere.

    Returns float array of sample points: dimensions x n_samples

    """
    samples = rng.randn(n_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, np.newaxis]

    if surface:
        return samples

    # generate magnitudes for vectors from uniform distribution
    scale = rng.rand(n_samples, 1) ** (1.0 / dimensions)

    # scale sample points
    samples *= scale

    return samples

DEFAULT_RCOND = 0.01


def lstsq_L2(activities, targets, rng, noise_amp=0.1):
    """Least-squares with L2 regularization."""
    sigma = noise_amp * activities.max()
    return _cholesky1(activities, targets, sigma).T


def lstsq_L2nz(activities, targets, rng, noise_amp=0.1):
    """Least-squares with L2 regularization on non-zero components."""

    # Compute the equivalent noise standard deviation. This equals the
    # base amplitude (noise_amp times the overall max activation) times
    # the square-root of the fraction of non-zero components.
    sigma = (noise_amp * activities.max()) * np.sqrt((activities > 0).mean(0))

    # sigma == 0 means the neuron is never active, so won't be used, but
    # we have to make sigma != 0 for numeric reasons.
    sigma[sigma == 0] = 1

    # Solve the LS problem using the Cholesky decomposition
    return _cholesky1(activities, targets, sigma).T


def _cholesky1(A, b, sigma):
    """Solve the given linear system(s) using the Cholesky decomposition."""
    reglambda = sigma ** 2 * A.shape[0]    # regularization parameter lambda

    G = np.dot(A.T, A)
    np.fill_diagonal(G, G.diagonal() + reglambda)
    b = np.dot(A.T, b)

    L = np.linalg.cholesky(G)
    L = np.linalg.inv(L.T)
    return np.dot(L, np.dot(L.T, b))

    # factor = sp.linalg.cho_factor(G, overwrite_a=True, check_finite=False)
    # return sp.linalg.cho_solve(factor, b, check_finite=False)
