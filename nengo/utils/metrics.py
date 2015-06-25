"""Metrics for comparing spike trains."""

from __future__ import absolute_import

import numpy as np

from nengo.utils.numpy import array


def _victor_purpura(a, b, q=1, dt=0.001):
    """Helper for computing Victor-Purpura measure on two spike trains."""
    assert len(a) == len(b)
    assert a.ndim == b.ndim == 1
    n = len(a)
    dp = np.zeros((n+1, n+1))

    # Boundary conditions (base cases)
    # This ensures that there are no spikes left in a[i:] or b[j:], otherwise
    # the cost is infinite because the trains weren't made equal.
    for i in range(n-1, -1, -1):
        dp[i, -1] = np.inf if a[i] else dp[i+1, -1]
    for j in range(n-1, -1, -1):
        dp[-1, j] = np.inf if b[j] else dp[-1, j+1]

    # Dynamic programming (recurrence relations)
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            if a[i] and b[j]:
                # Align spikes a[i] with b[j], or remove both
                dp[i][j] = min(dp[i+1][j+1] + abs(i-j)*q*dt, 2)
            elif a[i]:
                # Remove a[i] (or add b[j])
                dp[i][j] = min(dp[i][j+1], dp[i+1][j+1] + 1)
            elif b[j]:
                # Remove b[j] (or add a[i])
                dp[i][j] = min(dp[i+1][j], dp[i+1][j+1] + 1)
            else:
                # No operation necessary
                dp[i][j] = dp[i+1][j+1]
    return dp[0][0]


def victor_purpura(a, b, q=1, dt=0.001):
    """Computes pairwise Victor-Purpura measure of spike-train synchrony.

    The method is given in [1]_, inspired by the edit distance of genetic
    sequences. The cost of aligning two spike trains is computed by the
    shortest edit path consisting of operations:
     - Add/remove a spike for cost 1
     - Shift a spike for cost q*dt

    The shortest path is computed efficiently using dynamic programming, with
    state given by the two subsequences.

    Note that this function is symmetric.

    Parameters
    ----------
    a : ndarray (`m`, `ka`) or (`m`)
        Spike train(s) of length `m`. Can optionally provide `ka` such trains.
    b : ndarray (`m`, `kb`) or (`m`)
        Spike train(s) of length `m`. Can optionally provide `kb` such trains.
    q : float, optional
        Cost of shifting a spike by one timestep. Defaults to 1.
    dt : float, optional
        Timestep of recorded spikes in seconds. Defaults to 0.001.

    Returns
    -------
    ndarray (`ka`, `kb`)
        Array of Victor-Purpura distances, where entry [i, j] compares
        train a[:, i] with b[:, j].

    References
    ----------
    .. [1] Victor, Jonathan D., and Keith P. Purpura. "Nature and precision of
       temporal coding in visual cortex: a metric-space analysis." Journal of
       neurophysiology 76.2 (1996): 1310-1326.
    """

    a = array(a, min_dims=2)
    b = array(b, min_dims=2)

    if a.shape[0] != b.shape[0]:
        raise ValueError("Mismatch in length of spike trains: a is %d while "
                         "b is %d." % (a.shape[0], b.shape[0]))

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input arrays must be 1 or 2 dimensional.")

    out = np.zeros((a.shape[1], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            out[i, j] = _victor_purpura(a[:, i], b[:, j], q, dt)

    return out
