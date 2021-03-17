import numpy as np
import pytest

from nengo.exceptions import ValidationError
from nengo.utils.least_squares_solvers import (
    SVD,
    BlockConjgrad,
    Cholesky,
    Conjgrad,
    ConjgradScipy,
    LSMRScipy,
    RandomizedSVD,
)


def run_solver(solver, m=30, n=20, d=1, cond=10, sys_rng=np.random, **kwargs):
    A = sys_rng.uniform(-1, 1, size=(m, n))
    x = sys_rng.uniform(-1, 1, size=(n, d))

    # set condition number of system
    assert cond > 1
    U, s, VH = np.linalg.svd(A, full_matrices=False)
    smax, smin = s.max(), s.min()
    new_min = smax / cond
    s = (s - smin) / (smax - smin) * (smax - new_min) + new_min
    A = np.dot(U * s, VH)

    # solve system
    y = np.dot(A, x)
    x2, _ = solver(A, y, **kwargs)

    return x, x2, A


@pytest.mark.parametrize("cond, sigma", [(5, 1e-4), (500, 1e-6), (1e5, 1e-8)])
def test_cholesky(cond, sigma, rng, allclose):
    rng = np.random
    solver = Cholesky()
    x, x2, _ = run_solver(solver, d=2, cond=cond, sys_rng=rng, sigma=sigma)

    tol = np.sqrt(cond) * 1e-7  # this is a guesstimate, and also depends on sigma
    assert allclose(x2, x, atol=tol, rtol=tol)


@pytest.mark.parametrize("cond, sigma, tol", [(5, 1e-4, 1e-2), (50, 1e-5, 1e-4)])
def test_conjgrad_scipy(cond, sigma, tol, rng, allclose):
    pytest.importorskip("scipy")
    solver = ConjgradScipy(tol=tol)
    x, x2, _ = run_solver(solver, d=2, cond=cond, sys_rng=rng, sigma=sigma)

    tol = cond * tol
    assert allclose(x2, x, atol=tol, rtol=tol)


@pytest.mark.parametrize("cond, sigma, tol", [(5, 1e-3, 1e-4), (10, 1e-3, 1e-4)])
def test_lsmr_scipy(cond, sigma, tol, rng, allclose):
    pytest.importorskip("scipy")
    solver = LSMRScipy(tol=tol)
    x, x2, _ = run_solver(solver, d=2, cond=cond, sys_rng=rng, sigma=sigma)

    tol = cond * tol
    assert allclose(x2, x, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "cond, sigma, tol, maxiters",
    [
        (5, 1e-8, 1e-2, None),  # standard run
        (50, 1e-8, 1e-4, 100),  # precision run (extra iterations help convergence)
        (1.1, 1e-15, 0, 1000),  # hit "no perceptible change in p" line
    ],
)
def test_conjgrad(cond, sigma, tol, maxiters, rng, allclose):
    solver = Conjgrad(tol=tol, maxiters=maxiters)
    x, x2, A = run_solver(solver, d=2, cond=cond, sys_rng=rng, sigma=sigma)

    # conjgrad stopping tol is based on residual in y, so compare y (`y = A.dot(x)`)
    # (this particularly helps in the ill-conditioned case)
    tol = cond * max(tol, 1e-5)
    assert allclose(A.dot(x2), A.dot(x), atol=tol, rtol=tol)


def test_conjgrad_errors():
    solver = Conjgrad(X0=np.ones((1, 1)))
    with pytest.raises(ValidationError, match=r"Must be shape \(2, 2\)"):
        solver(A=np.ones((2, 2)), Y=np.ones((2, 2)), sigma=0.001)


@pytest.mark.parametrize("cond, tol", [(5, 1e-2), (100, 1e-3)])
def test_blockconjgrad(cond, tol, rng, allclose):
    x, x2, _ = run_solver(
        BlockConjgrad(tol=tol), d=5, cond=cond, sys_rng=rng, sigma=1e-8
    )
    assert allclose(x2, x, atol=tol, rtol=tol)


def test_blockconjgrad_errors():
    solver = BlockConjgrad(X0=np.ones((1, 1)))
    with pytest.raises(ValidationError, match=r"Must be shape \(2, 2\)"):
        solver(A=np.ones((2, 2)), Y=np.ones((2, 2)), sigma=0.1)


@pytest.mark.parametrize("cond", [5, 1000])
def test_svd(cond, rng, allclose):
    x, x2, _ = run_solver(SVD(), d=5, cond=cond, sys_rng=rng, sigma=1e-8)
    assert allclose(x2, x, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("cond", [5, 1000])
def test_randomized_svd_fallback(cond, rng, allclose):
    """Test the specific case where RandomizedSVD falls back to SVD"""
    pytest.importorskip("sklearn")
    m, n = 30, 20
    solver = RandomizedSVD(n_components=min(m, n))
    x, x2, _ = run_solver(solver, m=m, n=n, d=5, cond=cond, sys_rng=rng, sigma=1e-8)
    assert allclose(x2, x, atol=1e-8, rtol=1e-8)
