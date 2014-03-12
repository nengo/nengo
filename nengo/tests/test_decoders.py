"""
TODO:
  - add a test to test each solver many times on different populations,
    and record the error.
"""

import logging

import numpy as np
import pytest

import nengo
from nengo.utils.distributions import UniformHypersphere
from nengo.utils.functions import filtfilt
from nengo.utils.numpy import rms
from nengo.utils.testing import Plotter, allclose
from nengo.decoders import (
    _cholesky, _conjgrad, _block_conjgrad,
    lstsq, lstsq_noise, lstsq_L2, lstsq_L2nz,
    lstsq_L1, lstsq_drop)

logger = logging.getLogger(__name__)


def sample_hypersphere(dimensions, n_samples, rng=np.random, surface=False):
    return UniformHypersphere(
        dimensions, surface=surface).sample(n_samples, rng=rng)


def test_cholesky():
    rng = np.random.RandomState(4829)

    m, n = 100, 100
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m, ))

    x0, _, _, _ = np.linalg.lstsq(A, b)
    x1 = _cholesky(A, b, 0, transpose=False)
    x2 = _cholesky(A, b, 0, transpose=True)
    assert np.allclose(x0, x1)
    assert np.allclose(x0, x2)


def test_conjgrad():
    rng = np.random.RandomState(4829)

    m, n = 100, 100
    d = 1
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m, d))

    sigma = 1
    x0 = _cholesky(A, b, sigma)
    x1, i = _conjgrad(A, b, sigma, tol=1e-3)
    # assert np.allclose(x0, x1, atol=1e-3, rtol=1e-5)
    assert np.allclose(x0, x1, atol=1e-5, rtol=1e-3)


def _get_AB(m, n, d, rng=np.random):
    """Return a system of LIF tuning curves."""
    E = sample_hypersphere(d, n, rng=rng, surface=True).T  # encoders
    B = sample_hypersphere(d, m, rng=rng)                  # eval points

    a = nengo.LIF(n)
    a.set_gain_bias(rng.uniform(50, 100, n), rng.uniform(-1, 1, n))
    A = a.rates(np.dot(B, E))
    return A, B


@pytest.mark.benchmark
def test_base_solvers_L2():
    rng = np.random.RandomState(39408)
    A, B = _get_AB(m=5000, n=3000, d=3, rng=rng)
    sigma = 0.1 * A.max()

    def time_solver(solver, *args, **kwargs):
        import time
        t = time.time()
        output = solver(*args, **kwargs)
        t = time.time() - t
        return output, t

    x1, t1 = time_solver(_cholesky, A, B, sigma)
    [x2, i2], t2 = time_solver(_conjgrad, A, B, sigma, tol=1e-2)
    [x3, i3], t3 = time_solver(_block_conjgrad, A, B, sigma, tol=1e-2)
    print(t1, t2, t3)
    print(i2, i3)

    assert np.allclose(x1, x2, atol=1e-5, rtol=1e-3)


@pytest.mark.benchmark
def test_base_solvers_L1():
    rng = np.random.RandomState(39408)
    A, B = _get_AB(m=500, n=100, d=1, rng=rng)

    def time_solver(solver, *args, **kwargs):
        import time
        t = time.time()
        output = solver(*args, **kwargs)
        t = time.time() - t
        return output, t

    l1 = 1e-4
    x0, t0 = time_solver(lstsq_L1, A, B, l1=l1, l2=0)
    print(t0)


def test_weights():
    solver = lstsq_L2
    rng = np.random.RandomState(39408)

    d = 2
    m, n = 100, 101
    n_samples = 1000

    a = nengo.LIF(m)  # population, for generating LIF tuning curves
    a.set_gain_bias(rng.uniform(50, 100, m), rng.uniform(-1, 1, m))

    ea = sample_hypersphere(d, m, rng=rng, surface=True).T  # pre encoders
    eb = sample_hypersphere(d, n, rng=rng, surface=True).T  # post encoders

    p = sample_hypersphere(d, n_samples, rng=rng)  # training eval points
    A = a.rates(np.dot(p, ea))                     # training activations
    X = p                                          # training targets

    # find decoders and multiply by encoders to get weights
    D = solver(A, X, rng=rng)
    W1 = np.dot(D, eb)

    # find weights directly
    W2 = solver(A, X, rng=rng, E=eb)

    # assert that post inputs are close on test points
    pt = sample_hypersphere(d, n_samples, rng=rng)  # testing eval points
    At = a.rates(np.dot(pt, ea))                    # testing activations
    Y1 = np.dot(At, W1)                             # post inputs from decoders
    Y2 = np.dot(At, W2)                             # post inputs from weights
    assert np.allclose(Y1, Y2)

    # assert that weights themselves are close (this is true for L2 weights)
    assert np.allclose(W1, W2)


@pytest.mark.benchmark
def test_solvers(Simulator, nl_nodirect):

    N = 100
    decoder_solvers = [lstsq, lstsq_noise, lstsq_L2, lstsq_L2nz, lstsq_L1]
    weight_solvers = [lstsq_L1, lstsq_drop]

    dt = 1e-3
    tfinal = 4

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    model = nengo.Model('test_solvers', seed=290)
    u = nengo.Node(output=input_function)
    # up = nengo.Probe(u)

    a = nengo.Ensemble(nl_nodirect(N), dimensions=1)
    ap = nengo.Probe(a)
    nengo.Connection(u, a)

    probes = []
    names = []
    for solver, arg in ([(s, 'decoder_solver') for s in decoder_solvers] +
                        [(s, 'weight_solver') for s in weight_solvers]):
        b = nengo.Ensemble(nl_nodirect(N), dimensions=1, seed=99)
        nengo.Connection(a, b, **{arg: solver})
        probes.append(nengo.Probe(b))
        names.append(solver.__name__ + (" (%s)" % arg[0]))

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    # ref = sim.data(up)
    ref = filtfilt(sim.data(ap), 20)
    outputs = np.array([sim.data(probe) for probe in probes])
    outputs_f = filtfilt(outputs, 20, axis=1)

    close = allclose(t, ref, outputs_f,
                     plotter=Plotter(Simulator, nl_nodirect),
                     filename='test_decoders.test_solvers.pdf',
                     labels=names,
                     atol=0.05, rtol=0, buf=100, delay=7)
    for name, c in zip(names, close):
        assert c, "Solver '%s' does not meet tolerances" % name


@pytest.mark.benchmark  # noqa: C901
def test_regularization(Simulator, nl_nodirect):

    ### TODO: multiple trials per parameter set, with different seeds

    solvers = [lstsq_L2, lstsq_L2nz]
    neurons = np.array([10, 20, 50, 100])
    regs = np.linspace(0.01, 0.3, 16)
    filters = np.linspace(0, 0.03, 11)

    buf = 0.2  # buffer for initial transients
    dt = 1e-3
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    # model = nengo.Model('test_solvers', seed=290)
    model = nengo.Model('test_regularization')
    u = nengo.Node(output=input_function)
    up = nengo.Probe(u)

    probes = np.zeros(
        (len(solvers), len(neurons), len(regs), len(filters)), dtype='object')

    for j, n_neurons in enumerate(neurons):
        a = nengo.Ensemble(nl_nodirect(n_neurons), dimensions=1)
        nengo.Connection(u, a)

        for i, solver in enumerate(solvers):
            for k, reg in enumerate(regs):
                reg_solver = lambda a, t, rng, reg=reg: solver(
                    a, t, rng=rng, noise_amp=reg)
                for l, filter in enumerate(filters):
                    probes[i, j, k, l] = nengo.Probe(
                        a, decoder_solver=reg_solver, filter=filter)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data(up)
    rmse_buf = lambda a, b: rms(a[t > buf] - b[t > buf])
    rmses = np.zeros(probes.shape)
    for i, probe in enumerate(probes.flat):
        rmses.flat[i] = rmse_buf(sim.data(probe), ref)
    rmses = rmses - rmses[:, :, [0], :]

    with Plotter(Simulator, nl_nodirect) as plt:
        plt.figure(figsize=(8, 12))
        X, Y = np.meshgrid(filters, regs)

        for i, solver in enumerate(solvers):
            for j, n_neurons in enumerate(neurons):
                plt.subplot(len(neurons), len(solvers), len(solvers)*j + i + 1)
                Z = rmses[i, j, :, :]
                plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 21))
                plt.xlabel('filter')
                plt.ylabel('reg')
                plt.title("%s (N=%d)" % (solver.__name__, n_neurons))

        plt.tight_layout()
        plt.savefig('test_decoders.test_regularization.pdf')
        plt.close()


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
