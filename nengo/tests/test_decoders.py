"""
TODO:
  - add a test to test each solver many times on different populations,
    and record the error.
"""
from __future__ import print_function
import logging

import numpy as np
import pytest

import nengo
from nengo.utils.distributions import UniformHypersphere
from nengo.utils.numpy import filtfilt, rms, norm
from nengo.utils.testing import Plotter, allclose, Timer
from nengo.decoders import (
    _cholesky, _conjgrad, _block_conjgrad, _conjgrad_scipy, _lsmr_scipy,
    lstsq, lstsq_noise, lstsq_L2, lstsq_L2nz,
    lstsq_L1, lstsq_drop,
    nnls, nnls_L2, nnls_L2nz)

logger = logging.getLogger(__name__)


def get_encoders(n_neurons, dims, rng=None):
    return UniformHypersphere(dims, surface=True).sample(n_neurons, rng=rng).T


def get_eval_points(n_points, dims, rng=None, sort=False):
    points = UniformHypersphere(dims, surface=False).sample(n_points, rng=rng)
    return points[np.argsort(points[:, 0])] if sort else points


def get_rate_function(n_neurons, dims, neuron_type=nengo.LIF, rng=None):
    neurons = neuron_type(n_neurons)
    gain, bias = neurons.gain_bias(
        rng.uniform(50, 100, n_neurons), rng.uniform(-1, 1, n_neurons))
    rates = lambda x: neurons.rates(x, gain, bias)
    return rates


def get_system(m, n, d, rng=None, sort=False):
    """Get a system of LIF tuning curves and the corresponding eval points."""
    eval_points = get_eval_points(m, d, rng=rng, sort=sort)
    encoders = get_encoders(n, d, rng=rng)
    rates = get_rate_function(n, d, rng=rng)
    return rates(np.dot(eval_points, encoders)), eval_points


def test_cholesky():
    rng = np.random.RandomState(4829)

    m, n = 100, 100
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m, ))

    x0, _, _, _ = np.linalg.lstsq(A, b)
    x1, _ = _cholesky(A, b, 0, transpose=False)
    x2, _ = _cholesky(A, b, 0, transpose=True)
    assert np.allclose(x0, x1)
    assert np.allclose(x0, x2)


def test_conjgrad():
    rng = np.random.RandomState(4829)
    A, b = get_system(1000, 100, 2, rng=rng)
    sigma = 0.1 * A.max()

    x0, _ = _cholesky(A, b, sigma)
    x1, _ = _conjgrad(A, b, sigma, tol=1e-3)
    x2, _ = _block_conjgrad(A, b, sigma, tol=1e-3)
    assert np.allclose(x0, x1, atol=1e-6, rtol=1e-3)
    assert np.allclose(x0, x2, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize('solver', [
    lstsq, lstsq_noise, lstsq_L2, lstsq_L2nz, lstsq_drop])
def test_decoder_solver(solver):
    rng = np.random.RandomState(39408)

    dims = 1
    n_neurons = 100
    n_points = 500

    rates = get_rate_function(n_neurons, dims, rng=rng)
    E = get_encoders(n_neurons, dims, rng=rng)

    train = get_eval_points(n_points, dims, rng=rng)
    Atrain = rates(np.dot(train, E))

    D, _ = solver(Atrain, train, rng=rng)

    test = get_eval_points(n_points, dims, rng=rng, sort=True)
    Atest = rates(np.dot(test, E))
    est = np.dot(Atest, D)
    rel_rmse = rms(est - test) / rms(test)

    with Plotter(nengo.Simulator) as plt:
        plt.plot(test, np.zeros_like(test), 'k--')
        plt.plot(test, test - est)
        plt.title("relative RMSE: %0.2e" % rel_rmse)
        plt.savefig('test_decoders.test_decoder_solver.%s.pdf'
                    % solver.__name__)
        plt.close()

    assert np.allclose(test, est, atol=3e-2, rtol=1e-3)
    assert rel_rmse < 0.02


@pytest.mark.parametrize('solver', [
    lstsq_noise, lstsq_L2, lstsq_L2nz])
def test_subsolvers(solver, tol=1e-2):
    rng = np.random.RandomState(89)
    get_rng = lambda: np.random.RandomState(87)

    A, b = get_system(500, 100, 5, rng=rng)
    x0, _ = solver(A, b, rng=get_rng(), solver=_cholesky)

    subsolvers = [_conjgrad, _block_conjgrad]
    for subsolver in subsolvers:
        x, info = solver(A, b, rng=get_rng(), solver=subsolver, tol=tol)
        rel_rmse = rms(x - x0) / rms(x0)
        assert rel_rmse < 3 * tol
        # the above 3 * tol is just a heuristic; the main purpose of this
        # test is to make sure that the subsolvers don't throw errors
        # in-situ. They are tested more robustly elsewhere.


@pytest.mark.optional
@pytest.mark.parametrize('solver', [lstsq_L1])
def test_decoder_solver_extra(solver):
    test_decoder_solver(solver)


@pytest.mark.parametrize('solver', [lstsq, lstsq_L2, lstsq_L2nz])
def test_weight_solver(solver):
    rng = np.random.RandomState(39408)

    dims = 2
    a_neurons, b_neurons = 100, 101
    n_points = 1000

    rates = get_rate_function(a_neurons, dims, rng=rng)
    Ea = get_encoders(a_neurons, dims, rng=rng)  # pre encoders
    Eb = get_encoders(b_neurons, dims, rng=rng)  # post encoders

    train = get_eval_points(n_points, dims, rng=rng)  # training eval points
    Atrain = rates(np.dot(train, Ea))                 # training activations
    Xtrain = train                                    # training targets

    # find decoders and multiply by encoders to get weights
    D, _ = solver(Atrain, Xtrain, rng=rng)
    W1 = np.dot(D, Eb)

    # find weights directly
    W2, _ = solver(Atrain, Xtrain, rng=rng, E=Eb)

    # assert that post inputs are close on test points
    test = get_eval_points(n_points, dims, rng=rng)  # testing eval points
    Atest = rates(np.dot(test, Ea))
    Y1 = np.dot(Atest, W1)                         # post inputs from decoders
    Y2 = np.dot(Atest, W2)                         # post inputs from weights
    assert np.allclose(Y1, Y2)

    # assert that weights themselves are close (this is true for L2 weights)
    assert np.allclose(W1, W2)


@pytest.mark.optional  # uses scipy
def test_scipy_solvers():
    rng = np.random.RandomState(4829)
    A, b = get_system(1000, 100, 2, rng=rng)
    sigma = 0.1 * A.max()

    x0, _ = _cholesky(A, b, sigma)
    x1, _ = _conjgrad_scipy(A, b, sigma)
    x2, _ = _lsmr_scipy(A, b, sigma)
    assert np.allclose(x0, x1, atol=1e-5, rtol=1e-3)
    assert np.allclose(x0, x2, atol=1e-5, rtol=1e-3)


@pytest.mark.optional  # uses scipy
@pytest.mark.parametrize('solver', [nnls, nnls_L2, nnls_L2nz])
def test_nnls(solver):
    rng = np.random.RandomState(39408)
    A, x = get_system(500, 100, 1, rng=rng, sort=True)
    y = x**2

    d, _ = solver(A, y, rng)
    yest = np.dot(A, d)
    rel_rmse = rms(yest - y) / rms(y)

    with Plotter(nengo.Simulator) as plt:
        plt.subplot(211)
        plt.plot(x, y, 'k--')
        plt.plot(x, yest)
        plt.ylim([-0.1, 1.1])
        plt.subplot(212)
        plt.plot(x, np.zeros_like(x), 'k--')
        plt.plot(x, yest - y)
        plt.savefig('test_decoders.test_nnls.%s.pdf' % solver.__name__)
        plt.close()

    assert np.allclose(yest, y, atol=3e-2, rtol=1e-3)
    assert rel_rmse < 0.02


@pytest.mark.benchmark
def test_base_solvers_L2():
    ref_solver = _cholesky
    solvers = [_conjgrad, _block_conjgrad, _conjgrad_scipy, _lsmr_scipy]

    rng = np.random.RandomState(39408)
    A, B = get_system(m=5000, n=3000, d=3, rng=rng)
    sigma = 0.1 * A.max()

    with Timer() as t0:
        x0, _ = ref_solver(A, B, sigma)

    xs = np.zeros((len(solvers),) + x0.shape)
    print()
    for i, solver in enumerate(solvers):
        with Timer() as t:
            xs[i], info = solver(A, B, sigma)
        print("%s: %0.3f (%0.2f) %s" % (
            solver.__name__, t.duration, t.duration / t0.duration, info))

    for solver, x in zip(solvers, xs):
        assert np.allclose(x0, x, atol=1e-5, rtol=1e-3), (
            "Solver %s" % solver.__name__)


@pytest.mark.benchmark
def test_base_solvers_L1():
    rng = np.random.RandomState(39408)
    A, B = get_system(m=500, n=100, d=1, rng=rng)

    l1 = 1e-4
    with Timer() as t:
        lstsq_L1(A, B, l1=l1, l2=0)
    print(t.duration)


@pytest.mark.benchmark
def test_solvers(Simulator, nl_nodirect):

    N = 100
    decoder_solvers = [lstsq, lstsq_noise, lstsq_L2, lstsq_L2nz, lstsq_L1]
    weight_solvers = [lstsq_L1, lstsq_drop]

    dt = 1e-3
    tfinal = 4

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    model = nengo.Network('test_solvers', seed=290)
    with model:
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

    sim = Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    # ref = sim.data[up]
    ref = filtfilt(sim.data[ap], 20)
    outputs = np.array([sim.data[probe] for probe in probes])
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

    # TODO: multiple trials per parameter set, with different seeds

    solvers = [lstsq_L2, lstsq_L2nz]
    neurons = np.array([10, 20, 50, 100])
    regs = np.linspace(0.01, 0.3, 16)
    filters = np.linspace(0, 0.03, 11)

    buf = 0.2  # buffer for initial transients
    dt = 1e-3
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    model = nengo.Network('test_regularization')
    with model:
        u = nengo.Node(output=input_function)
        up = nengo.Probe(u)

        probes = np.zeros(
            (len(solvers), len(neurons), len(regs), len(filters)),
            dtype='object')

        for j, n_neurons in enumerate(neurons):
            a = nengo.Ensemble(nl_nodirect(n_neurons), dimensions=1)
            nengo.Connection(u, a)

            for i, solver in enumerate(solvers):
                for k, reg in enumerate(regs):
                    reg_solver = lambda a, t, rng, reg=reg: solver(
                        a, t, rng=rng, noise_amp=reg)
                    for l, synapse in enumerate(filters):
                        probes[i, j, k, l] = nengo.Probe(
                            a, decoder_solver=reg_solver, synapse=synapse)

    sim = Simulator(model, dt=dt)
    sim.run(tfinal)
    t = sim.trange()

    ref = sim.data[up]
    rmse_buf = lambda a, b: rms(a[t > buf] - b[t > buf])
    rmses = np.zeros(probes.shape)
    for i, probe in enumerate(probes.flat):
        rmses.flat[i] = rmse_buf(sim.data[probe], ref)
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


@pytest.mark.benchmark
def test_eval_points_static(Simulator):
    solver = lstsq_L2

    rng = np.random.RandomState(0)
    n = 100
    d = 5

    eval_points = np.logspace(np.log10(300), np.log10(5000), 21)
    eval_points = np.round(eval_points).astype('int')
    max_points = eval_points.max()
    n_trials = 25
    # n_trials = 100

    rmses = np.nan * np.zeros((len(eval_points), n_trials))

    for trial in range(n_trials):
        # make a population for generating LIF tuning curves
        a = nengo.LIF(n)
        gain, bias = a.gain_bias(
            # rng.uniform(50, 100, n), rng.uniform(-1, 1, n))
            rng.uniform(50, 100, n), rng.uniform(-0.9, 0.9, n))

        e = get_encoders(n, d, rng=rng)

        # make one activity matrix with the max number of eval points
        train = get_eval_points(max_points, d, rng=rng)
        test = get_eval_points(max_points, d, rng=rng)
        Atrain = a.rates(np.dot(train, e), gain, bias)
        Atest = a.rates(np.dot(test, e), gain, bias)

        for i, n_points in enumerate(eval_points):
            Di, _ = solver(Atrain[:n_points], train[:n_points], rng=rng)
            rmses[i, trial] = rms(np.dot(Atest, Di) - test)

    rmses_norm1 = rmses - rmses.mean(0, keepdims=True)
    rmses_norm2 = (rmses - rmses.mean(0, keepdims=True)
                   ) / rmses.std(0, keepdims=True)

    with Plotter(Simulator) as plt:
        def make_plot(rmses):
            mean = rmses.mean(1)
            low = rmses.min(1)
            high = rmses.max(1)
            std = rmses.std(1)
            plt.semilogx(eval_points, mean, 'k-')
            plt.semilogx(eval_points, mean - std, 'k--')
            plt.semilogx(eval_points, mean + std, 'k--')
            plt.semilogx(eval_points, high, 'r-')
            plt.semilogx(eval_points, low, 'b-')
            plt.xlim([eval_points[0], eval_points[-1]])
            # plt.xticks(eval_points, eval_points)

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        make_plot(rmses)
        plt.subplot(3, 1, 2)
        make_plot(rmses_norm1)
        plt.subplot(3, 1, 3)
        make_plot(rmses_norm2)
        plt.savefig('test_decoders.test_eval_points_static.pdf')
        plt.close()


@pytest.mark.benchmark
def test_eval_points(Simulator, nl_nodirect):
    rng = np.random.RandomState(0)
    n = 100
    d = 5
    filter = 0.08
    dt = 1e-3

    eval_points = np.logspace(np.log10(300), np.log10(5000), 11)
    eval_points = np.round(eval_points).astype('int')
    max_points = eval_points.max()
    n_trials = 1

    rmses = np.nan * np.zeros((len(eval_points), n_trials))
    for j in range(n_trials):
        points = rng.normal(size=(max_points, d))
        points *= (rng.uniform(size=max_points)
                   / norm(points, axis=-1))[:, None]

        rng_j = np.random.RandomState(348 + j)
        seed = 903824 + j

        # generate random input in unit hypersphere
        x = rng_j.normal(size=d)
        x *= rng_j.uniform() / norm(x)

        for i, n_points in enumerate(eval_points):
            model = nengo.Network(
                'test_eval_points(%d,%d)' % (i, j), seed=seed)
            with model:
                u = nengo.Node(output=x)
                a = nengo.Ensemble(nl_nodirect(n * d), d,
                                   eval_points=points[:n_points])
                nengo.Connection(u, a, synapse=0)
                up = nengo.Probe(u)
                ap = nengo.Probe(a)

            with Timer() as timer:
                sim = Simulator(model, dt=dt)
            sim.run(10 * filter)

            t = sim.trange()
            xt = filtfilt(sim.data[up], filter / dt)
            yt = filtfilt(sim.data[ap], filter / dt)
            t0 = 5 * filter
            t1 = 7 * filter
            tmask = (t > t0) & (t < t1)

            rmses[i, j] = rms(yt[tmask] - xt[tmask])
            print("done %d (%d) in %0.3f s" % (n_points, j, timer.duration))

    # subtract out mean for each model
    rmses_norm = rmses - rmses.mean(0, keepdims=True)

    with Plotter(Simulator, nl_nodirect) as plt:
        mean = rmses_norm.mean(1)
        low = rmses_norm.min(1)
        high = rmses_norm.max(1)
        plt.semilogx(eval_points, mean, 'k-')
        plt.semilogx(eval_points, high, 'r-')
        plt.semilogx(eval_points, low, 'b-')
        plt.xlim([eval_points[0], eval_points[-1]])
        plt.xticks(eval_points, eval_points)
        plt.savefig('test_decoders.test_eval_points.pdf')
        plt.close()


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
