"""
TODO:
  - add a test to test each solver many times on different populations,
    and record the error.
"""
import numpy as np
import pytest

import nengo
from nengo.dists import UniformHypersphere
from nengo.utils.compat import iteritems, range
from nengo.utils.numpy import rms, norm
from nengo.utils.stdlib import Timer
from nengo.utils.testing import allclose
from nengo.solvers import (
    lstsq, Lstsq, LstsqNoise, LstsqL2, LstsqL2nz,
    LstsqL1, LstsqDrop, Nnls, NnlsL2, NnlsL2nz)


class Factory(object):
    def __init__(self, klass, *args, **kwargs):
        self.klass = klass
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        args = [v() if isinstance(v, Factory) else v for v in self.args]
        kwargs = {k: v() if isinstance(v, Factory) else v
                  for k, v in iteritems(self.kwargs)}
        return self.klass(*args, **kwargs)

    def __str__(self):
        try:
            inst = self()
        except:
            inst = "%s(args=%s, kwargs=%s)" % (
                self.klass, self.args, self.kwargs)
        return str(inst)

    def __repr__(self):
        try:
            inst = self()
        except:
            inst = "<%r instance>" % (self.klass.__name__)
        return repr(inst)


def get_encoders(n_neurons, dims, rng=None):
    return UniformHypersphere(surface=True).sample(n_neurons, dims, rng=rng).T


def get_eval_points(n_points, dims, rng=None, sort=False):
    points = UniformHypersphere(surface=False).sample(n_points, dims, rng=rng)
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


def test_cholesky(rng):
    m, n = 100, 100
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m, ))

    x0, _, _, _ = np.linalg.lstsq(A, b)
    x1, _ = lstsq.Cholesky(transpose=False)(A, b, 0)
    x2, _ = lstsq.Cholesky(transpose=True)(A, b, 0)
    assert np.allclose(x0, x1)
    assert np.allclose(x0, x2)


def test_conjgrad(rng):
    A, b = get_system(1000, 100, 2, rng=rng)
    sigma = 0.1 * A.max()

    x0, _ = lstsq.Cholesky()(A, b, sigma)
    x1, _ = lstsq.Conjgrad(tol=1e-3)(A, b, sigma)
    x2, _ = lstsq.BlockConjgrad(tol=1e-3)(A, b, sigma)
    assert np.allclose(x0, x1, atol=1e-6, rtol=1e-3)
    assert np.allclose(x0, x2, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize('Solver', [
    Lstsq, LstsqNoise, LstsqL2, LstsqL2nz, LstsqDrop])
def test_decoder_solver(Solver, plt, rng):
    solver = Solver()

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

    plt.plot(test, np.zeros_like(test), 'k--')
    plt.plot(test, test - est)
    plt.title("relative RMSE: %0.2e" % rel_rmse)

    atol = 3.5e-2 if isinstance(solver, (LstsqNoise, LstsqDrop)) else 1.5e-2
    assert np.allclose(test, est, atol=atol, rtol=1e-3)
    assert rel_rmse < 0.02


@pytest.mark.parametrize('Solver', [
    LstsqNoise, LstsqL2, LstsqL2nz])
def test_subsolvers(Solver, seed, rng, tol=1e-2):
    get_rng = lambda: np.random.RandomState(seed)

    A, b = get_system(500, 100, 5, rng=rng)
    x0, _ = Solver(solver=lstsq.Cholesky())(A, b, rng=get_rng())

    subsolvers = [lstsq.Conjgrad, lstsq.BlockConjgrad]
    for subsolver in subsolvers:
        x, info = Solver(solver=subsolver(tol=tol))(A, b, rng=get_rng())
        rel_rmse = rms(x - x0) / rms(x0)
        assert rel_rmse < 4 * tol
        # the above 4 * tol is just a heuristic; the main purpose of this
        # test is to make sure that the subsolvers don't throw errors
        # in-situ. They are tested more robustly elsewhere.


@pytest.mark.parametrize('Solver', [
    Factory(LstsqL2, solver=Factory(lstsq.RandomizedSVD)),
    LstsqL1])
def test_decoder_solver_extra(Solver, plt, rng):
    pytest.importorskip('sklearn')
    test_decoder_solver(Solver, plt, rng)


@pytest.mark.parametrize('Solver', [Lstsq, LstsqL2, LstsqL2nz])
def test_weight_solver(Solver, rng):
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
    D, _ = Solver()(Atrain, Xtrain, rng=rng)
    W1 = np.dot(D, Eb)

    # find weights directly
    W2, _ = Solver(weights=True)(Atrain, Xtrain, rng=rng, E=Eb)

    # assert that post inputs are close on test points
    test = get_eval_points(n_points, dims, rng=rng)  # testing eval points
    Atest = rates(np.dot(test, Ea))
    Y1 = np.dot(Atest, W1)                         # post inputs from decoders
    Y2 = np.dot(Atest, W2)                         # post inputs from weights
    assert np.allclose(Y1, Y2)

    # assert that weights themselves are close (this is true for L2 weights)
    assert np.allclose(W1, W2)


def test_scipy_solvers(rng):
    pytest.importorskip('scipy', minversion='0.11')  # version for lsmr

    A, b = get_system(1000, 100, 2, rng=rng)
    sigma = 0.1 * A.max()

    x0, _ = lstsq.Cholesky()(A, b, sigma)
    x1, _ = lstsq.ConjgradScipy()(A, b, sigma)
    x2, _ = lstsq.LSMRScipy()(A, b, sigma)
    assert np.allclose(x0, x1, atol=2e-5, rtol=1e-3)
    assert np.allclose(x0, x2, atol=2e-5, rtol=1e-3)


@pytest.mark.parametrize('Solver', [Nnls, NnlsL2, NnlsL2nz])
def test_nnls(Solver, plt, rng):
    pytest.importorskip('scipy')

    A, x = get_system(500, 100, 1, rng=rng, sort=True)
    y = x**2

    d, _ = Solver()(A, y, rng)
    yest = np.dot(A, d)
    rel_rmse = rms(yest - y) / rms(y)

    plt.subplot(211)
    plt.plot(x, y, 'k--')
    plt.plot(x, yest)
    plt.ylim([-0.1, 1.1])
    plt.subplot(212)
    plt.plot(x, np.zeros_like(x), 'k--')
    plt.plot(x, yest - y)

    assert np.allclose(yest, y, atol=3e-2, rtol=1e-3)
    assert rel_rmse < 0.02


@pytest.mark.slow
def test_subsolvers_L2(rng, logger):
    pytest.importorskip('scipy', minversion='0.11')  # version for lsmr

    ref_solver = lstsq.Cholesky()
    solvers = [lstsq.Conjgrad(), lstsq.BlockConjgrad(),
               lstsq.ConjgradScipy(), lstsq.LSMRScipy()]

    A, B = get_system(m=2000, n=1000, d=10, rng=rng)
    sigma = 0.1 * A.max()

    with Timer() as t0:
        x0, _ = ref_solver(A, B, sigma)

    xs = np.zeros((len(solvers),) + x0.shape)
    for i, solver in enumerate(solvers):
        with Timer() as t:
            xs[i], info = solver(A, B, sigma)
        logger.info('solver: %r' % solver)
        logger.info('duration: %0.3f', t.duration)
        logger.info('duration relative to reference solver: %0.2f',
                    (t.duration / t0.duration))
        logger.info('info: %s', info)

    for solver, x in zip(solvers, xs):
        assert np.allclose(x0, x, atol=1e-5, rtol=1e-3), (
            "Solver %s" % solver.__name__)


@pytest.mark.noassertions
def test_subsolvers_L1(rng, logger):
    pytest.importorskip('sklearn')

    A, B = get_system(m=2000, n=1000, d=10, rng=rng)

    l1 = 1e-4
    with Timer() as t:
        LstsqL1(l1=l1, l2=0)(A, B, rng=rng)
    logger.info('duration: %0.3f', t.duration)


def test_compare_solvers(Simulator, plt, seed):
    pytest.importorskip('sklearn')

    N = 70
    decoder_solvers = [
        Lstsq(), LstsqNoise(), LstsqL2(), LstsqL2nz(), LstsqL1()]
    weight_solvers = [LstsqL1(weights=True), LstsqDrop(weights=True)]

    tfinal = 4

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=input_function)
        a = nengo.Ensemble(N, dimensions=1)
        nengo.Connection(u, a)
        ap = nengo.Probe(a)

        probes = []
        names = []
        for solver in decoder_solvers + weight_solvers:
            b = nengo.Ensemble(N, dimensions=1, seed=seed + 1)
            nengo.Connection(a, b, solver=solver)
            probes.append(nengo.Probe(b))
            names.append("%s(%s)" % (
                type(solver).__name__, 'w' if solver.weights else 'd'))

    with Simulator(model) as sim:
        sim.run(tfinal)
    t = sim.trange()

    # ref = sim.data[up]
    ref = nengo.Lowpass(0.02).filtfilt(sim.data[ap], dt=sim.dt)
    outputs = np.array([sim.data[probe][:, 0] for probe in probes]).T
    outputs_f = nengo.Lowpass(0.02).filtfilt(outputs, dt=sim.dt)

    close = allclose(t, ref, outputs_f,
                     atol=0.07, rtol=0, buf=0.1, delay=0.007,
                     plt=plt, labels=names, individual_results=True)

    for name, c in zip(names, close):
        assert c, "Solver '%s' does not meet tolerances" % name


@pytest.mark.slow
@pytest.mark.noassertions
def test_regularization(Simulator, nl_nodirect, plt):

    # TODO: multiple trials per parameter set, with different seeds

    Solvers = [LstsqL2, LstsqL2nz]
    neurons = np.array([10, 20, 50, 100])
    regs = np.linspace(0.01, 0.3, 16)
    filters = np.linspace(0, 0.03, 11)

    buf = 0.2  # buffer for initial transients
    tfinal = 3 + buf

    def input_function(t):
        return np.interp(t, [1, 3], [-1, 1], left=-1, right=1)

    model = nengo.Network('test_regularization')
    with model:
        model.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(output=input_function)
        up = nengo.Probe(u)

        probes = np.zeros(
            (len(Solvers), len(neurons), len(regs), len(filters)),
            dtype='object')

        for j, n_neurons in enumerate(neurons):
            a = nengo.Ensemble(n_neurons, dimensions=1)
            nengo.Connection(u, a)

            for i, Solver in enumerate(Solvers):
                for k, reg in enumerate(regs):
                    for l, synapse in enumerate(filters):
                        probes[i, j, k, l] = nengo.Probe(
                            a, solver=Solver(reg=reg), synapse=synapse)

    with Simulator(model) as sim:
        sim.run(tfinal)
    t = sim.trange()

    ref = sim.data[up]
    rmse_buf = lambda a, b: rms(a[t > buf] - b[t > buf])
    rmses = np.zeros(probes.shape)
    for i, probe in enumerate(probes.flat):
        rmses.flat[i] = rmse_buf(sim.data[probe], ref)
    rmses = rmses - rmses[:, :, [0], :]

    plt.figure(figsize=(8, 12))
    X, Y = np.meshgrid(filters, regs)

    for i, Solver in enumerate(Solvers):
        for j, n_neurons in enumerate(neurons):
            plt.subplot(len(neurons), len(Solvers), len(Solvers)*j + i + 1)
            Z = rmses[i, j, :, :]
            plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 21))
            plt.xlabel('filter')
            plt.ylabel('reg')
            plt.title("%s (N=%d)" % (Solver.__name__, n_neurons))

    plt.tight_layout()


@pytest.mark.slow
@pytest.mark.noassertions
def test_eval_points_static(plt, rng):
    solver = LstsqL2()

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

    def make_plot(rmses):
        mean = rmses.mean(1)
        low = rmses.min(1)
        high = rmses.max(1)
        std = rmses.std(1)
        plt.semilogx(eval_points, mean, 'k-', label='mean')
        plt.semilogx(eval_points, mean - std, 'k--', label='+/- std')
        plt.semilogx(eval_points, mean + std, 'k--')
        plt.semilogx(eval_points, high, 'r-', label='high')
        plt.semilogx(eval_points, low, 'b-', label='low')
        plt.xlim([eval_points[0], eval_points[-1]])
        # plt.xticks(eval_points, eval_points)
        plt.legend(fontsize=8, loc=1)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    make_plot(rmses)
    plt.ylabel('rmse')
    plt.subplot(3, 1, 2)
    make_plot(rmses_norm1)
    plt.ylabel('rmse - mean')
    plt.subplot(3, 1, 3)
    make_plot(rmses_norm2)
    plt.ylabel('(rmse - mean) / std')


@pytest.mark.slow
@pytest.mark.noassertions
def test_eval_points(Simulator, nl_nodirect, plt, seed, rng, logger):
    n = 100
    d = 5
    filter = 0.08

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
            model = nengo.Network(seed=seed)
            with model:
                model.config[nengo.Ensemble].neuron_type = nl_nodirect()
                u = nengo.Node(output=x)
                a = nengo.Ensemble(n * d, dimensions=d,
                                   eval_points=points[:n_points])
                nengo.Connection(u, a, synapse=0)
                up = nengo.Probe(u)
                ap = nengo.Probe(a)

            with Timer() as timer:
                sim = Simulator(model)
            sim.run(10 * filter)
            sim.close()

            t = sim.trange()
            xt = nengo.Lowpass(filter).filtfilt(sim.data[up], dt=sim.dt)
            yt = nengo.Lowpass(filter).filtfilt(sim.data[ap], dt=sim.dt)
            t0 = 5 * filter
            t1 = 7 * filter
            tmask = (t > t0) & (t < t1)

            rmses[i, j] = rms(yt[tmask] - xt[tmask])
            logger.info('trial %d', j)
            logger.info('  n_points: %d', n_points)
            logger.info('  duration: %0.3f s', timer.duration)

    # subtract out mean for each model
    rmses_norm = rmses - rmses.mean(0, keepdims=True)

    mean = rmses_norm.mean(1)
    low = rmses_norm.min(1)
    high = rmses_norm.max(1)
    plt.semilogx(eval_points, mean, 'k-')
    plt.semilogx(eval_points, high, 'r-')
    plt.semilogx(eval_points, low, 'b-')
    plt.xlim([eval_points[0], eval_points[-1]])
    plt.xticks(eval_points, eval_points)
