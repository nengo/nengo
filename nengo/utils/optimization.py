from __future__ import absolute_import

import collections

import numpy as np
import scipy.optimize

import nengo
from nengo.params import Deferral


class ArgumentStoreSolver(nengo.solvers.Solver):
    solver = nengo.solvers.SolverParam('solver')

    def __init__(self, solver):
        super(ArgumentStoreSolver, self).__init__()
        self.solver = solver
        self.A = None
        self.Y = None
        self.X = None
        self.info = None
        self.weights = False

    def __call__(self, A, Y, rng=None, E=None):
        assert E is None  # FIXME handle not None case, can we handle it?

        self.A = A
        self.Y = Y
        self.X, self.info = self.solver(A, Y, rng=rng)
        return self.X, self.info

    def __hash__(self):
        items = list(self.__dict__.items())
        items.sort(key=lambda item: item[0])

        hashes = []
        for k, v in items:
            if k in ['A', 'Y', 'X']:
                continue
            if isinstance(v, np.ndarray):
                if v.size < 1e5:
                    a = v[:]
                    a.setflags(write=False)
                    hashes.append(hash(a))
                else:
                    raise ValidationError("array is too large to hash", attr=k)
            elif isinstance(v, collections.Iterable):
                hashes.append(hash(tuple(v)))
            elif isinstance(v, collections.Callable):
                hashes.append(hash(v.__code__))
            else:
                hashes.append(hash(v))

        return hash(tuple(hashes))


class MSEPredictorCoefficients(object):
    def __init__(self, Simulator, solver, noise, ens, rng=None):
        self.Simulator = Simulator
        self.solver = solver
        self.noise = noise
        self.ens = ens
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def approximate_noise_mse(self, n_neurons, fn):
        # FIXME seed
        solver = ArgumentStoreSolver(self.solver)
        with nengo.Network(seed=None, add_to_container=False) as m:
            ens_copy = self.ens.copy()
            ens_copy.n_neurons = n_neurons
            conn = nengo.Connection(
                ens_copy,
                nengo.Ensemble(
                    1, self.ens.dimensions, neuron_type=nengo.Direct()),
                solver=solver, function=fn)
        with self.Simulator(m):
            pass
        noise_A = solver.A + self.rng.randn(*solver.A.shape) * (
            self.noise * np.max(solver.A))

        x_hat = np.dot(noise_A, solver.X)
        return np.mean(np.square(np.squeeze(solver.Y) - np.squeeze(x_hat)))


    def obtain_mse_predictor_coefficients(
            self, basis_size, min_neurons, n_trials=20):
        n_neurons = 2 ** np.arange(min_neurons, 11, 1)
        w = np.empty((basis_size, 2))
        for p in range(basis_size):
            fn = lambda x: np.polynomial.legendre.legval(
                x, np.eye(basis_size)[p])

            mses = [
                np.mean([self.approximate_noise_mse(n, fn) for i in range(n_trials)]) for n in n_neurons]
            # FIXME should the arange(len(n_neurons)) be the actual exponents
            # used in n_neurons?
            # Seems to be corrected for in another place ... but doing it
            # correctly here would make things easier
            w[p] = np.polyfit(np.arange(len(n_neurons)), np.log(mses), 1)

        opt_slope = np.polyfit(np.arange(len(w)), w[:, 0], 1)

        intercept_fn = lambda p, x: p[0] * p[2] ** (-x) - p[1]
        error_fn = lambda p, x, y: intercept_fn(p, x) - y

        p0 = [-6., -1., 2.]
        opt_inter, _ = scipy.optimize.leastsq(error_fn, p0, args=(
            np.arange(len(w)), w[:, 1]))

        first_p = w[0, :]
        return opt_slope, opt_inter, first_p


class NNeuronsForAccuracy(Deferral):
    def __init__(
            self, fn, mse, solver=nengo.solvers.LstsqL2(), noise=.1,
            basis_size=7, min_neurons=6, rng=None):
        super(NNeuronsForAccuracy, self).__init__()
        self.fn = fn
        self.mse = mse
        self.solver = solver
        self.noise = noise
        self.basis_size = basis_size
        self.min_neurons = min_neurons
        self.rng = rng

    def default_fn(self, model, ens):
        opt_slope, opt_inter, first_p = MSEPredictorCoefficients(
            model.simulator.__class__, self.solver, self.noise, ens,
            self.rng).obtain_mse_predictor_coefficients(
                self.basis_size, self.min_neurons)

        n_eval_points = ens.n_eval_points
        if n_eval_points is None:
            n_eval_points = 100  # FIXME
        x = ens.eval_points.sample(n_eval_points, ens.dimensions)  # FIXME no sample?
        x = np.squeeze(x)
        y = self.fn(x)

        basis_max = 10
        for i in range(basis_max):
            C = np.polynomial.legendre.legfit(x, y, i)
            basis_mse = np.mean(np.square(
                np.polynomial.legendre.legval(x, C) - y))
            if  basis_mse < self.mse / 10.:
                break

        def predict_mse(p, n):
            f1 = opt_slope[0] * p + opt_slope[1]
            f2 = opt_inter[0] * opt_inter[2] ** (-p) - opt_inter[1]
            f3 = np.log2(n) - self.min_neurons
            return np.squeeze(np.exp(np.outer(f1, f3).T + f2))

        def fitfunc(n):
            mse_total = C[0] * np.exp(
                first_p[0] * (np.log2(n) - self.min_neurons) + first_p[1])
            for i in range(1, len(C)):
                mse_total += C[i] * predict_mse(i, n)
            return mse_total

        p0 = 100
        result = int(scipy.optimize.fmin(
            lambda p: np.square(fitfunc(p) - self.mse), p0))
        print result
        return result
