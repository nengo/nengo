"""
Functions concerned with solving for decoders or full weight matrices.

Many of the solvers in this file can solve for decoders or weight matrices,
depending on whether the post-population encoders `E` are provided (see below).
Solvers that are only intended to solve for either decoders or weights can
remove the `E` parameter or make it manditory as they see fit.
"""
import logging
import time

import numpy as np

import nengo.utils.least_squares_solvers as lstsq
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam, FrozenObject, NdarrayParam, NumberParam, Parameter)
from nengo.utils.compat import range, with_metaclass
from nengo.utils.least_squares_solvers import (
    format_system, rmses, LeastSquaresSolverParam)
from nengo.utils.magic import DocstringInheritor

logger = logging.getLogger(__name__)


class Solver(with_metaclass(DocstringInheritor, FrozenObject)):
    """Decoder or weight solver."""

    weights = BoolParam('weights')

    def __init__(self, weights=False):
        super(Solver, self).__init__()
        self.weights = weights

    def __call__(self, A, Y, rng=None, E=None):
        """Call the solver.

        Parameters
        ----------
        A : (n_eval_points, n_neurons) array_like
            Matrix of the neurons' activities at the evaluation points
        Y : (n_eval_points, dimensions) array_like
            Matrix of the target decoded values for each of the D dimensions,
            at each of the evaluation points.
        rng : `numpy.random.RandomState`, optional (Default: None)
            A random number generator to use as required. If None,
            the ``numpy.random`` module functions will be used.
        E : (dimensions, post.n_neurons) array_like, optional (Default: None)
            Array of post-population encoders. Providing this tells the solver
            to return an array of connection weights rather than decoders.

        Returns
        -------
        X : (n_neurons, dimensions) or (n_neurons, post.n_neurons) ndarray
            (n_neurons, dimensions) array of decoders (if ``solver.weights``
            is False) or (n_neurons, post.n_neurons) array of weights
            (if ``'solver.weights`` is True).
        info : dict
            A dictionary of information about the solver. All dictionaries have
            an ``'rmses'`` key that contains RMS errors of the solve.
            Other keys are unique to particular solvers.
        """
        raise NotImplementedError("Solvers must implement '__call__'")

    def mul_encoders(self, Y, E, copy=False):
        """Helper function that projects signal ``Y`` onto encoders ``E``.

        Parameters
        ----------
        Y : ndarray
            The signal of interest.
        E : (dimensions, n_neurons) array_like or None
            Array of encoders. If None, ``Y`` will be returned unchanged.
        copy : bool, optional (Default: False)
            Whether a copy of ``Y`` should be returned if ``E`` is None.
        """
        if self.weights and E is None:
            raise ValidationError(
                "Encoders must be provided for weight solver", attr='E')
        if not self.weights and E is not None:
            raise ValidationError(
                "Encoders must be 'None' for decoder solver", attr='E')

        return np.dot(Y, E) if E is not None else Y.copy() if copy else Y


class SolverParam(Parameter):
    def coerce(self, instance, solver):
        self.check_type(instance, solver, Solver)
        return super(SolverParam, self).coerce(instance, solver)


class Lstsq(Solver):
    """Unregularized least-squares solver.

    Parameters
    ----------
    weights : bool, optional (Default: False)
        If False, solve for decoders. If True, solve for weights.
    rcond : float, optional (Default: 0.01)
        Cut-off ratio for small singular values (see `numpy.linalg.lstsq`).

    Attributes
    ----------
    rcond : float
        Cut-off ratio for small singular values (see `numpy.linalg.lstsq`).
    weights : bool
        If False, solve for decoders. If True, solve for weights.
    """

    rcond = NumberParam('noise', low=0)

    def __init__(self, weights=False, rcond=0.01):
        super(Lstsq, self).__init__(weights=weights)
        self.rcond = rcond

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y = self.mul_encoders(Y, E)
        X, residuals2, rank, s = np.linalg.lstsq(A, Y, rcond=self.rcond)
        t = time.time() - tstart
        return X, {'rmses': rmses(A, X, Y),
                   'residuals': np.sqrt(residuals2),
                   'rank': rank,
                   'singular_values': s,
                   'time': t}


class _LstsqNoiseSolver(Solver):
    """Base class for least-squares solvers with noise."""

    noise = NumberParam('noise', low=0)
    solver = LeastSquaresSolverParam('solver')

    def __init__(self, weights=False, noise=0.1, solver=lstsq.Cholesky()):
        """
        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.
        noise : float, optional (Default: 0.1)
            Amount of noise, as a fraction of the neuron activity.
        solver : `.LeastSquaresSolver`, optional (Default: ``Cholesky()``)
            Subsolver to use for solving the least squares problem.

        Attributes
        ----------
        noise : float
            Amount of noise, as a fraction of the neuron activity.
        solver : `.LeastSquaresSolver`
            Subsolver to use for solving the least squares problem.
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        super(_LstsqNoiseSolver, self).__init__(weights=weights)
        self.noise = noise
        self.solver = solver


class LstsqNoise(_LstsqNoiseSolver):
    """Least-squares solver with additive Gaussian white noise."""

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        rng = np.random if rng is None else rng
        sigma = self.noise * A.max()
        A = A + rng.normal(scale=sigma, size=A.shape)
        X, info = self.solver(A, Y, 0, rng=rng)
        info['time'] = time.time() - tstart
        return self.mul_encoders(X, E), info


class LstsqMultNoise(_LstsqNoiseSolver):
    """Least-squares solver with multiplicative white noise."""

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        rng = np.random if rng is None else rng
        A = A + rng.normal(scale=self.noise, size=A.shape) * A
        X, info = self.solver(A, Y, 0, rng=rng)
        info['time'] = time.time() - tstart
        return self.mul_encoders(X, E), info


class _LstsqL2Solver(Solver):
    """Base class for L2-regularized least-squares solvers."""

    reg = NumberParam('reg', low=0)
    solver = LeastSquaresSolverParam('solver')

    def __init__(self, weights=False, reg=0.1, solver=lstsq.Cholesky()):
        """
        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.
        reg : float, optional (Default: 0.1)
            Amount of regularization, as a fraction of the neuron activity.
        solver : `.LeastSquaresSolver`, optional (Default: ``Cholesky()``)
            Subsolver to use for solving the least squares problem.

        Attributes
        ----------
        reg : float
            Amount of regularization, as a fraction of the neuron activity.
        solver : `.LeastSquaresSolver`
            Subsolver to use for solving the least squares problem.
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        super(_LstsqL2Solver, self).__init__(weights=weights)
        self.reg = reg
        self.solver = solver


class LstsqL2(_LstsqL2Solver):
    """Least-squares solver with L2 regularization."""

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        sigma = self.reg * A.max()
        X, info = self.solver(A, Y, sigma, rng=rng)
        info['time'] = time.time() - tstart
        return self.mul_encoders(X, E), info


class LstsqL2nz(_LstsqL2Solver):
    """Least-squares solver with L2 regularization on non-zero components."""

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        # Compute the equivalent noise standard deviation. This equals the
        # base amplitude (noise_amp times the overall max activation) times
        # the square-root of the fraction of non-zero components.
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))

        # sigma == 0 means the neuron is never active, so won't be used, but
        # we have to make sigma != 0 for numeric reasons.
        sigma[sigma == 0] = sigma.max()

        X, info = self.solver(A, Y, sigma, rng=rng)
        info['time'] = time.time() - tstart
        return self.mul_encoders(X, E), info


class LstsqL1(Solver):
    """Least-squares solver with L1 and L2 regularization (elastic net).

    This method is well suited for creating sparse decoders or weight matrices.
    """

    l1 = NumberParam('l1', low=0)
    l2 = NumberParam('l2', low=0)

    def __init__(self, weights=False, l1=1e-4, l2=1e-6):
        """
        .. note:: Requires `scikit-learn <http://scikit-learn.org/stable/>`_.

        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.
        l1 : float, optional (Default: 1e-4)
            Amount of L1 regularization.
        l2 : float, optional (Default: 1e-6)
            Amount of L2 regularization.

        Attributes
        ----------
        l1 : float
            Amount of L1 regularization.
        l2 : float
            Amount of L2 regularization.
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        import sklearn.linear_model  # noqa F401, import to check existence
        assert sklearn.linear_model
        super(LstsqL1, self).__init__(weights=weights)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, A, Y, rng=None, E=None):
        import sklearn.linear_model
        tstart = time.time()
        Y = self.mul_encoders(Y, E, copy=True)  # copy since 'fit' may modify Y

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
        t = time.time() - tstart
        infos = {'rmses': rmses(A, X, Y), 'time': t}
        return X, infos


class LstsqDrop(Solver):
    """Find sparser decoders/weights by dropping small values.

    This solver first solves for coefficients (decoders/weights) with
    L2 regularization, drops those nearest to zero, and retrains remaining.
    """

    drop = NumberParam('drop', low=0, high=1)
    solver1 = SolverParam('solver1')
    solver2 = SolverParam('solver2')

    def __init__(self, weights=False, drop=0.25,
                 solver1=LstsqL2(reg=0.001), solver2=LstsqL2(reg=0.1)):
        """
        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.
        drop : float, optional (Default: 0.25)
            Fraction of decoders or weights to set to zero.
        solver1 : Solver, optional (Default: ``LstsqL2(reg=0.001)``)
            Solver for finding the initial decoders.
        solver2 : Solver, optional (Default: ``LstsqL2(reg=0.1)``)
            Used for re-solving for the decoders after dropout.

        Attributes
        ----------
        drop : float
            Fraction of decoders or weights to set to zero.
        solver1 : Solver
            Solver for finding the initial decoders.
        solver2 : Solver
            Used for re-solving for the decoders after dropout.
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        super(LstsqDrop, self).__init__(weights=weights)
        self.drop = drop
        self.solver1 = solver1
        self.solver2 = solver2

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)

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

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y), 'info0': info0, 'info1': info1,
                'time': t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info


class Nnls(Solver):
    """Non-negative least-squares solver without regularization.

    Similar to `.Lstsq`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    def __init__(self, weights=False):
        """
        .. note:: Requires
                  `SciPy <https://docs.scipy.org/doc/scipy/reference/>`_.

        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.

        Attributes
        ----------
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        import scipy.optimize  # import here too to throw error early
        assert scipy.optimize
        super(Nnls, self).__init__(weights=weights)

    def __call__(self, A, Y, rng=None, E=None):
        import scipy.optimize

        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        Y = self.mul_encoders(Y, E, copy=True)
        d = Y.shape[1]

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for i in range(d):
            X[:, i], residuals[i] = scipy.optimize.nnls(A, Y[:, i])

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y), 'residuals': residuals, 'time': t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info


class NnlsL2(Nnls):
    """Non-negative least-squares solver with L2 regularization.

    Similar to `.LstsqL2`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    reg = NumberParam('reg', low=0)

    def __init__(self, weights=False, reg=0.1):
        """
        .. note:: Requires
                  `SciPy <https://docs.scipy.org/doc/scipy/reference/>`_.

        Parameters
        ----------
        weights : bool, optional (Default: False)
            If False, solve for decoders. If True, solve for weights.
        reg : float, optional (Default: 0.1)
            Amount of regularization, as a fraction of the neuron activity.

        Attributes
        ----------
        reg : float
            Amount of regularization, as a fraction of the neuron activity.
        weights : bool
            If False, solve for decoders. If True, solve for weights.
        """
        super(NnlsL2, self).__init__(weights=weights)
        self.reg = reg

    def _solve(self, A, Y, rng, E, sigma=0.):
        import scipy.optimize

        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        Y = self.mul_encoders(Y, E, copy=True)
        d = Y.shape[1]

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma**2)
        GY = np.dot(A.T, Y.clip(0, None))
        # ^ TODO: why is it better if we clip Y to be positive here?

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for i in range(d):
            X[:, i], residuals[i] = scipy.optimize.nnls(GA, GY[:, i])

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y), 'residuals': residuals, 'time': t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info

    def __call__(self, A, Y, rng=None, E=None):
        return self._solve(A, Y, rng, E, sigma=self.reg * A.max())


class NnlsL2nz(NnlsL2):
    """Non-negative least-squares with L2 regularization on nonzero components.

    Similar to `.LstsqL2nz`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    def __call__(self, A, Y, rng=None, E=None):
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))
        sigma[sigma == 0] = 1
        return self._solve(A, Y, rng, E, sigma=sigma)


class NoSolver(Solver):
    """Manually pass in weights, bypassing the decoder solver.

    Parameters
    ----------
    values : (n_neurons, n_weights) array_like, optional (Default: None)
        The array of decoders or weights to use.
        If ``weights`` is ``False``, ``n_weights`` is the expected
        output dimensionality. If ``weights`` is ``True``,
        ``n_weights`` is the number of neurons in the post ensemble.
        If ``None``, which is the default, the solver will return an
        appropriately sized array of zeros.
    weights : bool, optional (Default: False)
        If False, ``values`` is interpreted as decoders.
        If True, ``values`` is interpreted as weights.

    Attributes
    ----------
    values : (n_neurons, n_weights) array_like, optional (Default: None)
        The array of decoders or weights to use.
        If ``weights`` is ``False``, ``n_weights`` is the expected
        output dimensionality. If ``weights`` is ``True``,
        ``n_weights`` is the number of neurons in the post ensemble.
        If ``None``, which is the default, the solver will return an
        appropriately sized array of zeros.
    weights : bool, optional (Default: False)
        If False, ``values`` is interpreted as decoders.
        If True, ``values`` is interpreted as weights.
    """

    values = NdarrayParam("values", optional=True, shape=("*", "*"))

    def __init__(self, values=None, weights=False):
        super(NoSolver, self).__init__(weights=weights)
        self.values = values

    def __call__(self, A, Y, rng=None, E=None):
        if self.values is None:
            n_neurons = np.asarray(A).shape[1]
            if self.weights:
                return np.zeros((n_neurons, np.asarray(E).shape[1])), {}
            else:
                return np.zeros((n_neurons, np.asarray(Y).shape[1])), {}

        return self.values, {}
