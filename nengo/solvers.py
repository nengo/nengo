"""Classes concerned with solving for decoders or full weight matrices.

.. inheritance-diagram:: nengo.solvers
   :parts: 1
   :top-classes: nengo.solvers.Solver, nengo.solvers.SolverParam

"""

import time

import numpy as np

import nengo.utils.least_squares_solvers as lstsq
from nengo.params import BoolParam, FrozenObject, NdarrayParam, NumberParam, Parameter
from nengo.utils.least_squares_solvers import (
    format_system,
    rmses,
    LeastSquaresSolverParam,
)


class Solver(FrozenObject):
    """Decoder or weight solver.

    A solver can be compositional or non-compositional. Non-compositional
    solvers must operate on the whole neuron-to-neuron weight matrix, while
    compositional solvers operate in the decoded state space, which is then
    combined with transform/encoders to generate the full weight matrix.
    See the solver's ``compositional`` class attribute to determine if it is
    compositional.
    """

    compositional = True

    weights = BoolParam("weights")

    def __init__(self, weights=False):
        super().__init__()
        self.weights = weights

    def __call__(self, A, Y, rng=np.random):
        """Call the solver.

        Parameters
        ----------
        A : (n_eval_points, n_neurons) array_like
            Matrix of the neurons' activities at the evaluation points
        Y : (n_eval_points, dimensions) array_like
            Matrix of the target decoded values for each of the D dimensions,
            at each of the evaluation points.
        rng : `numpy.random.mtrand.RandomState`, optional
            A random number generator to use as required.

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


class SolverParam(Parameter):
    """A parameter in which the value is a `.Solver` instance."""

    def coerce(self, instance, solver):
        self.check_type(instance, solver, Solver)
        return super().coerce(instance, solver)


class Lstsq(Solver):
    """Unregularized least-squares solver.

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.
    rcond : float, optional
        Cut-off ratio for small singular values (see `numpy.linalg.lstsq`).

    Attributes
    ----------
    rcond : float
        Cut-off ratio for small singular values (see `numpy.linalg.lstsq`).
    weights : bool
        If False, solve for decoders. If True, solve for weights.
    """

    rcond = NumberParam("noise", low=0)

    def __init__(self, weights=False, rcond=0.01):
        super().__init__(weights=weights)
        self.rcond = rcond

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        X, residuals2, rank, s = np.linalg.lstsq(A, Y, rcond=self.rcond)
        t = time.time() - tstart
        return (
            X,
            {
                "rmses": rmses(A, X, Y),
                "residuals": np.sqrt(residuals2),
                "rank": rank,
                "singular_values": s,
                "time": t,
            },
        )


def _add_noise_param_docs(cls):
    cls.__doc__ += """

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.
    noise : float, optional
        Amount of noise, as a fraction of the neuron activity.
    solver : `.LeastSquaresSolver`, optional
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
    return cls


@_add_noise_param_docs
class LstsqNoise(Solver):
    """Least-squares solver with additive Gaussian white noise."""

    noise = NumberParam("noise", low=0)
    solver = LeastSquaresSolverParam("solver")

    def __init__(self, weights=False, noise=0.1, solver=lstsq.Cholesky()):
        super().__init__(weights=weights)
        self.noise = noise
        self.solver = solver

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        sigma = self.noise * A.max()
        A = A + rng.normal(scale=sigma, size=A.shape)
        X, info = self.solver(A, Y, 0, rng=rng)
        info["time"] = time.time() - tstart
        return X, info


@_add_noise_param_docs
class LstsqMultNoise(LstsqNoise):
    """Least-squares solver with multiplicative white noise."""

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        A = A + rng.normal(scale=self.noise, size=A.shape) * A
        X, info = self.solver(A, Y, 0, rng=rng)
        info["time"] = time.time() - tstart
        return X, info


def _add_l2_param_docs(cls):
    cls.__doc__ += """

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.
    reg : float, optional
        Amount of regularization, as a fraction of the neuron activity.
    solver : `.LeastSquaresSolver`, optional
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
    return cls


@_add_l2_param_docs
class LstsqL2(Solver):
    """Least-squares solver with L2 regularization."""

    reg = NumberParam("reg", low=0)
    solver = LeastSquaresSolverParam("solver")

    def __init__(self, weights=False, reg=0.1, solver=lstsq.Cholesky()):
        super().__init__(weights=weights)
        self.reg = reg
        self.solver = solver

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        sigma = self.reg * A.max()
        X, info = self.solver(A, Y, sigma, rng=rng)
        info["time"] = time.time() - tstart
        return X, info


@_add_l2_param_docs
class LstsqL2nz(LstsqL2):
    """Least-squares solver with L2 regularization on non-zero components."""

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        # Compute the equivalent noise standard deviation. This equals the
        # base amplitude (noise_amp times the overall max activation) times
        # the square-root of the fraction of non-zero components.
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))

        # sigma == 0 means the neuron is never active, so won't be used, but
        # we have to make sigma != 0 for numeric reasons.
        sigma[sigma == 0] = sigma.max()

        X, info = self.solver(A, Y, sigma, rng=rng)
        info["time"] = time.time() - tstart
        return X, info


class LstsqL1(Solver):
    """Least-squares solver with L1 and L2 regularization (elastic net).

    This method is well suited for creating sparse decoders or weight matrices.

    .. note:: Requires `scikit-learn <https://scikit-learn.org/stable/>`_.

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.
    l1 : float, optional
        Amount of L1 regularization.
    l2 : float, optional
        Amount of L2 regularization.
    max_iter : int, optional
        Maximum number of iterations for the underlying elastic net.

    Attributes
    ----------
    l1 : float
        Amount of L1 regularization.
    l2 : float
        Amount of L2 regularization.
    weights : bool
        If False, solve for decoders. If True, solve for weights.
    max_iter : int
        Maximum number of iterations for the underlying elastic net.
    """

    compositional = False

    l1 = NumberParam("l1", low=0)
    l2 = NumberParam("l2", low=0)

    def __init__(self, weights=False, l1=1e-4, l2=1e-6, max_iter=1000):
        # import to check existence
        import sklearn.linear_model  # pylint: disable=import-outside-toplevel

        assert sklearn.linear_model
        super().__init__(weights=weights)
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter

    def __call__(self, A, Y, rng=np.random):
        import sklearn.linear_model  # pylint: disable=import-outside-toplevel

        tstart = time.time()
        Y = np.array(Y)  # copy since 'fit' may modify Y

        # TODO: play around with regularization constants (I just guessed).
        #   Do we need to scale regularization by number of neurons, to get
        #   same level of sparsity? esp. with weights? Currently, setting
        #   l1=1e-3 works well with weights when connecting 1D populations
        #   with 100 neurons each.
        a = self.l1 * A.max()  # L1 regularization
        b = self.l2 * A.max() ** 2  # L2 regularization
        alpha = a + b
        l1_ratio = a / (a + b)

        # --- solve least-squares A * X = Y
        model = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=self.max_iter
        )
        model.fit(A, Y)
        X = model.coef_.T
        X.shape = (A.shape[1], Y.shape[1]) if Y.ndim > 1 else (A.shape[1],)
        t = time.time() - tstart
        infos = {"rmses": rmses(A, X, Y), "time": t}
        return X, infos


class LstsqDrop(Solver):
    """Find sparser decoders/weights by dropping small values.

    This solver first solves for coefficients (decoders/weights) with
    L2 regularization, drops those nearest to zero, and retrains remaining.

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.
    drop : float, optional
        Fraction of decoders or weights to set to zero.
    solver1 : Solver, optional
        Solver for finding the initial decoders.
    solver2 : Solver, optional
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

    compositional = False

    drop = NumberParam("drop", low=0, high=1)
    solver1 = SolverParam("solver1")
    solver2 = SolverParam("solver2")

    def __init__(
        self,
        weights=False,
        drop=0.25,
        solver1=LstsqL2(reg=0.001),
        solver2=LstsqL2(reg=0.1),
    ):
        super().__init__(weights=weights)
        self.drop = drop
        self.solver1 = solver1
        self.solver2 = solver2

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)

        # solve for coefficients using standard solver
        X, info0 = self.solver1(A, Y, rng=rng)

        # drop weights close to zero, based on `drop` ratio
        Xabs = np.sort(np.abs(X.flat))
        threshold = Xabs[int(np.round(self.drop * Xabs.size))]
        X[np.abs(X) < threshold] = 0

        # retrain nonzero weights
        for i in range(X.shape[1]):
            nonzero = X[:, i] != 0
            if nonzero.sum() > 0:
                X[nonzero, i], info1 = self.solver2(A[:, nonzero], Y[:, i], rng=rng)

        t = time.time() - tstart
        info = {"rmses": rmses(A, X, Y), "info0": info0, "info1": info1, "time": t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info


def _add_nnls_param_docs(l2=False):
    reg_attr = """
    reg : float
        Amount of regularization, as a fraction of the neuron activity.\
    """
    reg_param = """
    reg : float, optional
        Amount of regularization, as a fraction of the neuron activity.\
    """

    docstring = """
    .. note:: Requires
              `SciPy <https://docs.scipy.org/doc/scipy/reference/>`_.

    Parameters
    ----------
    weights : bool, optional
        If False, solve for decoders. If True, solve for weights.{reg_param}

    Attributes
    ----------{reg_attr}
    weights : bool
        If False, solve for decoders. If True, solve for weights.
    """.format(
        reg_param=reg_param if l2 else "", reg_attr=reg_attr if l2 else ""
    )

    def _actually_add_nnls_param_docs(cls):
        cls.__doc__ += docstring
        return cls

    return _actually_add_nnls_param_docs


@_add_nnls_param_docs(l2=False)
class Nnls(Solver):
    """Non-negative least-squares solver without regularization.

    Similar to `.Lstsq`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    compositional = False

    def __init__(self, weights=False):
        # import here too to throw error early
        import scipy.optimize  # pylint: disable=import-outside-toplevel

        assert scipy.optimize
        super().__init__(weights=weights)

    def __call__(self, A, Y, rng=np.random):
        import scipy.optimize  # pylint: disable=import-outside-toplevel

        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        d = Y.shape[1]

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for i in range(d):
            X[:, i], residuals[i] = scipy.optimize.nnls(A, Y[:, i])

        t = time.time() - tstart
        info = {"rmses": rmses(A, X, Y), "residuals": residuals, "time": t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info


@_add_nnls_param_docs(l2=True)
class NnlsL2(Nnls):
    """Non-negative least-squares solver with L2 regularization.

    Similar to `.LstsqL2`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    reg = NumberParam("reg", low=0)

    def __init__(self, weights=False, reg=0.1):
        super().__init__(weights=weights)
        self.reg = reg

    def _solve(self, A, Y, sigma=0.0):
        import scipy.optimize  # pylint: disable=import-outside-toplevel

        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        d = Y.shape[1]

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma ** 2)
        GY = np.dot(A.T, Y.clip(0, None))
        # ^ TODO: why is it better if we clip Y to be positive here?

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for i in range(d):
            X[:, i], residuals[i] = scipy.optimize.nnls(GA, GY[:, i])

        t = time.time() - tstart
        info = {"rmses": rmses(A, X, Y), "residuals": residuals, "time": t}
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info

    def __call__(self, A, Y, rng=np.random):
        return self._solve(A, Y, sigma=self.reg * A.max())


@_add_nnls_param_docs(l2=True)
class NnlsL2nz(NnlsL2):
    """Non-negative least-squares with L2 regularization on nonzero components.

    Similar to `.LstsqL2nz`, except the output values are non-negative.

    If solving for non-negative **weights**, it is important that the
    intercepts of the post-population are also non-negative, since neurons with
    negative intercepts will never be silent, affecting output accuracy.
    """

    def __call__(self, A, Y, rng=np.random):
        sigma = (self.reg * A.max()) * np.sqrt((A > 0).mean(axis=0))
        sigma[sigma == 0] = 1
        return self._solve(A, Y, sigma=sigma)


class NoSolver(Solver):
    """Manually pass in weights, bypassing the decoder solver.

    Parameters
    ----------
    values : (n_neurons, size_out) array_like, optional
        The array of decoders to use.
        ``size_out`` is the dimensionality of the decoded signal (determined
        by the connection function).
        If ``None``, which is the default, the solver will return an
        appropriately sized array of zeros.
    weights : bool, optional
        If False, connection will use factored weights (decoders from this
        solver, transform, and encoders).
        If True, connection will use a full weight matrix (created by
        linearly combining decoder, transform, and encoders).

    Attributes
    ----------
    values : (n_neurons, size_out) array_like, optional
        The array of decoders to use.
        ``size_out`` is the dimensionality of the decoded signal (determined
        by the connection function).
        If ``None``, which is the default, the solver will return an
        appropriately sized array of zeros.
    weights : bool, optional
        If False, connection will use factored weights (decoders from this
        solver, transform, and encoders).
        If True, connection will use a full weight matrix (created by
        linearly combining decoder, transform, and encoders).
    """

    compositional = True

    values = NdarrayParam("values", optional=True, shape=("*", "*"))

    def __init__(self, values=None, weights=False):
        super().__init__(weights=weights)
        self.values = values

    def __call__(self, A, Y, rng=None):
        if self.values is None:
            n_neurons = np.asarray(A).shape[1]

            return np.zeros((n_neurons, np.asarray(Y).shape[1])), {}

        return self.values, {}
