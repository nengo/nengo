"""Classes concerned with solving for decoders or full weight matrices.

.. inheritance-diagram:: nengo.solvers
   :parts: 1
   :top-classes: nengo.solvers.Solver, nengo.solvers.SolverParam

"""

import time

import numpy as np

from nengo import npext
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam,
    FrozenObject,
    IntParam,
    NdarrayParam,
    NumberParam,
    Parameter,
)


def format_system(A, Y):
    """Extract data from A/Y matrices."""

    assert Y.ndim > 0
    m, n = A.shape
    matrix_in = Y.ndim > 1
    d = Y.shape[1] if matrix_in else 1
    Y = Y if matrix_in else Y[:, None]
    return Y, m, n, d, matrix_in


def rmses(A, X, Y):
    """Returns the root-mean-squared error (RMSE) of the solution X."""
    return npext.rms(Y - np.dot(A, X), axis=0)


class LeastSquaresSolver(FrozenObject):
    """Linear least squares system solver.

    These solvers are to be passed as arguments to `~.Solver` objects.

    Examples
    --------

    .. testcode::

       from nengo.solvers import LstsqL2, SVD

       with nengo.Network():
           ens_a = nengo.Ensemble(10, 1)
           ens_b = nengo.Ensemble(10, 1)
           nengo.Connection(ens_a, ens_b, solver=LstsqL2(solver=SVD()))
    """

    def __call__(self, A, Y, sigma, rng=None):
        raise NotImplementedError("LeastSquaresSolver must implement call")


class Cholesky(LeastSquaresSolver):
    """Solve a least-squares system using the Cholesky decomposition."""

    transpose = BoolParam("transpose", optional=True)

    def __init__(self, transpose=None):
        super().__init__()
        self.transpose = transpose

    def __call__(self, A, Y, sigma, rng=None):
        m, n = A.shape
        transpose = self.transpose
        if transpose is None:
            # transpose if matrix is fat, but not if sigmas for each neuron
            transpose = m < n and sigma.size == 1

        if transpose:
            # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
            G = np.dot(A, A.T)
            b = Y
        else:
            # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
            G = np.dot(A.T, A)
            b = np.dot(A.T, Y)

        # add L2 regularization term 'lambda' = m * sigma**2
        np.fill_diagonal(G, G.diagonal() + m * sigma ** 2)

        try:
            import scipy.linalg  # pylint: disable=import-outside-toplevel

            factor = scipy.linalg.cho_factor(G, overwrite_a=True)
            X = scipy.linalg.cho_solve(factor, b)
        except ImportError:
            L = np.linalg.cholesky(G)
            L = np.linalg.inv(L.T)
            X = np.dot(L, np.dot(L.T, b))

        X = np.dot(A.T, X) if transpose else X
        info = {"rmses": rmses(A, X, Y)}
        return X, info


class ConjgradScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's conjugate gradient.

    Parameters
    ----------
    tol : float
        Relative tolerance of the CG solver (see [1]_ for details).
    atol : float
        Absolute tolerance of the CG solver (see [1]_ for details).

    References
    ----------
    .. [1] scipy.sparse.linalg.cg documentation,
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
    """

    tol = NumberParam("tol", low=0)
    atol = NumberParam("atol", low=0)

    def __init__(self, tol=1e-4, atol=1e-8):
        super().__init__()
        self.tol = tol
        self.atol = atol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = m * sigma ** 2
        calcAA = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        G = scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=calcAA, matmat=calcAA, dtype=A.dtype
        )
        B = np.dot(A.T, Y)

        X = np.zeros((n, d), dtype=B.dtype)
        infos = np.zeros(d, dtype="int")
        itns = np.zeros(d, dtype="int")
        for i in range(d):
            # use the callback to count the number of iterations
            def callback(x, i=i):
                itns[i] += 1

            try:
                X[:, i], infos[i] = scipy.sparse.linalg.cg(
                    G, B[:, i], tol=self.tol, callback=callback, atol=self.atol
                )
            except TypeError as e:  # pragma: no cover
                # no atol parameter in Scipy < 1.1.0
                if "atol" not in str(e):
                    raise e
                X[:, i], infos[i] = scipy.sparse.linalg.cg(
                    G, B[:, i], tol=self.tol, callback=callback
                )

        info = {"rmses": rmses(A, X, Y), "iterations": itns, "info": infos}
        return X if matrix_in else X.ravel(), info


class LSMRScipy(LeastSquaresSolver):
    """Solve a least-squares system using Scipy's LSMR."""

    tol = NumberParam("tol", low=0)

    def __init__(self, tol=1e-4):
        super().__init__()
        self.tol = tol

    def __call__(self, A, Y, sigma, rng=None):
        import scipy.sparse.linalg  # pylint: disable=import-outside-toplevel

        Y, m, n, d, matrix_in = format_system(A, Y)

        damp = sigma * np.sqrt(m)
        X = np.zeros((n, d), dtype=Y.dtype)
        itns = np.zeros(d, dtype="int")
        for i in range(d):
            X[:, i], _, itns[i], _, _, _, _, _ = scipy.sparse.linalg.lsmr(
                A, Y[:, i], damp=damp, atol=self.tol, btol=self.tol
            )

        info = {"rmses": rmses(A, X, Y), "iterations": itns}
        return X if matrix_in else X.ravel(), info


class Conjgrad(LeastSquaresSolver):
    """Solve a least-squares system using conjugate gradient."""

    tol = NumberParam("tol", low=0)
    maxiters = IntParam("maxiters", low=1, optional=True)
    X0 = NdarrayParam("X0", shape=("*", "*"), optional=True)

    def __init__(self, tol=1e-2, maxiters=None, X0=None):
        super().__init__()
        self.tol = tol
        self.maxiters = maxiters
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)
        X = np.zeros((n, d)) if self.X0 is None else np.array(self.X0)
        if X.shape != (n, d):
            raise ValidationError(
                "Must be shape %s, got %s" % ((n, d), X.shape), attr="X0", obj=self
            )

        damp = m * sigma ** 2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        iters = -np.ones(d, dtype="int")
        for i in range(d):
            X[:, i], iters[i] = self._conjgrad_iters(
                G, B[:, i], X[:, i], maxiters=self.maxiters, rtol=rtol
            )

        info = {"rmses": rmses(A, X, Y), "iterations": iters}
        return X if matrix_in else X.ravel(), info

    @staticmethod
    def _conjgrad_iters(calcAx, b, x, maxiters=None, rtol=1e-6):
        """Solve a single-RHS linear system using conjugate gradient."""

        if maxiters is None:
            maxiters = b.shape[0]

        r = b - calcAx(x)
        p = r.copy()
        rsold = np.dot(r, r)

        for i in range(maxiters):
            Ap = calcAx(p)
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = np.dot(r, r)
            beta = rsnew / rsold

            if np.sqrt(rsnew) < rtol:
                break

            if beta < 1e-12:  # no perceptible change in p
                break

            # p = r + beta*p
            p *= beta
            p += r
            rsold = rsnew

        return x, i + 1


class BlockConjgrad(LeastSquaresSolver):
    """Solve a multiple-RHS least-squares system using block conj. gradient."""

    tol = NumberParam("tol", low=0)
    X0 = NdarrayParam("X0", shape=("*", "*"), optional=True)

    def __init__(self, tol=1e-2, X0=None):
        super().__init__()
        self.tol = tol
        self.X0 = X0

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, n, d, matrix_in = format_system(A, Y)
        sigma = np.asarray(sigma, dtype="float")
        sigma = sigma.reshape(sigma.size, 1)

        X = np.zeros((n, d)) if self.X0 is None else np.array(self.X0)
        if X.shape != (n, d):
            raise ValidationError(
                "Must be shape %s, got %s" % ((n, d), X.shape), attr="X0", obj=self
            )

        damp = m * sigma ** 2
        rtol = self.tol * np.sqrt(m)
        G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
        B = np.dot(A.T, Y)

        # --- conjugate gradient
        R = B - G(X)
        P = np.array(R)
        Rsold = np.dot(R.T, R)
        AP = np.zeros((n, d))

        maxiters = int(n / d)
        for i in range(maxiters):
            AP = G(P)
            alpha = np.linalg.solve(np.dot(P.T, AP), Rsold)
            X += np.dot(P, alpha)
            R -= np.dot(AP, alpha)

            Rsnew = np.dot(R.T, R)
            if (np.diag(Rsnew) < rtol ** 2).all():
                break

            beta = np.linalg.solve(Rsold, Rsnew)
            P = R + np.dot(P, beta)
            Rsold = Rsnew

        info = {"rmses": rmses(A, X, Y), "iterations": i + 1}
        return X if matrix_in else X.ravel(), info


class SVD(LeastSquaresSolver):
    """Solve a least-squares system using full SVD."""

    def __call__(self, A, Y, sigma, rng=None):
        Y, m, _, _, matrix_in = format_system(A, Y)
        U, s, V = np.linalg.svd(A, full_matrices=0)
        si = s / (s ** 2 + m * sigma ** 2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {"rmses": npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.ravel(), info


class RandomizedSVD(LeastSquaresSolver):
    """Solve a least-squares system using a randomized (partial) SVD.

    Useful for solving large matrices quickly, but non-optimally.

    Parameters
    ----------
    n_components : int, optional
        The number of SVD components to compute. A small survey of activity
        matrices suggests that the first 60 components capture almost all
        the variance.
    n_oversamples : int, optional
        The number of additional samples on the range of A.
    n_iter : int, optional
        The number of power iterations to perform (can help with noisy data).

    See also
    --------
    sklearn.utils.extmath.randomized_svd : Function used by this class
    """

    n_components = IntParam("n_components", low=1)
    n_oversamples = IntParam("n_oversamples", low=0)
    n_iter = IntParam("n_iter", low=0)

    def __init__(self, n_components=60, n_oversamples=10, n_iter=0):
        from sklearn.utils.extmath import (  # pylint: disable=import-outside-toplevel
            randomized_svd,
        )

        assert randomized_svd
        super().__init__()
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter

    def __call__(self, A, Y, sigma, rng=np.random):
        from sklearn.utils.extmath import (  # pylint: disable=import-outside-toplevel
            randomized_svd,
        )

        Y, m, n, _, matrix_in = format_system(A, Y)
        if min(m, n) <= self.n_components + self.n_oversamples:
            # more efficient to do a full SVD
            return SVD()(A, Y, sigma, rng=rng)

        U, s, V = randomized_svd(
            A,
            self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=rng,
        )
        si = s / (s ** 2 + m * sigma ** 2)
        X = np.dot(V.T, si[:, None] * np.dot(U.T, Y))
        info = {"rmses": npext.rms(Y - np.dot(A, X), axis=0)}
        return X if matrix_in else X.ravel(), info


class LeastSquaresSolverParam(Parameter):
    def coerce(self, instance, solver):
        self.check_type(instance, solver, LeastSquaresSolver)
        return super().coerce(instance, solver)


class Solver(FrozenObject):
    """Decoder or weight solver.

    A solver can have the ``weights`` parameter equal to ``True`` or ``False``.

    Weight solvers are used to form neuron-to-neuron weight matrices.
    They can be compositional or non-compositional. Non-compositional
    solvers must operate on the whole neuron-to-neuron weight matrix
    (i.e., each target is a separate postsynaptic current, without the bias
    term), while compositional solvers operate in the decoded state-space
    (i.e., each target is a dimension in state-space). Compositional solvers
    then combine the returned ``X`` with the transform and/or encoders to
    generate the full weight matrix.

    For a solver to be compositional, the following property must be true::

        X = solver(A, Y)  if and only if  L(X) = solver(A, L(Y))

    where ``L`` is some arbitrary linear operator (i.e., the transform and/or
    encoders for the postsynaptic population). This property can then be
    leveraged by the backend for efficiency. See the solver's
    ``compositional`` class attribute to determine if it is compositional.

    Non-weight solvers always operate in the decoded state-space regardless of
    whether they are compositional or non-compositional.
    """

    compositional = True

    weights = BoolParam("weights")

    def __init__(self, weights=False):
        super().__init__()
        self.weights = weights

    def __call__(self, A, Y, rng=np.random):
        """Call the solver.

        .. note:: ``n_targets`` is ``dimensions`` if ``solver.weights`` is ``False``
                  and ``post.n_neurons`` if ``solver.weights`` is ``True``.

        Parameters
        ----------
        A : (n_eval_points, n_neurons) array_like
            Matrix of the neurons' activities at the evaluation points.
        Y : (n_eval_points, n_targets) array_like
            Matrix of target values at the evaluation points.
        rng : `numpy.random.RandomState`, optional
            A random number generator to use as required.

        Returns
        -------
        X : (n_neurons, n_targets) array_like
            Matrix of weights used to map activities onto targets.
            A typical solver will approximate ``dot(A, X) ~= Y`` subject to
            some constraints on ``X``.
        info : dict
            A dictionary of information about the solver. All dictionaries have
            an ``'rmses'`` key that contains RMS errors of the solve (one per
            target). Other keys are unique to particular solvers.

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

    def __init__(self, weights=False, noise=0.1, solver=Cholesky()):
        super().__init__(weights=weights)
        self.noise = noise
        self.solver = solver

    def __call__(self, A, Y, rng=np.random):
        tstart = time.time()
        sigma = self.noise * np.amax(np.abs(A))
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

    def __init__(self, weights=False, reg=0.1, solver=Cholesky()):
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
        sigma = (self.reg * np.amax(np.abs(A))) * np.sqrt((np.abs(A) > 0).mean(axis=0))

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
        info1s = []
        for i in range(X.shape[1]):
            info1 = None
            nonzero = X[:, i] != 0
            if nonzero.sum() > 0:
                X[nonzero, i], info1 = self.solver2(A[:, nonzero], Y[:, i], rng=rng)
            info1s.append(info1)

        t = time.time() - tstart
        info = {"rmses": rmses(A, X, Y), "info0": info0, "info1s": info1s, "time": t}
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
