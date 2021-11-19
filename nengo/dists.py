import warnings

import numpy as np

import nengo.utils.numpy as npext
from nengo.exceptions import ConvergenceError, ValidationError
from nengo.params import (
    BoolParam,
    EnumParam,
    FrozenObject,
    IntParam,
    NdarrayParam,
    NumberParam,
    Parameter,
    Unconfigurable,
)
from nengo.utils.numpy import is_integer
from nengo.utils.paths import data_dir


class Distribution(FrozenObject):
    """A base class for probability distributions.

    The only thing that a probabilities distribution need to define is a
    `.Distribution.sample` method. This base class ensures that all
    distributions accept the same arguments for the sample function.
    """

    def _sample_shape(self, n, d=None):
        """Returns output shape for sample method."""
        return (n,) if d is None else (n, d)

    def sample(self, n, d=None, rng=np.random):
        """Samples the distribution.

        Parameters
        ----------
        n : int
            Number samples to take.
        d : int or None, optional
            The number of dimensions to return. If this is an int, the return
            value will be of shape ``(n, d)``. If None, the return
            value will be of shape ``(n,)``.
        rng : `numpy.random.RandomState`, optional
            Random number generator state.

        Returns
        -------
        samples : (n,) or (n, d) array_like
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """
        raise NotImplementedError("Distributions should implement sample.")


class DistributionParam(Parameter):
    """A Distribution."""

    equatable = True

    def coerce(self, instance, dist):  # pylint: disable=arguments-renamed
        self.check_type(instance, dist, Distribution)
        return super().coerce(instance, dist)


class DistOrArrayParam(NdarrayParam):
    """Can be a Distribution or samples from a distribution."""

    def __init__(
        self,
        name,
        default=Unconfigurable,
        sample_shape=None,
        sample_dtype=np.float64,
        optional=False,
        readonly=None,
    ):
        super().__init__(
            name=name,
            default=default,
            shape=sample_shape,
            dtype=sample_dtype,
            optional=optional,
            readonly=readonly,
        )

    def coerce(self, instance, distorarray):  # pylint: disable=arguments-renamed
        if isinstance(distorarray, Distribution):
            return Parameter.coerce(self, instance, distorarray)
        return super().coerce(instance, distorarray)


def get_samples(dist_or_samples, n, d=None, rng=np.random):
    """Convenience function to sample a distribution or return samples.

    Use this function in situations where you accept an argument that could
    be a distribution, or could be an ``array_like`` of samples.

    Examples
    --------

    .. testcode::

       from nengo.dists import get_samples

       rng = np.random.RandomState(seed=0)

       def mean(values, n=100):
           samples = get_samples(values, n=n, rng=rng)
           print(f"{np.mean(samples):.4f}")

       mean([1, 2, 3, 4])
       mean(nengo.dists.Gaussian(0, 1))

    .. testoutput::

       2.5000
       0.0598

    Parameters
    ----------
    dist_or_samples : `.Distribution` or (n, d) array_like
        Source of the samples to be returned.
    n : int
        Number samples to take.
    d : int or None, optional
        The number of dimensions to return.
    rng : RandomState, optional
        Random number generator.

    Returns
    -------
    samples : (n, d) array_like

    """
    if isinstance(dist_or_samples, Distribution):
        return dist_or_samples.sample(n, d=d, rng=rng)
    return np.array(dist_or_samples)


class PDF(Distribution):
    """An arbitrary distribution from a PDF.

    Parameters
    ----------
    x : vector_like (n,)
        Values of the points to sample from (interpolated).
    p : vector_like (n,)
        Probabilities of the ``x`` points.
    """

    x = NdarrayParam("x", shape="*")
    p = NdarrayParam("p", shape="*")

    def __init__(self, x, p):
        super().__init__()

        psum = np.sum(p)
        if np.abs(psum - 1) > 1e-8:
            raise ValidationError(
                f"PDF must sum to one (sums to {psum:f})", attr="p", obj=self
            )

        self.x = x
        self.p = p
        if len(self.x) != len(self.p):
            raise ValidationError(
                "`x` and `p` must be the same length", attr="p", obj=self
            )

        # make cumsum = [0] + cumsum, cdf = 0.5 * (cumsum[:-1] + cumsum[1:])
        cumsum = np.cumsum(p)
        cumsum *= 0.5
        cumsum[1:] = cumsum[:-1] + cumsum[1:]
        self.cdf = cumsum

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        return np.interp(rng.uniform(size=shape), self.cdf, self.x)


class Uniform(Distribution):
    """A uniform distribution.

    It's equally likely to get any scalar between ``low`` and ``high``.

    Note that the order of ``low`` and ``high`` doesn't matter;
    if ``low < high`` this will still work, and ``low`` will still
    be a closed interval while ``high`` is open.

    Parameters
    ----------
    low : Number
        The closed lower bound of the uniform distribution; samples >= low
    high : Number
        The open upper bound of the uniform distribution; samples < high
    integer : boolean, optional
        If true, sample from a uniform distribution of integers. In this case,
        low and high should be integers.
    """

    low = NumberParam("low")
    high = NumberParam("high")
    integer = BoolParam("integer")

    def __init__(self, low, high, integer=False):
        super().__init__()
        self.low = low
        self.high = high
        self.integer = integer

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        if self.integer:
            return rng.randint(low=self.low, high=self.high, size=shape)
        else:
            return rng.uniform(low=self.low, high=self.high, size=shape)


class Gaussian(Distribution):
    """A Gaussian distribution.

    This represents a bell-curve centred at ``mean`` and with
    spread represented by the standard deviation, ``std``.

    Parameters
    ----------
    mean : Number
        The mean of the Gaussian.
    std : Number
        The standard deviation of the Gaussian.

    Raises
    ------
    ValidationError if std is <= 0

    """

    mean = NumberParam("mean")
    std = NumberParam("std", low=0, low_open=True)

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        return rng.normal(loc=self.mean, scale=self.std, size=shape)


class Exponential(Distribution):
    """An exponential distribution (optionally with high values clipped).

    If ``high`` is left to its default value of infinity, this is a standard
    exponential distribution. If ``high`` is set, then any sampled values at
    or above ``high`` will be clipped so they are slightly below ``high``.
    This is useful for thresholding.

    The probability distribution function (PDF) is given by::

               |  0                                 if x < shift
        p(x) = | 1/scale * exp(-(x - shift)/scale)  if x >= shift and x < high
               |  n                                 if x == high - eps
               |  0                                 if x >= high

    where ``n`` is such that the PDF integrates to one, and ``eps`` is an
    infinitesimally small number such that samples of ``x`` are strictly less
    than ``high`` (in practice, ``eps`` depends on floating point precision).

    Parameters
    ----------
    scale : float
        The scale parameter (inverse of the rate parameter lambda). Larger
        values make the distribution narrower (sharper peak).
    shift : float, optional
        Amount to shift the distribution by. There will be no values smaller
        than this shift when sampling from the distribution.
    high : float, optional
        All values larger than or equal to this value will be clipped to
        slightly less than this value.
    """

    scale = NumberParam("scale", low=0, low_open=True)
    shift = NumberParam("shift")
    high = NumberParam("high")

    def __init__(self, scale, shift=0.0, high=np.inf):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.high = high

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        x = rng.exponential(self.scale, shape) + self.shift
        high = np.nextafter(self.high, np.asarray(-np.inf, dtype=x.dtype))
        return npext.clip(x, self.shift, high)


class UniformHypersphere(Distribution):
    """Uniform distribution on or in an n-dimensional unit hypersphere.

    Sample points are uniformly distributed across the volume (default) or
    surface of an n-dimensional unit hypersphere.

    Parameters
    ----------
    surface : bool, optional
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
    min_magnitude : Number, optional
        Lower bound on the returned vector magnitudes (such that they are in
        the range ``[min_magnitude, 1]``). Must be in the range [0, 1).
        Ignored if ``surface`` is ``True``.
    """

    surface = BoolParam("surface")
    min_magnitude = NumberParam("min_magnitude", low=0, high=1, high_open=True)

    def __init__(self, surface=False, min_magnitude=0):
        super().__init__()
        if surface and min_magnitude > 0:
            warnings.warn("min_magnitude ignored because surface is True")
        self.surface = surface
        self.min_magnitude = min_magnitude

    def sample(self, n, d=None, rng=np.random):
        if d is None or d < 1:  # check this, since other dists allow d = None
            raise ValidationError("Dimensions must be a positive integer", "d")

        samples = rng.randn(n, d)
        samples /= npext.norm(samples, axis=1, keepdims=True)

        if self.surface:
            return samples

        # Generate magnitudes for vectors from uniform distribution.
        # The (1 / d) exponent ensures that samples are uniformly distributed
        # in n-space and not all bunched up at the centre of the sphere.
        samples *= rng.uniform(low=self.min_magnitude ** d, high=1, size=(n, 1)) ** (
            1.0 / d
        )

        return samples


class QuasirandomSequence(Distribution):
    """Sequence for quasi Monte Carlo sampling the ``[0, 1]``-cube.

    This is similar to ``np.random.uniform(0, 1, size=(num, d))``, but with the
    additional property that each ``d``-dimensional point is uniformly scattered.

    While the sequence is defined deterministically, we introduce two stochastic
    elements to encourage heterogeneity in models using these sequences.
    First, we offset the start of the sequence by a random number between 0 and 1
    to ensure we don't oversample points aligned to the step size.
    Second, we shuffle the resulting sequence before returning to ensure we don't
    introduce correlations between parameters sampled from this distribution.

    This is based on the tutorial and code from [#]_.

    See Also
    --------
    ScatteredHypersphere

    References
    ----------
    .. [#] Martin Roberts. "The Unreasonable Effectiveness of Quasirandom Sequences."
       https://web.archive.org/web/20211029094035/http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    Examples
    --------

    .. testcode::

       rd = nengo.dists.QuasirandomSequence().sample(10000, 2)
       plt.scatter(*rd.T, c=np.arange(len(rd)), cmap='Blues', s=7)
    """

    @staticmethod
    def _phi(d, tol=1e-10):
        """Newton-Raphson-Method to calculate ``g = phi_d``."""
        x = 1.0
        for _ in range(100):
            dx = (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
            x -= dx
            if abs(dx) < tol:
                break
        else:
            raise ConvergenceError(f"'phi' computation did not converge for d={d}")

        return x

    def sample(self, n, d=1, rng=np.random):
        if d is None or d < 1:  # check this, since other dists allow d = None
            raise ValidationError("Dimensions must be a positive integer", "d")

        seed = rng.uniform(0, 1, size=d)
        if d == 1:
            # Tile the points optimally
            z = (seed + (1 / n) * np.arange(n, dtype=np.float64)) % 1
            z = z[:, np.newaxis]
        else:
            phi_inv = 1.0 / self._phi(d)
            alpha = phi_inv ** np.arange(1, d + 1)
            z = (seed + np.outer(np.arange(1, n + 1), alpha)) % 1
        rng.shuffle(z)
        return z


class ScatteredHypersphere(Distribution):
    r"""Quasirandom distribution over the hypersphere or hyperball.

    Applies a spherical transform to the given quasirandom sequence
    (by default `.QuasirandomSequence`) to obtain uniformly scattered samples.

    This distribution has the nice mathematical property that the discrepancy
    between the empirical distribution and :math:`n` samples is
    :math:`\widetilde{\mathcal{O}} (1 / n)` as opposed to
    :math:`\mathcal{O} (1 / \sqrt{n})` for the Monte Carlo method [1]_.
    This means that the number of samples is effectively squared, making this
    useful as a means for sampling ``eval_points`` and ``encoders``.

    Parameters
    ----------
    surface : bool, optional
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
    min_magnitude : Number, optional
        Lower bound on the returned vector magnitudes (such that they are in
        the range ``[min_magnitude, 1]``). Must be in the range [0, 1).
        Ignored if ``surface`` is ``True``.
    base : `.Distribution`, optional
        The base distribution from which to sample quasirandom numbers.
    method : {"sct-approx", "sct", "tfww"}
        Method to use for mapping points to the hypersphere.

        * "sct-approx": Same as "sct", but uses lookup table to approximate the
          beta distribution, making it faster with almost exactly the same result.
        * "sct": Use the exact Spherical Coordinate Transform
          (section 1.5.2 of [1]_).
        * "tfww": Use the Tashiro-Fang-Wang-Wong method (section 4.3 of [1]_).
          Faster than "sct" and "sct-approx", with the same level of uniformity
          for larger numbers of samples (``n >= 4000``, approximately).

    See Also
    --------
    UniformHypersphere
    QuasirandomSequence

    Notes
    -----
    The `.QuasirandomSequence` distribution is mostly deterministic.
    Nondeterminism comes from a random ``d``-dimensional rotation.

    References
    ----------
    .. [1] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.

    Examples
    --------
    Plot points sampled from the surface of the sphere in 3 dimensions:

    .. testcode::

       from mpl_toolkits.mplot3d import Axes3D

       points = nengo.dists.ScatteredHypersphere(surface=True).sample(1000, d=3)

       ax = plt.subplot(111, projection="3d")
       ax.scatter(*points.T, s=5)

    Plot points sampled from the volume of the sphere in 2 dimensions (i.e. circle):

    .. testcode::

       points = nengo.dists.ScatteredHypersphere(surface=False).sample(1000, d=2)
       plt.scatter(*points.T, s=5)
    """

    surface = BoolParam("surface")
    min_magnitude = NumberParam("min_magnitude", low=0, high=1, high_open=True)
    base = DistributionParam("base")
    method = EnumParam("method", values=("sct-approx", "sct", "tfww"))

    def __init__(
        self,
        surface=False,
        min_magnitude=0,
        base=QuasirandomSequence(),
        method="sct-approx",
    ):
        super().__init__()
        if surface and min_magnitude > 0:
            warnings.warn("min_magnitude ignored because surface is True")
        self.surface = surface
        self.min_magnitude = min_magnitude
        self.base = base
        self.method = method

        if self.method == "sct":
            import scipy.special  # pylint: disable=import-outside-toplevel

            assert scipy.special

    @classmethod
    def spherical_coords_ppf(cls, dims, y, approx=False):
        if not approx:
            import scipy.special  # pylint: disable=import-outside-toplevel

        y_reflect = np.where(y < 0.5, y, 1 - y)
        if approx:
            z_sq = _betaincinv22.lookup(dims, 2 * y_reflect)
        else:
            z_sq = scipy.special.betaincinv(dims / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)

    @classmethod
    def spherical_transform_sct(cls, samples, approx=False):
        """Map samples from the ``[0, 1]``-cube onto the hypersphere.

        Uses the SCT method described in section 1.5.3 of Fang and Wang (1994).
        """
        samples = np.asarray(samples)
        samples = samples[:, np.newaxis] if samples.ndim == 1 else samples
        n, d = samples.shape

        # inverse transform method (section 1.5.2)
        coords = np.empty_like(samples)
        for j in range(d):
            coords[:, j] = cls.spherical_coords_ppf(d - j, samples[:, j], approx=approx)

        # spherical coordinate transform
        mapped = np.ones((n, d + 1))
        i = np.ones(d)
        i[-1] = 2.0
        s = np.sin(i[np.newaxis, :] * np.pi * coords)
        c = np.cos(i[np.newaxis, :] * np.pi * coords)
        mapped[:, 1:] = np.cumprod(s, axis=1)
        mapped[:, :-1] *= c
        return mapped

    @staticmethod
    def spherical_transform_tfww(c_samples):
        """Map samples from the ``[0, 1]``-cube onto the hypersphere surface.

        Uses the TFWW method described in section 4.3 of Fang and Wang (1994).
        """
        c_samples = np.asarray(c_samples)
        c_samples = c_samples[:, np.newaxis] if c_samples.ndim == 1 else c_samples
        n, s1 = c_samples.shape
        s = s1 + 1

        x_samples = np.zeros((n, s))

        if s == 2:
            phi = 2 * np.pi * c_samples[:, 0]
            x_samples[:, 0] = np.cos(phi)
            x_samples[:, 1] = np.sin(phi)
            return x_samples

        even = s % 2 == 0
        m = s // 2 if even else (s - 1) // 2

        g = np.zeros((n, m + 1))
        g[:, -1] = 1
        for j in range(m - 1, 0, -1):
            g[:, j] = g[:, j + 1] * c_samples[:, j - 1] ** (
                (1.0 / j) if even else (2.0 / (2 * j + 1))
            )

        d = np.sqrt(np.diff(g, axis=1))

        phi = c_samples[:, m - 1 :]
        if even:
            phi *= 2 * np.pi
            x_samples[:, 0::2] = d * np.cos(phi)
            x_samples[:, 1::2] = d * np.sin(phi)
        else:
            # there is a mistake in eq. 4.3.7 here, see eq. 1.5.28 for correct version
            phi[:, 1:] *= 2 * np.pi
            f = 2 * d[:, 0] * np.sqrt(phi[:, 0] * (1 - phi[:, 0]))
            x_samples[:, 0] = d[:, 0] * (1 - 2 * phi[:, 0])
            x_samples[:, 1] = f * np.cos(phi[:, 1])
            x_samples[:, 2] = f * np.sin(phi[:, 1])
            if s > 3:
                x_samples[:, 3::2] = d[:, 1:] * np.cos(phi[:, 2:])
                x_samples[:, 4::2] = d[:, 1:] * np.sin(phi[:, 2:])

        return x_samples

    @staticmethod
    def random_orthogonal(d, rng=np.random):
        """Returns a random orthogonal matrix."""
        m = rng.standard_normal((d, d))
        u, _, v = np.linalg.svd(m)
        return np.dot(u, v)

    def sample(self, n, d=1, rng=np.random):
        if d == 1 and self.surface:
            return np.sign(self.base.sample(n, d, rng) - 0.5)

        if d == 1:
            pos_samples = self.base.sample(int(n / 2), d, rng)
            neg_samples = self.base.sample(n - pos_samples.size, d, rng)
            if self.min_magnitude > 0:
                for samples in [pos_samples, neg_samples]:
                    samples *= 1.0 - self.min_magnitude
                    samples += self.min_magnitude
            samples = np.vstack([pos_samples, -1 * neg_samples])
            rng.shuffle(samples)
            return samples

        radius = None
        if self.surface:
            samples = self.base.sample(n, d - 1, rng)
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:]
            if self.min_magnitude != 0:
                min_d = self.min_magnitude ** d
                radius *= 1 - min_d
                radius += min_d
            radius **= 1.0 / d

        if self.method == "sct":
            mapped = self.spherical_transform_sct(samples, approx=False)
        elif self.method == "sct-approx":
            mapped = self.spherical_transform_sct(samples, approx=True)
        else:
            assert self.method == "tfww"
            mapped = self.spherical_transform_tfww(samples)

        # radius adjustment for ball
        if radius is not None:
            mapped *= radius

        # random rotation
        rotation = self.random_orthogonal(d, rng=rng)
        return np.dot(mapped, rotation)


class Choice(Distribution):
    """Discrete distribution across a set of possible values.

    The same as Numpy random's `~numpy.random.RandomState.choice`,
    except can take vector or matrix values for the choices.

    Parameters
    ----------
    options : (N, ...) array_like
        The options (choices) to choose between. The choice is always done
        along the first axis, so if ``options`` is a matrix, the options are
        the rows of that matrix.
    weights : (N,) array_like, optional
        Weights controlling the probability of selecting each option. Will
        automatically be normalized. If None, weights be uniformly distributed.
    """

    options = NdarrayParam("options", shape=("*", "..."))
    weights = NdarrayParam("weights", shape=("*"), optional=True)

    def __init__(self, options, weights=None):
        super().__init__()
        self.options = options
        self.weights = weights

        weights = np.ones(len(self.options)) if self.weights is None else self.weights
        if len(weights) != len(self.options):
            raise ValidationError(
                f"Number of weights ({len(weights)}) must match "
                f"number of options ({len(self.options)})",
                attr="weights",
                obj=self,
            )
        if not all(weights >= 0):
            raise ValidationError(
                "All weights must be non-negative", attr="weights", obj=self
            )
        total = float(weights.sum())
        if total <= 0:
            raise ValidationError(
                f"Sum of weights must be positive (got {total:f})",
                attr="weights",
                obj=self,
            )
        self.p = weights / total

    @property
    def dimensions(self):
        return 0 if self.options.ndim == 1 else np.prod(self.options.shape[1:])

    def sample(self, n, d=None, rng=np.random):
        if d is not None and self.dimensions != d:
            raise ValidationError(
                f"Options must be of dimensionality {d} (got {self.dimensions})",
                attr="options",
                obj=self,
            )

        i = np.searchsorted(np.cumsum(self.p), rng.rand(n))
        return self.options[i]


class Samples(Distribution):
    """A set of samples.

    This class is a subclass of `.Distribution` so that it can be used in any
    situation that calls for a  `.Distribution`. However, the call to
    `.Distribution.sample` must match the dimensions of the samples or
    a `.ValidationError` will be raised.

    Parameters
    ----------
    samples : (n, d) array_like
        ``n`` and ``d`` must match what is eventually passed to
         `.Distribution.sample`.
    """

    samples = NdarrayParam("samples", shape=("...",))

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def sample(self, n, d=None, rng=np.random):
        samples = np.array(self.samples)
        shape = (n,) if d is None else (n, d)

        if d is None:
            samples = samples.squeeze()

        if d is not None and samples.ndim == 1:
            samples = samples[..., np.newaxis]

        if samples.shape[0] != shape[0]:
            raise ValidationError(
                "Wrong number of samples requested; got "
                f"{n}, should be {samples.shape[0]}",
                attr="samples",
                obj=self,
            )
        elif d is None and len(samples.shape) != 1:
            raise ValidationError(
                "Wrong sample dimensionality requested; got "
                f"'None', should be {samples.shape[1]}",
                attr="samples",
                obj=self,
            )
        elif d is not None and samples.shape[1] != shape[1]:
            raise ValidationError(
                "Wrong sample dimensionality requested; got "
                f"{d}, should be {samples.shape[1]}",
                attr="samples",
                obj=self,
            )

        return samples


class SqrtBeta(Distribution):
    """Distribution of the square root of a Beta distributed random variable.

    Given ``n + m`` dimensional random unit vectors, the length of subvectors
    with ``m`` elements will be distributed according to this distribution.

    Parameters
    ----------
    n: int
        Number of subvectors.
    m: int, optional
        Length of each subvector.

    See also
    --------
    nengo.dists.SubvectorLength
    """

    n = IntParam("n", low=0)
    m = IntParam("m", low=0)

    def __init__(self, n, m=1):
        super().__init__()
        self.n = n
        self.m = m

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        return np.sqrt(rng.beta(self.m / 2.0, self.n / 2.0, size=shape))

    def cdf(self, x):
        """Cumulative distribution function.

        .. note:: Requires SciPy.

        Parameters
        ----------
        x : array_like
            Evaluation points in [0, 1].

        Returns
        -------
        cdf : array_like
            Probability that ``X <= x``.
        """
        from scipy.special import betainc  # pylint: disable=import-outside-toplevel

        sq_x = x * x
        return np.where(
            sq_x < 1.0, betainc(self.m / 2.0, self.n / 2.0, sq_x), np.ones_like(x)
        )

    def pdf(self, x):
        """Probability distribution function.

        .. note:: Requires SciPy.

        Parameters
        ----------
        x : array_like
            Evaluation points in [0, 1].

        Returns
        -------
        pdf : array_like
            Probability density at ``x``.
        """
        from scipy.special import beta  # pylint: disable=import-outside-toplevel

        return (
            2
            / beta(0.5 * self.m, 0.5 * self.n)
            * x ** (self.m - 1)
            * (1 - x * x) ** (0.5 * self.n - 1)
        )

    def ppf(self, y):
        """Percent point function (inverse cumulative distribution).

        .. note:: Requires SciPy.

        Parameters
        ----------
        y : array_like
            Cumulative probabilities in [0, 1].

        Returns
        -------
        ppf : array_like
            Evaluation points ``x`` in [0, 1] such that ``P(X <= x) = y``.
        """
        from scipy.special import betaincinv  # pylint: disable=import-outside-toplevel

        sq_x = betaincinv(self.m / 2.0, self.n / 2.0, y)
        return np.sqrt(sq_x)


class SubvectorLength(SqrtBeta):
    """Distribution of the length of a subvectors of a unit vector.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the complete unit vector.
    subdimensions : int, optional
        Dimensionality of the subvector.

    See also
    --------
    nengo.dists.SqrtBeta
    """

    def __init__(self, dimensions, subdimensions=1):
        super().__init__(dimensions - subdimensions, subdimensions)

    @property
    def dimensions(self):
        return self.n + self.m

    @property
    def subdimensions(self):
        return self.m


class CosineSimilarity(SubvectorLength):
    """Distribution of the cosine of the angle between two random vectors.

    The "cosine similarity" is the cosine of the angle between two vectors,
    which is equal to the dot product of the vectors, divided by the L2-norms
    of the individual vectors. When these vectors are unit length, this is then
    simply the distribution of their dot product.

    This is also equivalent to the distribution of a single coefficient from a
    unit vector (a single dimension of ``UniformHypersphere(surface=True)``).
    Furthermore, ``CosineSimilarity(d+2)`` is equivalent to the distribution of
    a single coordinate from points uniformly sampled from the d-dimensional
    unit ball (a single dimension of
    ``UniformHypersphere(surface=False).sample(n, d)``). These relationships
    have been detailed in [Voelker2017]_.

    This can be used to calculate an intercept ``c = ppf(1 - p)`` such that
    ``dot(u, v) >= c`` with probability ``p``, for random unit vectors ``u``
    and ``v``. In other words, a neuron with intercept ``ppf(1 - p)`` will
    fire with probability ``p`` for a random unit length input.

    .. [Voelker2017]
       `Aaron R. Voelker, Jan Gosmann, and Terrence C. Stewart.
       Efficiently sampling vectors and coordinates from the n-sphere and
       n-ball. Technical Report, Centre for Theoretical Neuroscience,
       Waterloo, ON, 2017
       <http://compneuro.uwaterloo.ca/publications/voelker2017.html>`_

    Parameters
    ----------
    dimensions: int
        Dimensionality of the complete unit vector.

    See also
    --------
    nengo.dists.SqrtBeta
    """

    def __init__(self, dimensions):
        super().__init__(dimensions)

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        sign = Choice((1, -1)).sample(np.prod(shape), rng=rng).reshape(*shape)
        return sign * super().sample(n, d, rng=rng)

    def cdf(self, x):
        return (super().cdf(x) * np.sign(x) + 1) / 2.0

    def pdf(self, x):
        return super().pdf(x) / 2.0

    def ppf(self, y):
        x = super().ppf(abs(y * 2 - 1))
        return np.where(y > 0.5, x, -x)


class _betaincinv22:
    """Look up values for ``betaincinv(dims / 2, 0.5, x)``."""

    path = str(data_dir / "betaincinv22_table.npz")
    table = None

    @classmethod
    def make_table(cls, n_interp=200, n_dims=50):
        """Save lookup table for ``_betaincinv22.lookup``."""
        import scipy.special  # pylint: disable=import-outside-toplevel

        rng = np.random.RandomState(0)

        n_dims_log = int(0.5 * n_dims)
        dims_lin = np.arange(1, n_dims - n_dims_log + 1)
        dims_log = np.round(
            np.logspace(np.log10(dims_lin[-1] + 1), 3, n_dims_log)
        ).astype(dims_lin.dtype)
        dims = np.unique(np.concatenate([dims_lin, dims_log]))

        x_table = []
        y_table = []
        for dim in dims:
            n_range = int(0.8 * n_interp)
            x0 = np.linspace(0, 1, n_interp - n_range)  # samples in the domain
            y0 = np.linspace(1e-16, 1 - 1e-7, n_range)  # samples in the range
            x1 = scipy.special.betainc(dim / 2.0, 0.5, y0)
            interp_x = np.unique(np.concatenate([x0, x1]))
            while len(interp_x) < n_interp:
                # add random points until we achieve the length
                interp_x = np.unique(
                    np.concatenate(
                        [interp_x, rng.uniform(size=n_interp - len(interp_x))]
                    )
                )

            interp_x.sort()
            assert interp_x.size == n_interp

            interp_y = scipy.special.betaincinv(dim / 2.0, 0.5, interp_x)
            x_table.append(interp_x)
            y_table.append(interp_y)

        x_table = np.asarray(x_table)
        y_table = np.asarray(y_table)

        np.savez(cls.path, dims=dims, x=x_table, y=y_table)

    @classmethod
    def load_table(cls):
        if cls.table is not None:
            return cls.table
        data = np.load(cls.path)
        cls.table = {d: (x, y) for d, x, y in zip(data["dims"], data["x"], data["y"])}
        assert np.all(np.diff(list(cls.table)) >= 1)
        return cls.table

    @classmethod
    def lookup(cls, dims, x):
        if not is_integer(dims) or dims < 1:
            raise ValidationError("must be an integer >= 1", attr="dims")

        table = cls.load_table()

        if dims in table:  # pylint: disable=unsupported-membership-test
            xp, yp = table[dims]
        else:
            known_dims = np.array(list(table))
            i = np.searchsorted(known_dims, dims)
            assert i > 0
            if i >= len(known_dims):
                # dims is larger than any dimension we have, so use the largest
                xp, yp = table[known_dims[-1]]
            else:
                # take average of two curves
                dims0, dims1 = known_dims[i - 1], known_dims[i]
                xp0, yp0 = table[dims0]
                xp1, yp1 = table[dims1]
                assert dims0 < dims < dims1
                ratio0 = (dims1 - dims) / (dims1 - dims0)
                ratio1 = 1 - ratio0
                xp = (ratio0 * xp0 + ratio1 * xp1) if len(xp0) == len(xp1) else xp0
                yp = ratio0 * np.interp(xp, xp0, yp0) + ratio1 * np.interp(xp, xp1, yp1)

        return np.interp(x, xp, yp)
