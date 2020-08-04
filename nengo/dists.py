import warnings

import numpy as np

from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam,
    IntParam,
    NdarrayParam,
    NumberParam,
    Parameter,
    Unconfigurable,
    FrozenObject,
)
import nengo.utils.numpy as npext


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
           print("%.4f" % np.mean(samples))

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
                "PDF must sum to one (sums to %f)" % psum, attr="p", obj=self
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
                "Number of weights (%d) must match number of options (%d)"
                % (len(weights), len(self.options)),
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
                "Sum of weights must be positive (got %f)" % total,
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
                "Options must be of dimensionality %d "
                "(got %d)" % (d, self.dimensions),
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
                "%d, should be %d" % (n, samples.shape[0]),
                attr="samples",
                obj=self,
            )
        elif d is None and len(samples.shape) != 1:
            raise ValidationError(
                "Wrong sample dimensionality requested; got "
                "'None', should be %d" % (samples.shape[1],),
                attr="samples",
                obj=self,
            )
        elif d is not None and samples.shape[1] != shape[1]:
            raise ValidationError(
                "Wrong sample dimensionality requested; got "
                "%d, should be %d" % (d, samples.shape[1]),
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

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
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

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
        sign = Choice((1, -1)).sample(np.prod(shape), rng=rng).reshape(*shape)
        return sign * super().sample(num, d, rng=rng)

    def cdf(self, x):
        return (super().cdf(x) * np.sign(x) + 1) / 2.0

    def pdf(self, x):
        return super().pdf(x) / 2.0

    def ppf(self, y):
        x = super().ppf(abs(y * 2 - 1))
        return np.where(y > 0.5, x, -x)


class DistributionParam(Parameter):
    """A Distribution."""

    equatable = True

    def coerce(self, instance, dist):
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

    def coerce(self, instance, distorarray):
        if isinstance(distorarray, Distribution):
            return Parameter.coerce(self, instance, distorarray)
        return super().coerce(instance, distorarray)
