from __future__ import absolute_import
import numpy as np

from nengo.exceptions import ValidationError
from nengo.params import (BoolParam, IntParam, NdarrayParam, NumberParam,
                          Parameter, Unconfigurable, FrozenObject)
import nengo.utils.numpy as npext


class Distribution(FrozenObject):
    """A base class for probability distributions.

    The only thing that a probabilities distribution need to define is a
    ``sample`` function. This base class ensures that all distributions
    accept the same arguments for the sample function.
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
            value will be of shape ``(n, d)``. If None (default), the return
            value will be of shape ``(n,)``.
        rng : RandomState, optional
            Random number generator state.

        Returns
        -------
        ndarray
            Samples as a 1d or 2d array depending on ``d``. The second
            dimension enumerates the dimensions of the process.
        """
        raise NotImplementedError("Distributions should implement sample.")


class PDF(Distribution):
    """An arbitrary distribution from a PDF.

    Parameters
    ----------
    x : vector_like (n,)
        Values of the points to sample from (interpolated).
    p : vector_like (n,)
        Probabilities of the `x` points.
    """

    x = NdarrayParam('x', shape='*')
    p = NdarrayParam('p', shape='*')

    def __init__(self, x, p):
        super(PDF, self).__init__()

        psum = np.sum(p)
        if np.abs(psum - 1) > 1e-8:
            raise ValidationError(
                "PDF must sum to one (sums to %f)" % psum, attr='p', obj=self)

        self.x = x
        self.p = p
        if len(self.x) != len(self.p):
            raise ValidationError(
                "`x` and `p` must be the same length", attr='p', obj=self)

        # make cumsum = [0] + cumsum, cdf = 0.5 * (cumsum[:-1] + cumsum[1:])
        cumsum = np.cumsum(p)
        cumsum *= 0.5
        cumsum[1:] = cumsum[:-1] + cumsum[1:]
        self.cdf = cumsum

    def __repr__(self):
        return "PDF(x=%r, p=%r)" % (self.x, self.p)

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

    low = NumberParam('low')
    high = NumberParam('high')
    integer = BoolParam('integer')

    def __init__(self, low, high, integer=False):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high
        self.integer = integer

    def __repr__(self):
        return "Uniform(low=%r, high=%r%s)" % (
            self.low, self.high, ", integer=True" if self.integer else "")

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
    ValidationError if ``std <= 0``

    """
    mean = NumberParam('mean')
    std = NumberParam('std', low=0, low_open=True)

    def __init__(self, mean, std):
        super(Gaussian, self).__init__()
        self.mean = mean
        self.std = std

    def __repr__(self):
        return "Gaussian(mean=%r, std=%r)" % (self.mean, self.std)

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        return rng.normal(loc=self.mean, scale=self.std, size=shape)


class Exponential(Distribution):
    """An exponential distribution (optionally with high values clipped).

    If `high` is left to its default value of infinity, this is a standard
    exponential distribution. If `high` is set, then any sampled values at
    or above `high` will be clipped so they are slightly below `high`. This is
    useful for thresholding and by extension, the AssociativeMemory.

    The probability distribution function (PDF) is given by
               |  0                                 if x < shift
        p(x) = | 1/scale * exp(-(x - shift)/scale)  if x >= shift and x < high
               |  n                                 if x == high - eps
               |  0                                 if x >= high
    where `n` is such that the PDF integrates to one, and `eps` is an
    infintesimally small number such that samples of `x` are strictly less than
    `high` (in practice, `eps` depends on the floating point precision).

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

    scale = NumberParam('scale', low=0, low_open=True)
    shift = NumberParam('shift')
    high = NumberParam('high')

    def __init__(self, scale, shift=0., high=np.inf):
        self.scale = scale
        self.shift = shift
        self.high = high

    def sample(self, n, d=None, rng=np.random):
        shape = self._sample_shape(n, d)
        x = rng.exponential(self.scale, shape) + self.shift
        high = np.nextafter(self.high, np.asarray(-np.inf, dtype=x.dtype))
        return np.clip(x, self.shift, high)


class UniformHypersphere(Distribution):
    """Uniform distribution on or in an n-dimensional unit hypersphere.

    Sample points are uniformly distibuted across the volume (default) or
    surface of an n-dimensional unit hypersphere.

    Parameters
    ----------
    surface : bool
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
        Default: False

    """
    surface = BoolParam('surface')

    def __init__(self, surface=False):
        super(UniformHypersphere, self).__init__()
        self.surface = surface

    def __repr__(self):
        return "UniformHypersphere(%s)" % (
            "surface=True" if self.surface else "")

    def sample(self, n, d, rng=np.random):
        if d is None or d < 1:  # check this, since other dists allow d = None
            raise ValidationError("Dimensions must be a positive integer", 'd')

        samples = rng.randn(n, d)
        samples /= npext.norm(samples, axis=1, keepdims=True)

        if self.surface:
            return samples

        # Generate magnitudes for vectors from uniform distribution.
        # The (1 / d) exponent ensures that samples are uniformly distributed
        # in n-space and not all bunched up at the centre of the sphere.
        samples *= rng.rand(n, 1) ** (1.0 / d)

        return samples


class Choice(Distribution):
    """Discrete distribution across a set of possible values.

    The same as `numpy.random.choice`, except can take vector or matrix values
    for the choices.

    Parameters
    ----------
    options : array_like (N, ...)
        The options (choices) to choose between. The choice is always done
        along the first axis, so if `options` is a matrix, the options are
        the rows of that matrix.
    weights : array_like (N,) (optional)
        Weights controlling the probability of selecting each option. Will
        automatically be normalized. Defaults to a uniform distribution.
    """
    options = NdarrayParam('options', shape=('*', '...'))
    weights = NdarrayParam('weights', shape=('*'), optional=True)

    def __init__(self, options, weights=None):
        super(Choice, self).__init__()
        self.options = options
        self.weights = weights

        weights = (np.ones(len(self.options)) if self.weights is None else
                   self.weights)
        if len(weights) != len(self.options):
            raise ValidationError(
                "Number of weights (%d) must match number of options (%d)"
                % (len(weights), len(self.options)), attr='weights', obj=self)
        if not all(weights >= 0):
            raise ValidationError("All weights must be non-negative",
                                  attr='weights', obj=self)
        total = float(weights.sum())
        if total <= 0:
            raise ValidationError("Sum of weights must be positive (got %f)"
                                  % total, attr='weights', obj=self)
        self.p = weights / total

    def __repr__(self):
        return "Choice(options=%r%s)" % (
            self.options,
            "" if self.weights is None else ", weights=%r" % self.weights)

    @property
    def dimensions(self):
        return np.prod(self.options.shape[1:])

    def sample(self, n, d=None, rng=np.random):
        if d is not None and self.dimensions != d:
            raise ValidationError("Options must be of dimensionality %d "
                                  "(got %d)" % (d, self.dimensions),
                                  attr='options', obj=self)

        i = np.searchsorted(np.cumsum(self.p), rng.rand(n))
        return self.options[i]


class SqrtBeta(Distribution):
    """Distribution of the square root of a Beta distributed random variable.

    Given `n + m` dimensional random unit vectors, the length of subvectors
    with `m` elements will be distributed according to this distribution.

    Parameters
    ----------
    n, m : Number
        Shape parameters of the distribution.

    See also
    --------
    SubvectorLength
    """
    n = IntParam('n', low=0)
    m = IntParam('m', low=0)

    def __init__(self, n, m=1):
        super(SqrtBeta, self).__init__()
        self.n = n
        self.m = m

    def __repr__(self):
        return "%s(n=%r, m=%r)" % (self.__class__.__name__, self.n, self.m)

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
        return np.sqrt(rng.beta(self.m / 2.0, self.n / 2.0, size=shape))

    def cdf(self, x):
        """Cumulative distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability that `X <= x`.
        """
        from scipy.special import betainc
        sq_x = x * x
        return np.where(
            sq_x < 1., betainc(self.m / 2.0, self.n / 2.0, sq_x),
            np.ones_like(x))

    def pdf(self, x):
        """Probability distribution function.

        Requires Scipy.

        Parameters
        ----------
        x : ndarray
            Evaluation points in [0, 1].

        Returns
        -------
        ndarray
            Probability density at `x`.
        """
        from scipy.special import beta
        return (2 / beta(self.m / 2.0, self.n / 2.0) * x ** (self.m - 1) *
                (1 - x * x) ** (self.n / 2.0 - 1))

    def ppf(self, y):
        """Percent point function (inverse cumulative distribution).

        Requires Scipy.

        Parameters
        ----------
        y : ndarray
            Cumulative probabilities in [0, 1].

        Returns
        -------
        ndarray
            Evaluation points `x` in [0, 1] such that `P(X <= x) = y`.
        """
        from scipy.special import betaincinv
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
    SqrtBeta
    """

    def __init__(self, dimensions, subdimensions=1):
        super(SubvectorLength, self).__init__(
            dimensions - subdimensions, subdimensions)

    def __repr__(self):
        return "%s(%r, subdimensions=%r)" % (
            self.__class__.__name__, self.n + self.m, self.m)


class CosineSimilarity(SubvectorLength):
    """Distribution of the cosine of the angle between two random vectors.

    The "cosine similarity" is the cosine of the angle between two vectors,
    which is equal to the dot product of the vectors, divided by the L2-norms
    of the individual vectors. When these vectors are unit length, this is then
    simply the distribution of their dot product.

    This is also equivalent to the distribution of a single coefficient from a
    unit vector (a single dimension of `UniformHypersphere(surface=True)`).

    This can be used to calculate an intercept `c = ppf(1 - p)` such that
    `dot(u, v) >= c` with probability `p`, for random unit vectors `u` and `v`.
    In other words, a neuron with intercept `ppf(1 - p)` will fire with
    probability `p` for a random unit length input.

    Parameters
    ----------
    dimensions: int
        Dimensionality of the complete unit vector.

    See also
    --------
    SqrtBeta
    """

    def __init__(self, dimensions):
        super(CosineSimilarity, self).__init__(dimensions)

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
        sign = Choice((1, -1)).sample(np.prod(shape), rng=rng).reshape(*shape)
        return sign * super(CosineSimilarity, self).sample(num, d, rng=rng)

    def cdf(self, x):
        return (super(CosineSimilarity, self).cdf(x) * np.sign(x) + 1) / 2.0

    def pdf(self, x):
        return super(CosineSimilarity, self).pdf(x) / 2.0

    def ppf(self, y):
        x = super(CosineSimilarity, self).ppf(abs(y*2 - 1))
        return np.where(y > 0.5, x, -x)


class DistributionParam(Parameter):
    """A Distribution."""

    equatable = True

    def validate(self, instance, dist):
        if dist is not None and not isinstance(dist, Distribution):
            raise ValidationError("'%s' is not a Distribution type" % dist,
                                  attr=self.name, obj=instance)
        super(DistributionParam, self).validate(instance, dist)


class DistOrArrayParam(NdarrayParam):
    """Can be a Distribution or samples from a distribution."""

    def __init__(self, name, default=Unconfigurable, sample_shape=None,
                 optional=False, readonly=None):
        super(DistOrArrayParam, self).__init__(
            name, default, sample_shape, optional, readonly)

    def validate(self, instance, distorarray):
        if isinstance(distorarray, Distribution):
            Parameter.validate(self, instance, distorarray)
            return distorarray
        return super(DistOrArrayParam, self).validate(instance, distorarray)
