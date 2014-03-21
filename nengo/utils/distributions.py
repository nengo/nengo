from __future__ import absolute_import
import numpy as np


class Distribution(object):
    """A base class for probability distributions.

    The only thing that a probabilities distribution needs is a
    ``sample`` function. This base class ensures that all distributions
    accept the same arguments for the sample function.
    """

    def sample(self, n, rng=np.random):
        raise NotImplementedError("Distributions should implement sample.")


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
    integer : boolean
        If true, sample from a uniform distribution of integers. In this case,
        low and high should be integers.
    """

    def __init__(self, low, high, integer=False):
        self.low = low
        self.high = high
        self.integer = integer

    def __eq__(self, other):
        return (self.low == other.low
                and self.high == other.high
                and self.integer == other.integer)

    def sample(self, n, rng=np.random):
        if self.integer:
            return rng.randint(low=self.low, high=self.high, size=n)
        else:
            return rng.uniform(low=self.low, high=self.high, size=n)


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
    ValueError if ``std <= 0``

    """

    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError("std must be greater than 0; passed %f" % std)
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.std

    def sample(self, n, rng=np.random):
        return rng.normal(loc=self.mean, scale=self.std, size=n)


class UniformHypersphere(Distribution):
    """Distributions over an n-dimensional unit hypersphere.

    Parameters
    ----------
    dimensions : Number
        The dimensionality of the hypersphere; i.e., the length
        of each sample point vector.
    surface : bool
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
        Default: False

    """

    def __init__(self, dimensions, surface=False):
        if dimensions < 1:
            raise ValueError("Hypersphere must have dimensions > 0")
        if not isinstance(dimensions, int):
            raise ValueError("Hyperphere only defined for integer dimensions")
        self.dimensions = dimensions
        self.surface = surface

    def sample(self, n, rng=np.random):
        samples = rng.randn(n, self.dimensions)

        # normalize magnitude of sampled points to be of unit length
        norm = np.sum(samples * samples, axis=1)
        samples /= np.sqrt(norm)[:, np.newaxis]

        if self.surface:
            return samples

        # generate magnitudes for vectors from uniform distribution
        scale = rng.rand(n, 1) ** (1.0 / self.dimensions)

        # scale sample points
        samples *= scale
        return samples
