from __future__ import absolute_import
import numpy as np

import nengo.utils.numpy as npext


class Distribution(object):
    """A base class for probability distributions.

    The only thing that a probabilities distribution need to define is a
    ``sample`` function. This base class ensures that all distributions
    accept the same arguments for the sample function.

    """

    def sample(self, n, d=None, rng=np.random):
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
    integer : boolean, optional
        If true, sample from a uniform distribution of integers. In this case,
        low and high should be integers.
    """

    def __init__(self, low, high, integer=False):
        self.low = low
        self.high = high
        self.integer = integer

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.low == other.low
                and self.high == other.high
                and self.integer == other.integer)

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
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
    ValueError if ``std <= 0``

    """

    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError("std must be greater than 0; passed %f" % std)
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.mean == other.mean
                and self.std == other.std)

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        return rng.normal(loc=self.mean, scale=self.std, size=shape)


class UniformHypersphere(Distribution):
    """Distributions over an n-dimensional unit hypersphere.

    Parameters
    ----------
    surface : bool
        Whether sample points should be distributed uniformly
        over the surface of the hyperphere (True),
        or within the hypersphere (False).
        Default: False

    """

    def __init__(self, surface=False):
        self.surface = surface

    def sample(self, n, d, rng=np.random):
        if d is None or d < 1:  # check this, since other dists allow d = None
            raise ValueError("Dimensions must be a positive integer")

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

    def __init__(self, options, weights=None):
        self.options = np.array(options)

        if weights is None:
            self.p = None
        else:
            weights = np.asarray(weights)
            if len(weights) != len(self.options):
                raise ValueError(
                    "Number of weights (%d) must match number of options (%d)"
                    % (len(weights), len(self.options)))
            if not all(weights >= 0):
                raise ValueError("All weights must be non-negative")
            total = float(weights.sum())
            if total <= 0:
                raise ValueError(
                    "Sum of weights must be positive (got %f)" % total)
            self.p = weights / total

    def sample(self, n, rng=np.random):
        i = rng.choice(len(self.options), p=self.p, size=n)
        return self.options[i]
