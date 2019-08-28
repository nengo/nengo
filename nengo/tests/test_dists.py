import numpy as np
from numpy import array  # pylint: disable=unused-import
import pytest

from nengo import npext
from nengo.dists import (
    Choice,
    CosineSimilarity,
    DistOrArrayParam,
    Exponential,
    Gaussian,
    PDF,
    Samples,
    SqrtBeta,
    Uniform,
    UniformHypersphere,
)
from nengo.exceptions import ValidationError


def test_pdf(rng, allclose):
    s = 0.25
    f = lambda x: (
        np.exp(-0.5 * (x + 0.5) ** 2 / s ** 2) + np.exp(-0.5 * (x - 0.5) ** 2 / s ** 2)
    )

    xref = np.linspace(-2, 2, 101)
    pref = f(xref)
    pref /= pref.sum()
    dist = PDF(xref, pref)

    n = 100000
    samples = dist.sample(n, rng=rng)
    h, xedges = np.histogram(samples, bins=101)
    x = 0.5 * (xedges[:-1] + xedges[1:])
    dx = np.diff(xedges)
    y = h / float(h.sum()) / dx
    z = f(x)
    z = z / z.sum() / dx
    assert allclose(y, z, atol=0.05)

    with pytest.raises(ValidationError, match="PDF must sum to one"):
        dist = PDF([0, 1, 2], [0.1, 1.1, 0.1])

    with pytest.raises(ValidationError, match="`x` and `p` must be the same length"):
        dist = PDF([0, 1], [0, 1, 0])


@pytest.mark.parametrize("low,high", [(-2, -1), (-4, 1), (1, 2), (1, -1)])
def test_uniform(low, high, rng, allclose):
    n = 200
    dist = Uniform(low, high)
    samples = dist.sample(n, rng=rng)
    if low < high:
        assert np.all(samples >= low)
        assert np.all(samples < high)
    else:
        assert np.all(samples <= low)
        assert np.all(samples > high)
    histogram, _ = np.histogram(samples, bins=5)
    assert allclose(histogram, np.mean(histogram), atol=0.1 * n)

    # test `integer=true`
    dist = Uniform(low, high, integer=True)
    if low < high:
        samples = dist.sample(n, rng=rng)
        assert np.all(samples >= low)
        assert np.all(samples < high)
        assert np.all(samples % 1 == 0)
    else:
        with pytest.raises(ValueError):
            samples = dist.sample(n, rng=rng)


@pytest.mark.parametrize("mean,std", [(0, 1), (0, 0), (10, 2)])
def test_gaussian(mean, std, rng):
    n = 500
    if std <= 0:
        with pytest.raises(ValueError):
            dist = Gaussian(mean, std)
    else:
        dist = Gaussian(mean, std)
        samples = dist.sample(n, rng=rng)
        assert abs(np.mean(samples) - mean) < 3 * std / np.sqrt(n)
        assert abs(np.std(samples) - std) < 0.25


@pytest.mark.parametrize(
    "scale,shift,high", [(1.0, 0.0, np.inf), (10.0, 0.0, 1.0), (0.1, 0.3, 1.0)]
)
def test_exponential(scale, shift, high, rng):
    n = 100
    dist = Exponential(scale, shift=shift, high=high)
    samples = dist.sample(n, rng=rng)
    assert np.all(samples >= shift)
    assert np.all(samples <= high)
    # approximation of 95% confidence interval
    ci = scale * 1.96 / np.sqrt(n)
    if scale + ci < high:
        assert abs(np.mean(samples - shift) - scale) < ci


@pytest.mark.parametrize(
    "min_magnitude,d", [(0, 1), (0, 2), (0, 5), (0.6, 1), (0.3, 2), (0.4, 5)]
)
def test_hypersphere_volume(min_magnitude, d, rng, allclose):
    n = 250 * d
    dist = UniformHypersphere(min_magnitude=min_magnitude)
    samples = dist.sample(n, d, rng=rng)
    assert samples.shape == (n, d)
    assert allclose(np.mean(samples, axis=0), 0, atol=0.1)
    assert np.all(npext.norm(samples, axis=1) >= min_magnitude)


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_hypersphere_surface(dimensions, rng, allclose):
    n = 200 * dimensions
    dist = UniformHypersphere(surface=True)
    samples = dist.sample(n, dimensions, rng=rng)
    assert samples.shape == (n, dimensions)
    assert allclose(npext.norm(samples, axis=1), 1)
    assert allclose(np.mean(samples, axis=0), 0, atol=0.25 / dimensions)


def test_hypersphere_errors():
    with pytest.raises(ValidationError, match="Must be of type 'bool'"):
        UniformHypersphere(0)

    with pytest.raises(ValidationError, match="Dimensions must be a positive integer"):
        UniformHypersphere().sample(1, d=-1)


def test_hypersphere_warns():
    with pytest.warns(UserWarning, match="min_magnitude ignored because surface"):
        UniformHypersphere(surface=True, min_magnitude=0.1)


@pytest.mark.parametrize("weights", [None, [5, 1, 2, 9], [3, 2, 1, 0]])
def test_choice(weights, rng, allclose):
    n = 1000
    choices = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    N = len(choices)

    dist = Choice(choices, weights=weights)
    # If d is passed, it has to match
    with pytest.raises(ValueError):
        dist.sample(n, d=4, rng=rng)
    sample = dist.sample(n, rng=rng)
    tsample = [tuple(point) for point in sample]
    tchoices = [tuple(choice) for choice in choices]

    # check that frequency of choices matches weights
    inds = [tchoices.index(s) for s in tsample]
    histogram, _ = np.histogram(inds, bins=np.linspace(-0.5, N - 0.5, N + 1))
    p_empirical = histogram / float(histogram.sum())
    p = np.ones(N) / N if dist.p is None else dist.p
    sterr = 1.0 / np.sqrt(n)  # expected maximum standard error
    assert allclose(p, p_empirical, atol=2 * sterr)


def test_choice_errors():
    with pytest.raises(ValidationError, match="Number of weights.*must match.*options"):
        Choice([2], [1, 2, 3])

    with pytest.raises(ValidationError, match="All weights must be non-negative"):
        Choice([2], [-1])

    with pytest.raises(ValidationError, match="Sum of weights must be positive"):
        Choice([1, 2], [0, 0])


@pytest.mark.parametrize("shape", [(12, 2), (7, 1), (7,), (1, 1)])
def test_samples(shape, rng, allclose):
    samples = rng.random_sample(size=shape)
    d = Samples(samples)
    dims = None if len(shape) == 1 else shape[1]
    assert allclose(d.sample(shape[0], dims), samples)


@pytest.mark.parametrize("samples", [[1.0, 2.0, 3.0], [[1, 2], [3, 4]]])
def test_samples_list(samples, allclose):
    d = Samples(samples)
    shape = np.array(samples).shape
    dims = None if len(shape) == 1 else shape[1]
    assert allclose(d.sample(shape[0], dims), samples)


def test_samples_errors(rng):
    samples = rng.random_sample(size=(12, 2))
    with pytest.raises(ValidationError):
        Samples(samples).sample(11, 2)
    with pytest.raises(ValidationError):
        Samples(samples).sample(12, 1)
    with pytest.raises(ValidationError):
        Samples(samples).sample(12)

    samples = rng.random_sample(size=12)
    with pytest.raises(ValidationError):
        Samples(samples).sample(12, 2)


@pytest.mark.parametrize("n,m", [(99, 1), (50, 50)])
def test_sqrt_beta(n, m, rng):
    num_samples = 1000
    num_bins = 5

    vectors = rng.randn(num_samples, n + m)
    vectors /= npext.norm(vectors, axis=1, keepdims=True)
    expectation, _ = np.histogram(npext.norm(vectors[:, :m], axis=1), bins=num_bins)

    dist = SqrtBeta(n, m)
    samples = dist.sample(num_samples, 1, rng=rng)
    histogram, _ = np.histogram(samples, bins=num_bins)

    assert np.all(np.abs(np.asfarray(histogram - expectation) / num_samples) < 0.16)


@pytest.mark.parametrize("n,m", [(4, 1), (10, 5)])
def test_sqrt_beta_analytical(n, m, rng, allclose):
    """Tests pdf, cdf, and ppf of SqrtBeta distribution."""
    pytest.importorskip("scipy")  # beta and betainc

    dt = 0.001
    x = np.arange(dt, 1 + dt, dt)

    dist = SqrtBeta(n, m)

    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    ppf = dist.ppf(cdf)

    # The pdf should reflect the samples
    num_samples = 2500
    num_bins = 5

    samples = dist.sample(num_samples, rng=rng)
    act_hist, _ = np.histogram(samples, bins=num_bins)
    bin_points = np.linspace(0, 1, num_bins + 1)
    bin_cdf = dist.cdf(bin_points)
    exp_freq = bin_cdf[1:] - bin_cdf[:-1]
    assert np.all(np.abs(np.asfarray(act_hist) / num_samples - exp_freq) < 0.1)

    # The cdf should be the accumulated pdf
    assert allclose(cdf, np.cumsum(pdf) * dt, atol=0.01)

    # The ppf should give back x
    assert allclose(x, ppf, atol=0.01)


@pytest.mark.parametrize("d", [2, 3, 10, 50])
def test_cosine_similarity(d, rng):
    """Tests CosineSimilarity sampling."""
    num_samples = 2500
    num_bins = 8

    # Check that it gives a single dimension from UniformHypersphere
    exp_dist = UniformHypersphere(surface=True)
    act_dist = CosineSimilarity(d)

    exp = exp_dist.sample(num_samples, d, rng=rng)[:, 0]
    act = act_dist.sample(num_samples, rng=rng)

    exp_hist, _ = np.histogram(exp, bins=num_bins)
    act_hist, _ = np.histogram(act, bins=num_bins)

    assert np.all(np.abs(np.asfarray(exp_hist - act_hist) / num_samples) < 0.15)


@pytest.mark.parametrize("d", [2, 3, 10])
def test_cosine_analytical(d, allclose):
    pytest.importorskip("scipy")  # beta, betainc, betaincinv

    dt = 0.0001
    x = np.arange(-1 + dt, 1, dt)

    def p(x, d):
        # unnormalized CosineSimilarity distribution, derived by Eric H.
        return (1 - x * x) ** ((d - 3) / 2.0)

    dist = CosineSimilarity(d)

    pdf_exp = dist.pdf(x)
    pdf_act = p(x, d)

    cdf_exp = dist.cdf(x)
    cdf_act = np.cumsum(pdf_act) / np.sum(pdf_act)

    # Check that we get the expected pdf after normalization
    assert allclose(pdf_exp / np.sum(pdf_exp), pdf_act / np.sum(pdf_act), atol=0.01)

    # Check that this accumulates to the expected cdf
    assert allclose(cdf_exp, cdf_act, atol=0.01)

    # Check that the inverse cdf gives back x
    assert allclose(dist.ppf(cdf_exp), x, atol=0.01)


def test_cosine_sample_shape(seed, allclose):
    """"Tests that CosineSimilarity sample has correct shape."""
    # sampling (n, d) should be the exact same as sampling (n*d,)
    n = 3
    d = 4
    dist = CosineSimilarity(2)
    a = dist.sample(n, d, rng=np.random.RandomState(seed))
    b = dist.sample(n * d, rng=np.random.RandomState(seed))
    assert allclose(a.flatten(), b)


@pytest.mark.parametrize("d,p", [(3, 0), (5, 0.4), (10, 0.7), (50, 1.0)])
def test_cosine_intercept(d, p, rng, allclose):
    """Tests CosineSimilarity inverse cdf for finding intercepts."""
    pytest.importorskip("scipy")  # betaincinv

    num_samples = 500

    exp_dist = UniformHypersphere(surface=True)
    act_dist = CosineSimilarity(d)

    dots = exp_dist.sample(num_samples, d, rng=rng)[:, 0]

    # Find the desired intercept so that dots >= c with probability p
    c = act_dist.ppf(1 - p)
    assert allclose(np.sum(dots >= c) / float(num_samples), p, atol=0.05)


def test_distorarrayparam():
    """DistOrArrayParams can be distributions or samples."""

    class Test:
        dp = DistOrArrayParam("dp", default=None, sample_shape=["*", "*"])

    inst = Test()
    inst.dp = UniformHypersphere()
    assert isinstance(inst.dp, UniformHypersphere)
    inst.dp = np.array([[1], [2], [3]])
    assert np.all(inst.dp == np.array([[1], [2], [3]]))
    with pytest.raises(ValueError):
        inst.dp = "a"
    # Sample must have correct dims
    with pytest.raises(ValueError):
        inst.dp = np.array([1])


def test_distorarrayparam_sample_shape():
    """sample_shape dictates the shape of the sample that can be set."""

    class Test:
        dp = DistOrArrayParam("dp", default=None, sample_shape=["d1", 10])
        d1 = 4

    inst = Test()
    # Distributions are still cool
    inst.dp = UniformHypersphere()
    assert isinstance(inst.dp, UniformHypersphere)
    # Must be shape (4, 10)
    inst.dp = np.ones((4, 10))
    assert np.all(inst.dp == np.ones((4, 10)))
    with pytest.raises(ValidationError):
        inst.dp = np.ones((10, 4))
    assert np.all(inst.dp == np.ones((4, 10)))


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = Uniform(-0.3, 0.6)
    b = Uniform(-0.3, 0.6)
    c = Uniform(-0.2, 0.6)

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely
