"""These are helper functions to simplify some operations in the SPA module."""

import numpy as np

import nengo.utils.numpy as npext
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_iterable


def similarity(data, vocab, normalize=False):
    """Return the similarity between some data and the vocabulary.

    Computes the dot products between all data vectors and each
    vocabulary vector. If ``normalize=True``, normalizes all vectors
    to compute the cosine similarity.

    Parameters
    ----------
    data: array_like
        The data used for comparison.
    vocab: Vocabulary or array_like
        Vocabulary (or list of vectors) to use to calculate
        the similarity values.
    normalize : bool, optional (Default: False)
        Whether to normalize all vectors, to compute the cosine similarity.
    """
    from nengo.spa.vocab import Vocabulary

    if isinstance(vocab, Vocabulary):
        vectors = vocab.vectors
    elif is_iterable(vocab):
        vectors = np.array(vocab, copy=False, ndmin=2)
    else:
        raise ValidationError("%r object is not a valid vocabulary"
                              % (type(vocab).__name__), attr='vocab')

    data = np.array(data, copy=False, ndmin=2)
    dots = np.dot(data, vectors.T)

    if normalize:
        # Zero-norm vectors should return zero, so avoid divide-by-zero error
        eps = np.nextafter(0, 1)  # smallest float above zero
        dnorm = np.maximum(npext.norm(data, axis=1, keepdims=True), eps)
        vnorm = np.maximum(npext.norm(vectors, axis=1, keepdims=True), eps)

        dots /= dnorm
        dots /= vnorm.T

    return dots


def _shared_factor_set(k):
    """Returns a set of all numbers sharing a common factor with `k`"""
    f = []
    sqrt = lambda x: int(np.sqrt(x))
    for i in range(2, sqrt(k) + 1):
        if k % i == 0:
            ki = int(k / i)
            f.extend(p * i for p in range(1, ki))
            f.extend(p * ki for p in range(1, i))

    return set(f)


def cyclic_vector(d, k, n=None, rng=np.random):
    """Leaves a target vector unchanged after k convolutions.

    For example, if a is any vector and b = cyclic_vector(d, 4):

        a * b * b * b * b = a

    where * is the circular convolution operator. By corollary:

        b * b * b * b = [1, 0, ..., 0]

    Parameters
    ----------
    d : int
        The number of dimensions of the generated vector(s).
    k : int
        The number of convolutions required to reproduce the input.
    n : int (optional)
        The number of vectors to generate.
    rng : random number generator (optional)
        The random number generator to use.

    Output
    ------
    u : (n, d) array
        Array of vector(s). If 'n' is None, the shape will be `(d,)`.
    """
    d, k = int(d), int(k)
    if k < 2:
        raise ValueError("'k' must be at least 2 (got %d)" % k)
    if d < 3:
        raise ValueError("'d' must be at least 3 (got %d)" % d)

    d2 = (d - 1) / 2
    nn = 1 if n is None else n

    # Pick roots r such that r**k == 1
    roots = np.exp(2.j * np.pi / k * np.arange(k))
    rootpow = rng.randint(0, k, size=(nn, d2))

    # Ensure at least one root power in each vector is coprime with k, so that
    # no vector will take LESS than k convolutions to reproduce itself.
    coprimes = set(range(1, k)) - _shared_factor_set(k)
    for i in range(nn):
        # TODO: better method than rejection sampling?
        while not any(p in coprimes for p in rootpow[i]):
            rootpow[i] = rng.randint(0, k, size=d2)

    # Create array of Fourier coefficients such that U**k == ones(d)
    U = np.ones((nn, d), dtype=np.complex128)
    U[:, 1:1+d2] = roots[rootpow]

    # For even k, U[:, d2+1] = 1 or -1 are both valid
    if k % 2 == 0:
        U[:, 1+d2] = 2 * rng.randint(0, 2, size=nn) - 1

    # Respect Fourier symmetry conditions for real vectors
    U[:, -d2:] = U[:, d2:0:-1].conj()

    u = np.fft.ifft(U, axis=1).real
    return u[0] if n is None else u
