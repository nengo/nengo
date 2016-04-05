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
