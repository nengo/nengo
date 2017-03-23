"""These are helper functions to simplify some operations in the SPA module."""
from itertools import combinations

import numpy as np

import nengo.utils.numpy as npext
from nengo.dists import CosineSimilarity
from nengo.exceptions import ValidationError
from nengo.spa.pointer import SemanticPointer
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

    if isinstance(data, SemanticPointer):
        data = data.v

    if isinstance(vocab, Vocabulary):
        vectors = vocab.vectors
    elif is_iterable(vocab):
        if isinstance(next(iter(vocab)), SemanticPointer):
            vocab = [p.v for p in vocab]
        vectors = np.array(vocab, copy=False, ndmin=2)
    else:
        raise ValidationError("%r object is not a valid vocabulary"
                              % (type(vocab).__name__), attr='vocab')

    dots = np.dot(vectors, data.T)

    if normalize:
        # Zero-norm vectors should return zero, so avoid divide-by-zero error
        eps = np.nextafter(0, 1)  # smallest float above zero
        dnorm = np.maximum(npext.norm(data.T, axis=0, keepdims=True), eps)
        vnorm = np.maximum(npext.norm(vectors, axis=1, keepdims=True), eps)

        if len(dots.shape) == 1:
            vnorm = np.squeeze(vnorm)

        dots /= dnorm
        dots /= vnorm

    return dots.T


def pairs(vocab):
    return set(x + '*' + y for x, y in combinations(vocab.keys(), 2))


def text(v, vocab, minimum_count=1, maximum_count=None,
         threshold=0.1, join=';', terms=None, normalize=False):
    """Return a human-readable text version of the provided vector.

    This is meant to give a quick text version of a vector for display
    purposes. To do this, compute the dot product between the vector
    and all the terms in the vocabulary. The top few vectors are
    chosen for inclusion in the text. It will try to only return
    terms with a match above the threshold, but will always return
    at least minimum_count and at most maximum_count terms. Terms
    are sorted from most to least similar.

    Parameters
    ----------
    v : SemanticPointer or ndarray
        The vector to convert into text.
    minimum_count : int, optional (Default: 1)
        Always return at least this many terms in the text.
    maximum_count : int, optional (Default: None)
        Never return more than this many terms in the text.
        If None, all terms will be returned.
    threshold : float, optional (Default: 0.1)
        How small a similarity for a term to be ignored.
    join : str, optional (Default: ';')
        The text separator to use between terms.
    terms : list, optional (Default: None)
        Only consider terms in this list of strings.
    normalize : bool, optional (Default: False)
        Whether to normalize the vector before computing similarity.
    """
    if not isinstance(v, SemanticPointer):
        v = SemanticPointer(v)
    if normalize:
        v = v.normalized()

    if terms is None:
        terms = vocab.keys()
        vectors = vocab.vectors
    else:
        vectors = vocab.parse_n(*terms)

    matches = list(zip(similarity(v, vectors), terms))
    matches.sort()
    matches.reverse()

    r = []
    for m in matches:
        if minimum_count is not None and len(r) < minimum_count:
            r.append(m)
        elif maximum_count is not None and len(r) == maximum_count:
            break
        elif threshold is None or m[0] > threshold:
            r.append(m)
        else:
            break

    return join.join(['%0.2f%s' % (sim, key) for (sim, key) in r])


def prob_cleanup(similarity, dimensions, vocab_size):
    """Estimate the chance of successful cleanup.

    This returns the chance that, out of vocab_size randomly chosen
    vectors, at least one of them will be closer to a particular
    vector than the value given by compare. To use this, compare
    your noisy vector with the ideal vector, pass that value in as
    the similarity parameter, and set ``vocab_size`` to be the number of
    competing vectors.

    Requires SciPy.
    """
    return CosineSimilarity(dimensions).cdf(similarity) ** vocab_size
