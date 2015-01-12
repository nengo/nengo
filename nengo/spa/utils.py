"""These are helper functions to simplify some operations in the SPA module."""

import numpy as np

import nengo
from nengo.spa.vocab import Vocabulary, VocabularyParam
from nengo.utils.compat import is_iterable


def enable_spa_params(model):
    """Enables the SPA specific parameters on a model.

    Parameters
    ----------
    model : Network
        Model to activate SPA specific parameters for.
    """
    for obj_type in [nengo.Node, nengo.Ensemble]:
        model.config[obj_type].set_param(
            'vocab', VocabularyParam(None, optional=True))


def similarity(data, vocab):
    """Return the similarity between some data and the vocabulary.

    Parameters
    ----------
    data: array_like
        The data used for comparison.
    vocab: spa.Vocabulary, array_like
        Vocabulary (or list of vectors) to use to calculate
        the similarity values

    """
    if isinstance(vocab, Vocabulary):
        probe_vectors = vocab.vectors.T
    elif is_iterable(vocab):
        probe_vectors = np.matrix(vocab).T
    else:
        probe_vectors = vocab.T

    return np.dot(data, probe_vectors)
