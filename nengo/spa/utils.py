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


def similarity(data, probe, vocab):
    """Return the similarity between the probed data and the vocabulary.

    Parameters
    ----------
    data: ProbeDict
        Collection of simulation data returned by sim.run() function call.
    probe: Probe
        Probe with desired data.
    vocab: spa.Vocabulary, list, np.ndarray, np.matrix
        Vocabulary (or list of vectors) to use to calculate
        the similarity values

    """
    if isinstance(vocab, Vocabulary):
        probe_vectors = vocab.vectors.T
    elif is_iterable(vocab):
        probe_vectors = np.matrix(vocab).T
    else:
        probe_vectors = vocab.T

    return np.dot(data[probe], probe_vectors)
