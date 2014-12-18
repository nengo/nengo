"""These are helper functions to simplify some operations in the SPA module."""

import numpy as np

from nengo.spa.vocab import Vocabulary
from nengo.utils.compat import is_iterable


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
