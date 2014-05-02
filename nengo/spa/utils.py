"""These are helper functions to simplify some operations in the SPA module."""

import numpy as np


def similarity(data, probe):
    """Return the similarity between the probed data and the vocabulary."""
    return np.dot(data[probe], probe.target.vocab.vectors.T)
