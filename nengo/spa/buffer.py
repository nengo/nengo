import warnings

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.spa.module import Module


class Buffer(Module):
    """A module capable of representing a single vector, with no memory.

    This is a minimal SPA module, useful for passing data along
    (for example, visual input).

    .. note:: Deprecated in Nengo 2.1.0. Use `.spa.State` instead.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the vector.
    subdimensions : int, optional (Default: 16)
        Size of the individual ensembles making up the vector.
        Must divide ``dimensions`` evenly.
    neurons_per_dimensions : int, optional (Default: 50)
        Number of neurons in an ensemble will be
        ``neurons_per_dimensions * subdimensions``.
    vocab : Vocabulary, optional (Default: None)
        The vocabulary to use to interpret the vector. If None,
        the default vocabulary for the given dimensionality is used.
    direct : bool, optional (Default: False)
        Whether or not to use direct mode for the neurons.

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """
    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=50,
                 vocab=None, direct=False, label=None, seed=None,
                 add_to_container=None):
        warnings.warn("Buffer is deprecated in favour of spa.State",
                      DeprecationWarning)
        super().__init__(label, seed, add_to_container)

        if vocab is None:
            # use the default one for this dimensionality
            vocab = dimensions
        elif vocab.dimensions != dimensions:
            raise ValidationError(
                "Dimensionality of given vocabulary (%d) does not match "
                "dimensionality of buffer (%d)" %
                (vocab.dimensions, dimensions), attr='dimensions', obj=self)

        # Subdimensions should be at most the number of dimensions
        subdimensions = min(dimensions, subdimensions)

        if dimensions % subdimensions != 0:
            raise ValidationError(
                "Number of dimensions (%d) must be divisible by subdimensions "
                "(%d)" % (dimensions, subdimensions),
                attr='dimensions', obj=self)

        with self:
            self.state = nengo.networks.EnsembleArray(
                neurons_per_dimension * subdimensions,
                dimensions // subdimensions,
                ens_dimensions=subdimensions,
                neuron_type=nengo.Direct() if direct else nengo.LIF(),
                radius=np.sqrt(float(subdimensions) / dimensions),
                label='state')

        self.inputs = dict(default=(self.state.input, vocab))
        self.outputs = dict(default=(self.state.output, vocab))
