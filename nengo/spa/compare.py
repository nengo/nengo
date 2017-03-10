import numpy as np

import nengo
from nengo.spa.module import Module


class Compare(Module):
    """A module for computing the dot product of two inputs.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared.
    vocab : Vocabulary, optional (Default: None)
        The vocabulary to use to interpret the vector. If None,
        the default vocabulary for the given dimensionality is used.
    neurons_per_multiply : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    input_magnitude : float, optional (Default: 1.0)
        The expected magnitude of the vectors to be multiplied.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """
    def __init__(self, dimensions, vocab=None, neurons_per_multiply=200,
                 input_magnitude=1.0, label=None, seed=None,
                 add_to_container=None):
        super(Compare, self).__init__(label, seed, add_to_container)
        if vocab is None:
            # use the default vocab for this number of dimensions
            vocab = dimensions

        with self:
            self.product = nengo.networks.Product(
                neurons_per_multiply, dimensions,
                input_magnitude=input_magnitude)

            # pylint: disable=invalid-name
            self.inputA = nengo.Node(size_in=dimensions, label='inputA')
            self.inputB = nengo.Node(size_in=dimensions, label='inputB')
            self.output = nengo.Node(size_in=1, label='output')

        self.inputs = dict(A=(self.inputA, vocab), B=(self.inputB, vocab))
        self.outputs = dict(default=(self.output, None))

        with self:
            nengo.Connection(self.inputA, self.product.A, synapse=None)
            nengo.Connection(self.inputB, self.product.B, synapse=None)
            nengo.Connection(self.product.output, self.output,
                             transform=np.ones((1, dimensions)))
