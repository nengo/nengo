import numpy as np

import nengo
from nengo.spa.module import Module


class Compare(Module):
    """A module for computing the dot product of two inputs.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared
    vocab : Vocabulary, optional
        The vocabulary to use to interpret the vectors
    neurons_per_multiply : int
        Number of neurons to use in each product computation
    input_magnitude : float
        Effective input magnitude for the multiplication.
        The actual input magnitude will be this value times sqrt(2)
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
