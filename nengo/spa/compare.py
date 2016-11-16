import numpy as np

import nengo
from nengo.params import Default, IntParam
from nengo.spa.module import Module
from nengo.spa.vocab import VocabularyOrDimParam


class Compare(Module):
    """A module for computing the dot product of two inputs.

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    neurons_per_dimension : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    kwargs
        Keyword arguments passed through to ``spa.Module``.
    """

    vocab = VocabularyOrDimParam('vocab', default=None, readonly=True)
    neurons_per_dimension = IntParam(
        'neurons_per_dimension', default=200, low=1, readonly=True)

    def __init__(self, vocab=Default, neurons_per_dimension=Default, **kwargs):
        super(Compare, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            self.product = nengo.networks.Product(
                self.neurons_per_dimension, self.vocab.dimensions)
            self.output = nengo.Node(size_in=1, label='output')
            nengo.Connection(self.product.output, self.output,
                             transform=np.ones((1, self.vocab.dimensions)))

        self.input_a = self.product.input_a
        self.input_b = self.product.input_b

        self.inputs = dict(
            input_a=(self.input_a, vocab), input_b=(self.input_b, vocab))
        self.outputs = dict(default=(self.output, None))
