import numpy as np

import nengo
from nengo.dists import Choice
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
    output_scaling : float
        Multiplier on the dot product result
    radius : float
        Effective radius for the multiplication.  The actual radius will
        be this value times sqrt(2)
    direct : boolean
        Whether or not to use direct mode for the neurons
    """
    def __init__(self, dimensions, vocab=None, neurons_per_multiply=200,
                 output_scaling=1.0, radius=1.0, direct=False,
                 label=None, seed=None, add_to_container=None):
        super(Compare, self).__init__(label, seed, add_to_container)
        if vocab is None:
            # use the default vocab for this number of dimensions
            vocab = dimensions

        self.output_scaling = output_scaling
        self.dimensions = dimensions

        with self:
            self.compare = nengo.networks.EnsembleArray(
                neurons_per_multiply, dimensions, ens_dimensions=2,
                neuron_type=nengo.Direct() if direct else nengo.LIF(),
                encoders=Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
                radius=radius * np.sqrt(2),
                label='compare')

            self.inputA = nengo.Node(size_in=dimensions, label='inputA')
            self.inputB = nengo.Node(size_in=dimensions, label='inputB')
            self.output = nengo.Node(size_in=1, label='output')

        self.inputs = dict(A=(self.inputA, vocab), B=(self.inputB, vocab))
        self.outputs = dict(default=(self.output, None))

        with self:
            nengo.Connection(self.inputA,
                             self.compare.input[::2], synapse=None)
            nengo.Connection(self.inputB,
                             self.compare.input[1::2], synapse=None)
            self.compare.add_output('product', lambda x: x[0] * x[1])

    def on_add(self, spa):
        Module.on_add(self, spa)

        with self:
            nengo.Connection(self.compare.product,
                             self.output,
                             transform=self.output_scaling *
                             np.ones((1, self.dimensions)))
