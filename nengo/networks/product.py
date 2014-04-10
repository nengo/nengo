import numpy as np

import nengo
from nengo.networks import EnsembleArray


class Product(nengo.Network):
    """Computes the element-wise product of two equally sized vectors."""

    def __init__(self, neurons, dimensions, radius=1, **ens_args):
        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.output = nengo.Node(size_in=dimensions, label="output")

        self.dimensions = dimensions

        array_radius = np.sqrt(2) * radius
        encoders = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]],
                            dtype='float') / array_radius
        encoders = np.tile(
            encoders, ((neurons.n_neurons / 4) + 1, 1))[:neurons.n_neurons]

        self.product = EnsembleArray(
            neurons, dimensions, dimensions=2, encoders=encoders,
            radius=array_radius, **ens_args)

        nengo.Connection(
            self.A, self.product.input[::2], synapse=None)
        nengo.Connection(
            self.B, self.product.input[1::2], synapse=None)

        self.product.add_output('product', lambda x: x[0] * x[1])
        nengo.Connection(
            self.product.product, self.output, synapse=None)

    def dot_product_transform(self, scale=1.0):
        """Returns a transform for output to compute the scaled dot product."""
        return scale*np.ones((1, self.dimensions))
