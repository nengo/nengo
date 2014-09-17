import numpy as np

import nengo
from nengo.dists import Choice
from nengo.networks.ensemblearray import EnsembleArray


class Product(nengo.Network):
    """Computes the element-wise product of two equally sized vectors."""

    def __init__(self, n_neurons, dimensions,
                 radius=1, encoders=nengo.Default, label=None, seed=None,
                 add_to_container=None, **ens_kwargs):
        super(Product, self).__init__(label, seed, add_to_container)
        self.config[nengo.Ensemble].update(ens_kwargs)

        with self:
            self.A = nengo.Node(size_in=dimensions, label="A")
            self.B = nengo.Node(size_in=dimensions, label="B")
            self.output = nengo.Node(size_in=dimensions, label="output")
            self.dimensions = dimensions

            if encoders is nengo.Default:
                encoders = Choice([[1, 1], [1, -1], [-1, 1], [-1, -1]])

            self.product = EnsembleArray(
                n_neurons, n_ensembles=dimensions, ens_dimensions=2,
                encoders=encoders, radius=np.sqrt(2) * radius)

            nengo.Connection(
                self.A, self.product.input[::2], synapse=None)
            nengo.Connection(
                self.B, self.product.input[1::2], synapse=None)

            self.output = self.product.add_output(
                'product', lambda x: x[0] * x[1])

    def dot_product_transform(self, scale=1.0):
        """Returns a transform for output to compute the scaled dot product."""
        return scale*np.ones((1, self.dimensions))
