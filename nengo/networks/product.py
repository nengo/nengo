import numpy as np

import nengo
from nengo.dists import Choice
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.stdlib import nested


def Product(n_neurons, dimensions, input_magnitude=1, config=None, net=None):
    """Computes the element-wise product of two equally sized vectors."""
    if net is None:
        net = nengo.Network(label="Product")

    if config is None:
        config = nengo.Config(nengo.Ensemble)
        config[nengo.Ensemble].encoders = Choice(
            [[1, 1], [1, -1], [-1, 1], [-1, -1]])

    with nested(net, config):
        net.A = nengo.Node(size_in=dimensions, label="A")
        net.B = nengo.Node(size_in=dimensions, label="B")
        net.output = nengo.Node(size_in=dimensions, label="output")

        net.product = EnsembleArray(n_neurons, n_ensembles=dimensions,
                                    ens_dimensions=2,
                                    radius=input_magnitude * np.sqrt(2))
        nengo.Connection(net.A, net.product.input[::2], synapse=None)
        nengo.Connection(net.B, net.product.input[1::2], synapse=None)
        net.output = net.product.add_output('product', lambda x: x[0] * x[1])

    return net


def dot_product_transform(dimensions, scale=1.0):
    """Returns a transform for output to compute the scaled dot product."""
    return scale * np.ones((1, dimensions))
