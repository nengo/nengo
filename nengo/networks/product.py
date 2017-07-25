import warnings

import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray


def Product(n_neurons, dimensions, input_magnitude=1., net=None, **kwargs):
    """Computes the element-wise product of two equally sized vectors.

    The network used to calculate the product is described in
    `Gosmann, 2015`_. A simpler version of this network can be found in the
    `Multiplication example
    <https://www.nengo.ai/nengo/examples/multiplication.html>`_.

    Note that this network is optimized under the assumption that both input
    values (or both values for each input dimensions of the input vectors) are
    uniformly and independently distributed. Visualized in a joint 2D space,
    this would give a square of equal probabilities for pairs of input values.
    This assumption is violated with non-uniform input value distributions
    (for example, if the input values follow a Gaussian or cosine similarity
    distribution). In that case, no square of equal probabilities is obtained,
    but a probability landscape with circular equi-probability lines. To obtain
    the optimal network accuracy, scale the *input_magnitude* by a factor of
    ``1 / sqrt(2)``.

    .. _Gosmann, 2015:
       http://nbviewer.jupyter.org/github/ctn-archive/technical-reports/blob/
       master/Precise-multiplications-with-the-NEF.ipynb#An-alternative-network

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.

        .. note:: These neurons will be distributed evenly across two
                  ensembles. If an odd number of neurons is specified, the
                  extra neuron will not be used.
    dimensions : int
        Number of dimensions in each of the vectors to be multiplied.

    input_magnitude : float, optional (Default: 1.)
        The expected magnitude of the vectors to be multiplied.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.
    kwargs
        Keyword arguments passed through to ``nengo.Network``.

    Returns
    -------
    net : Network
        The newly built product network, or the provided ``net``.

    Attributes
    ----------
    net.input_a : Node
        The first vector to be multiplied.
    net.input_b : Node
        The second vector to be multiplied.
    net.output : Node
        The resulting product.
    net.sq1 : EnsembleArray
        Represents the first squared term. See `Gosmann, 2015`_ for details.
    net.sq2 : EnsembleArray
        Represents the second squared term. See `Gosmann, 2015`_ for details.
    """
    if net is None:
        kwargs.setdefault('label', "Product")
        net = nengo.Network(**kwargs)
    else:
        warnings.warn("The 'net' argument is deprecated.", DeprecationWarning)

    with net:
        net.input_a = net.A = nengo.Node(size_in=dimensions, label="input_a")
        net.input_b = net.B = nengo.Node(size_in=dimensions, label="input_b")
        net.output = nengo.Node(size_in=dimensions, label="output")

        net.sq1 = EnsembleArray(
            max(1, n_neurons // 2), n_ensembles=dimensions, ens_dimensions=1,
            radius=input_magnitude * np.sqrt(2))
        net.sq2 = EnsembleArray(
            max(1, n_neurons // 2), n_ensembles=dimensions, ens_dimensions=1,
            radius=input_magnitude * np.sqrt(2))

        tr = 1. / np.sqrt(2.)
        nengo.Connection(
            net.input_a, net.sq1.input, transform=tr, synapse=None)
        nengo.Connection(
            net.input_b, net.sq1.input, transform=tr, synapse=None)
        nengo.Connection(
            net.input_a, net.sq2.input, transform=tr, synapse=None)
        nengo.Connection(
            net.input_b, net.sq2.input, transform=-tr, synapse=None)

        sq1_out = net.sq1.add_output('square', np.square)
        nengo.Connection(sq1_out, net.output, transform=.5, synapse=None)
        sq2_out = net.sq2.add_output('square', np.square)
        nengo.Connection(sq2_out, net.output, transform=-.5, synapse=None)

    return net


def dot_product_transform(dimensions, scale=1.0):
    """Returns a transform for output to compute the scaled dot product."""
    return scale * np.ones((1, dimensions))
