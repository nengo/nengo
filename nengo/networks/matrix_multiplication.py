import numpy as np

import nengo


def MatrixMult(n_neurons, shape_a, shape_b, net=None):
    """Computes the matrix product A*B.

    Both matrices need to be two dimensional.

    See the `Matrix Multiplication example`_ for a description of the network
    internals.

    Parameters
    ----------
    n_neurons : int
        Number of neurons used per product of two scalars.

        .. note:: If an odd number of neurons is given, one less neuron will be
                  used per product to obtain an even number. This is due to
                  the implementation the `.Product` network.
    shape_a : tuple
        Shape of the A input matrix.
    shape_b : tuple
        Shape of the B input matrix.
    net : Network, optional (Default: None)
        A network in which the network components will be built.
        This is typically used to provide a custom set of Nengo object
        defaults through modifying ``net.config``.

    Returns
    -------
    net : Network
        The newly built matrix multiplication network, or the provided ``net``.
    """

    if len(shape_a) != 2:
        raise ValueError("Shape {} is not two dimensional.".format(shape_a))
    if len(shape_b) != 2:
        raise ValueError("Shape {} is not two dimensional.".format(shape_a))
    if shape_a[1] != shape_b[0]:
        raise ValueError(
            "Matrix dimensions {} and  {} are incompatible".format(
                shape_a, shape_b))

    if net is None:
        net = nengo.Network(label="Matrix multiplication")

    size_a = np.prod(shape_a)
    size_b = np.prod(shape_b)

    with net:
        net.input_a = nengo.Node(size_in=size_a)
        net.input_b = nengo.Node(size_in=size_b)

        # The C matrix is composed of populations that each contain
        # one element of A and one element of B.
        # These elements will be multiplied together in the next step.
        size_c = size_a * shape_b[1]
        net.C = nengo.networks.Product(n_neurons, size_c)

        # Determine the transformation matrices to get the correct pairwise
        # products computed.  This looks a bit like black magic but if
        # you manually try multiplying two matrices together, you can see
        # the underlying pattern.  Basically, we need to build up D1*D2*D3
        # pairs of numbers in C to compute the product of.  If i,j,k are the
        # indexes into the D1*D2*D3 products, we want to compute the product
        # of element (i,j) in A with the element (j,k) in B.  The index in
        # A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
        # The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
        # two values per ensemble.  We add 1 to the B index so it goes into
        # the second value in the ensemble.
        transform_a = np.zeros((size_c, size_a))
        transform_b = np.zeros((size_c, size_b))

        for i, j, k in np.ndindex(shape_a[0], *shape_b):
            c_index = (j + k * shape_b[0] + i * size_b)
            transform_a[c_index][j + i * shape_b[0]] = 1
            transform_b[c_index][k + j * shape_b[1]] = 1

        nengo.Connection(
            net.input_a, net.C.A, transform=transform_a, synapse=None)
        nengo.Connection(
            net.input_b, net.C.B, transform=transform_b, synapse=None)

        # Now do the appropriate summing
        size_output = shape_a[0] * shape_b[1]
        net.output = nengo.Node(size_in=size_output)

        # The mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)
        transform_c = np.zeros((size_output, size_c))
        for i in range(size_c):
            transform_c[i // shape_b[0]][i] = 1

        nengo.Connection(
            net.C.output, net.output, transform=transform_c, synapse=None)

    return net
