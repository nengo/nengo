import warnings

import numpy as np

import nengo
from nengo.exceptions import ObsoleteError
from nengo.networks.ensemblearray import EnsembleArray


class Product(nengo.Network):
    """Computes the element-wise product of two equally sized vectors.

    The network used to calculate the product is described in
    `Gosmann, 2015`_. A simpler version of this network can be found in the
    :doc:`Multiplication example <examples/basic/multiplication>`.

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
       https://nbviewer.jupyter.org/github/ctn-archive/technical-reports/blob/
       master/Precise-multiplications-with-the-NEF.ipynb

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.

        .. note:: These neurons will be distributed evenly across two
                  ensembles. If an odd number of neurons is specified, the
                  extra neuron will not be used.
    dimensions : int
        Number of dimensions in each of the vectors to be multiplied.

    input_magnitude : float, optional
        The expected magnitude of the vectors to be multiplied.
        This value is used to determine the radius of the ensembles
        computing the element-wise product.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    input_a : Node
        The first vector to be multiplied.
    input_b : Node
        The second vector to be multiplied.
    output : Node
        The resulting product.
    sq1 : EnsembleArray
        Represents the first squared term. See `Gosmann, 2015`_ for details.
    sq2 : EnsembleArray
        Represents the second squared term. See `Gosmann, 2015`_ for details.
    """
    def __init__(self, n_neurons, dimensions, input_magnitude=1., **kwargs):
        if 'net' in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault('label', "Product")
        super().__init__(**kwargs)

        with self:
            self.input_a = nengo.Node(size_in=dimensions, label="input_a")
            self.input_b = nengo.Node(size_in=dimensions, label="input_b")
            self.output = nengo.Node(size_in=dimensions, label="output")

            self.sq1 = EnsembleArray(
                max(1, n_neurons // 2),
                n_ensembles=dimensions,
                ens_dimensions=1,
                radius=input_magnitude * np.sqrt(2),
            )
            self.sq2 = EnsembleArray(
                max(1, n_neurons // 2),
                n_ensembles=dimensions,
                ens_dimensions=1,
                radius=input_magnitude * np.sqrt(2),
            )

            tr = 1. / np.sqrt(2.)
            nengo.Connection(
                self.input_a, self.sq1.input, transform=tr, synapse=None)
            nengo.Connection(
                self.input_b, self.sq1.input, transform=tr, synapse=None)
            nengo.Connection(
                self.input_a, self.sq2.input, transform=tr, synapse=None)
            nengo.Connection(
                self.input_b, self.sq2.input, transform=-tr, synapse=None)

            sq1_out = self.sq1.add_output('square', np.square)
            nengo.Connection(sq1_out, self.output, transform=.5, synapse=None)
            sq2_out = self.sq2.add_output('square', np.square)
            nengo.Connection(sq2_out, self.output, transform=-.5, synapse=None)

    @property
    def A(self):
        warnings.warn(DeprecationWarning("Use 'input_a' instead of 'A'."))
        return self.input_a

    @property
    def B(self):
        warnings.warn(DeprecationWarning("Use 'input_b' instead of 'B'."))
        return self.input_b


def dot_product_transform(dimensions, scale=1.0):
    """Returns a transform for output to compute the scaled dot product."""
    return scale * np.ones((1, dimensions))
