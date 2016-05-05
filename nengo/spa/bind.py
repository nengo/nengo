import nengo
from nengo.exceptions import ValidationError
from nengo.networks.circularconvolution import HeuristicRadius
from nengo.params import Default, FunctionParam
from nengo.spa.module import Module


class Bind(Module):
    """A module for binding together two inputs.

    Binding is done with circular convolution. For more details on how
    this is computed, see the underlying `~.network.CircularConvolution`
    network.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared.
    vocab : Vocabulary, optional (Default: None)
        The vocabulary to use to interpret the vectors. If None,
        the default vocabulary for the given dimensionality is used.
    n_neurons : int, optional (Default: 200)
        Number of neurons to use in each product computation.
    invert_a, invert_b : bool, optional (Default: False, False)
        Whether to reverse the order of elements in either
        the first input (``invert_a``) or the second input (``invert_b``).
        Flipping the second input will make the network perform circular
        correlation instead of circular convolution.
    input_magnitude : float, optional (Default: 1.0)
        The expected magnitude of the vectors to be convolved.
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

    radius_method = FunctionParam(
        'radius_method', default=HeuristicRadius, readonly=True)

    def __init__(self, dimensions, vocab=None, n_neurons=200, invert_a=False,
                 invert_b=False, input_magnitude=1.0, radius_method=Default,
                 label=None, seed=None, add_to_container=None):
        super(Bind, self).__init__(label, seed, add_to_container)

        self.radius_method = radius_method

        if vocab is None:
            # use the default vocab for this number of dimensions
            vocab = dimensions
        elif vocab.dimensions != dimensions:
            raise ValidationError(
                "Dimensionality of given vocabulary (%d) does "
                "not match dimensionality of buffer (%d)" %
                (vocab.dimensions, dimensions), attr='dimensions', obj=self)

        with self:
            self.cc = nengo.networks.CircularConvolution(
                n_neurons, dimensions, invert_a, invert_b,
                input_magnitude=input_magnitude,
                radius_method=self.radius_method)
            self.A = self.cc.A
            self.B = self.cc.B
            self.output = self.cc.output

        self.inputs = dict(A=(self.A, vocab), B=(self.B, vocab))
        self.outputs = dict(default=(self.output, vocab))
