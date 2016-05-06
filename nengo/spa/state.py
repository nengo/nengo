import nengo
from nengo.exceptions import ValidationError
from nengo.networks.unitvectorarray import HeuristicRadius, UnitVectorArray
from nengo.params import Default, FunctionParam, IntParam
from nengo.spa.module import Module


class State(Module):
    """A module capable of representing a single vector, with optional memory.

    This is a minimal SPA module, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the vector.
    subdimensions : int, optional (Default: 16)
        Size of the individual ensembles making up the vector.
        Must divide ``dimensions`` evenly.
    neurons_per_dimensions : int, optional (Default: 50)
        Number of neurons in an ensemble will be
        ``neurons_per_dimensions * subdimensions``.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    radius_method : function
        Function that provides the radius for the underlying ensembles. It
        takes `(dimensions, subdimensions)` as arguments.
    vocab : Vocabulary, optional (Default: None)
        The vocabulary to use to interpret the vector. If None,
        the default vocabulary for the given dimensionality is used.

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
    dim_per_ensemble = IntParam('dim_per_ensemble', default=16, readonly=True)

    def __init__(
            self, dimensions, subdimensions=Default, neurons_per_dimension=50,
            feedback=0.0, feedback_synapse=0.1, represent_identity=True,
            radius_method=Default, vocab=None, label=None, seed=None,
            add_to_container=None):
        super(State, self).__init__(label, seed, add_to_container)
        self.dim_per_ensemble = subdimensions

        self.radius_method = radius_method

        if vocab is None:
            # use the default one for this dimensionality
            vocab = dimensions
        elif vocab.dimensions != dimensions:
            raise ValidationError(
                "Dimensionality of given vocabulary (%d) does not "
                "match dimensionality of buffer (%d)" %
                (vocab.dimensions, dimensions), attr='dimensions', obj=self)

        # Subdimensions should be at most the number of dimensions
        subdimensions = min(dimensions, self.dim_per_ensemble)

        if dimensions % subdimensions != 0:
            raise ValidationError(
                "Dimensions (%d) must be divisible by subdimensions (%d)"
                % (dimensions, subdimensions), attr='dimensions', obj=self)

        with self:
            self.state_ensembles = UnitVectorArray(
                neurons_per_dimension, dimensions, subdimensions,
                represent_identity=represent_identity,
                radius=self.radius_method, label="state")
            self.input = self.state_ensembles.input
            self.output = self.state_ensembles.output

        self.inputs = dict(default=(self.input, vocab))
        self.outputs = dict(default=(self.output, vocab))

        with self:
            if feedback is not None and feedback != 0.0:
                nengo.Connection(self.output, self.input,
                                 transform=feedback,
                                 synapse=feedback_synapse)
