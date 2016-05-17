from __future__ import division

import numpy as np

from nengo.networks.ensemblearray import EnsembleArray


def HeuristicRadius(dimensions, subdimensions):
    """Uses a heuristic to determine the radius for representing a unit vector.

    The heuristic used is 3.5 * sqrt(subdimensions / dimensions).
    """
    return 3.5 * np.sqrt(subdimensions / dimensions)


class UnitVectorArray(EnsembleArray):
    """An array of ensembles optimized to represent unit vectors.

    Parameters
    ----------
    n_neurons_per_dim : int
        The number of neurons per dimensions.
    dimensions : int
        The number of dimensions to represent.

    subdimensions : int, optional (Default: 16)
        The dimensionality of each sub-ensemble.
    represent_identity : bool, optional (Default: False)
        Whether to use a radius of one for the first ensemble. This will give
        a better representation of the identity vector at the cost of a worse
        representation of other vectors (more so for low dimensionality and
        higher `subdimensions`).
    radius : function, optional (Default: HeuristicRadius)
        Function that takes `(dimensions, subdimensions)` as argument and
        provides the radius for the ensembles.
    label : str, optional (Default: None)
        A name to assign this UnitVectorArray.
        Used for visualization and debugging.
    seed : int, optional (Default: None)
        Random number seed that will be used in the build step.
    add_to_container : bool, optional (Default: None)
        Determines if this network will be added to the current container.
        If None, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.

    Attributes
    ----------
    dimensions_per_ensemble : int
        The dimensionality of each sub-ensemble.
    ea_ensembles : list
        The sub-ensembles in the ensemble array.
    input : Node
        A node that provides input to all of the ensembles in the array.
    n_ensembles : int
        The number of sub-ensembles to create.
    n_neurons : int
        The number of neurons in each sub-ensemble.
    subdimensions : int
        Number of subdimensions per ensemble.
    represent_identity : bool
        Whether the radius of the first ensemble was set to one for a better
        representation of the identity vector.
    radius : function, optional (Default: HeuristicRadius)
        Function that takes `(dimensions, subdimensions)` as argument and
        provides the radius for the ensembles.
    neuron_input : Node or None
        A node that provides input to all the neurons in the ensemble array.
        None unless created in `~.EnsembleArray.add_neuron_input`.
    neuron_output : Node or None
        A node that gathers neural output from all the neurons in the ensemble
        array. None unless created in `~.EnsembleArray.add_neuron_output`.
    output : Node
        A node that gathers decoded output from all of the ensembles
        in the array.
    """
    def __init__(
            self, n_neurons_per_dim, dimensions, subdimensions=16,
            represent_identity=False, radius=HeuristicRadius, label=None,
            seed=None, add_to_container=None):
        if dimensions % subdimensions != 0:
            raise ValueError("Dimensions is not divisible by subdimensions.")

        super(UnitVectorArray, self).__init__(
            n_neurons_per_dim * subdimensions, dimensions // subdimensions,
            subdimensions, radius=radius(dimensions, subdimensions),
            label=label, seed=seed, add_to_container=add_to_container)

        self.n_neurons_per_dim = n_neurons_per_dim
        self.subdimensions = subdimensions
        self.represent_identity = represent_identity

        if represent_identity:
            self.ea_ensembles[0].radius = 1.
