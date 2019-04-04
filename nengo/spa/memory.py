import warnings

import nengo
from nengo.spa.buffer import Buffer


class Memory(Buffer):
    """A SPA module capable of storing a vector over time.

    Parameters are the same as `.spa.Buffer`, with the addition of
    ``synapse`` and ``tau``.

    .. note:: Deprecated in Nengo 2.1.0. Use `.spa.State` instead.

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
    synapse : float, optional (Default: 0.01)
        Synaptic filter to use on recurrent connection.
    vocab : Vocabulary, optional (Default: None)
        The vocabulary to use to interpret the vector. If None,
        the default vocabulary for the given dimensionality is used.
    tau : float or None, optional (Default: None)
        Effective time constant of the integrator. If None, it should
        have an infinite time constant.
    direct : bool, optional (Default: False)
        Whether or not to use direct mode for the neurons.

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=50,
                 synapse=0.01, vocab=None, tau=None, direct=False,
                 label=None, seed=None, add_to_container=None):
        warnings.warn("Memory is deprecated in favour of spa.State",
                      DeprecationWarning)
        super().__init__(
            dimensions=dimensions,
            subdimensions=subdimensions,
            neurons_per_dimension=neurons_per_dimension,
            vocab=vocab,
            direct=direct,
            label=label,
            seed=seed,
            add_to_container=add_to_container)

        if tau is None:
            transform = 1.0
        else:
            transform = 1.0 - synapse / tau

        with self:
            nengo.Connection(self.state.output, self.state.input,
                             transform=transform, synapse=synapse)
