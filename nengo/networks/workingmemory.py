import warnings

import numpy as np

import nengo
from nengo.networks import EnsembleArray


def InputGatedMemory(n_neurons, dimensions, feedback=1.0,
                     difference_gain=1.0, recurrent_synapse=0.1,
                     difference_synapse=None, net=None, **kwargs):
    """Stores a given vector in memory, with input controlled by a gate.

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.
    dimensions : int
        Dimensionality of the vector.

    feedback : float, optional (Default: 1.0)
        Strength of the recurrent connection from the memory to itself.
    difference_gain : float, optional (Default: 1.0)
        Strength of the connection from the difference ensembles to the
        memory ensembles.
    recurrent_synapse : float, optional (Default: 0.1)

    difference_synapse : Synapse (Default: None)
        If None, ...
    kwargs
        Keyword arguments passed through to ``nengo.Network``.

    Returns
    -------
    net : Network
        The newly built memory network, or the provided ``net``.

    Attributes
    ----------
    net.diff : EnsembleArray
        Represents the difference between the desired vector and
        the current vector represented by ``mem``.
    net.gate : Node
        With input of 0, the network is not gated, and ``mem`` will be updated
        to minimize ``diff``. With input greater than 0, the network will be
        increasingly gated such that ``mem`` will retain its current value,
        and ``diff`` will be inhibited.
    net.input : Node
        The desired vector.
    net.mem : EnsembleArray
        Integrative population that stores the vector.
    net.output : Node
        The vector currently represented by ``mem``.
    net.reset : Node
        With positive input, the ``mem`` population will be inhibited,
        effectively wiping out the vector currently being remembered.

    """
    if net is None:
        kwargs.setdefault('label', "Input gated memory")
        net = nengo.Network(**kwargs)
    else:
        warnings.warn("The 'net' argument is deprecated.", DeprecationWarning)

    if difference_synapse is None:
        difference_synapse = recurrent_synapse

    n_total_neurons = n_neurons * dimensions

    with net:
        # integrator to store value
        net.mem = EnsembleArray(n_neurons, dimensions, label="mem")
        nengo.Connection(net.mem.output, net.mem.input,
                         transform=feedback,
                         synapse=recurrent_synapse)

        # calculate difference between stored value and input
        net.diff = EnsembleArray(n_neurons, dimensions, label="diff")
        nengo.Connection(net.mem.output, net.diff.input, transform=-1)

        # feed difference into integrator
        nengo.Connection(net.diff.output, net.mem.input,
                         transform=difference_gain,
                         synapse=difference_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        net.gate = nengo.Node(size_in=1)
        net.diff.add_neuron_input()
        nengo.Connection(net.gate, net.diff.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -10,
                         synapse=None)

        # reset input (if reset=1, remove all values, and set to 0)
        net.reset = nengo.Node(size_in=1)
        nengo.Connection(net.reset, net.mem.add_neuron_input(),
                         transform=np.ones((n_total_neurons, 1)) * -3,
                         synapse=None)

    net.input = net.diff.input
    net.output = net.mem.output

    return net
