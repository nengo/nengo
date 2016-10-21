import nengo


def Integrator(recurrent_tau, n_neurons, dimensions, net=None):
    """An ensemble that accumulates input and maintains state.

    This is accomplished through scaling the input signal and recurrently
    connecting an ensemble to itself to maintain state.

    Parameters
    ----------
    recurrent_tau : float
        Time constant on the recurrent connection.
    n_neurons : int
        Number of neurons in the recurrently connected ensemble.
    dimensions : int
        Dimensionality of the input signal and ensemble.

    net : Network, optional (Default: None)
        A network in which the network components will be built.
        This is typically used to provide a custom set of Nengo object
        defaults through modifying ``net.config``.

    Returns
    -------
    net : Network
        The newly built product network, or the provided ``net``.

    Attributes
    ----------
    net.ensemble : Ensemble
        The recurrently connected ensemble.
    net.input : Node
        Provides the input signal.
    """
    if net is None:
        net = nengo.Network(label="Integrator")
    with net:
        net.input = nengo.Node(size_in=dimensions)
        net.ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions)
        nengo.Connection(net.ensemble, net.ensemble, synapse=recurrent_tau)
        nengo.Connection(net.input, net.ensemble,
                         transform=recurrent_tau, synapse=None)
    net.output = net.ensemble
    return net
