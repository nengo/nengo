import nengo


def Oscillator(recurrent_tau, frequency, n_neurons, net=None):
    """A two-dimensional ensemble with interacting recurrent connections.

    The ensemble connects to itself in a manner similar to the integrator;
    however, here the two dimensions interact with each other to implement
    a cyclic oscillator.

    Parameters
    ----------
    recurrent_tau : float
        Time constant on the recurrent connection.
    frequency : float
        Desired frequency, in Hz, of the cyclic oscillation.
    n_neurons : int
        Number of neurons in the recurrently connected ensemble.

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
        The recurrently connected oscillatory ensemble.
    net.input : Node
        Provides the input signal.
    """
    if net is None:
        net = nengo.Network(label="Oscillator")
    with net:
        net.input = nengo.Node(label="In", size_in=2)
        net.ensemble = nengo.Ensemble(
            n_neurons, dimensions=2, label="Oscillator")
        tA = [[1, -frequency * recurrent_tau],
              [frequency * recurrent_tau, 1]]
        nengo.Connection(net.ensemble, net.ensemble,
                         synapse=recurrent_tau, transform=tA)
        nengo.Connection(net.input, net.ensemble, synapse=None)
    net.output = net.ensemble
    return net
