import nengo


def Integrator(recurrent_tau, n_neurons, dimensions, net=None):
    """An emsemble that connects to itself."""
    if net is None:
        net = nengo.Network(label="Integrator")
    with net:
        net.input = nengo.Node(size_in=dimensions)
        net.ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions)
        nengo.Connection(net.ensemble, net.ensemble, synapse=recurrent_tau)
        nengo.Connection(net.input, net.ensemble,
                         transform=recurrent_tau, synapse=None)
    return net
