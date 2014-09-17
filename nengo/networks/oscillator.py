import nengo


def Oscillator(recurrent_tau, frequency, n_neurons, net=None):
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
    return net
