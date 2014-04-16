import nengo


class Integrator(nengo.Network):
    def __init__(self, recurrent_tau, **ens_args):
        dimensions = ens_args.get('dimensions', 1)
        self.input = nengo.Node(size_in=dimensions)
        self.ensemble = nengo.Ensemble(**ens_args)
        nengo.Connection(self.ensemble, self.ensemble, synapse=recurrent_tau)
        nengo.Connection(self.input, self.ensemble, transform=recurrent_tau)
