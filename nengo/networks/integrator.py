import nengo


class Integrator(nengo.Network):
    """An emsemble that connects to itself."""
    def __init__(self, recurrent_tau, **ens_args):
        dimensions = ens_args.get('dimensions', 1)
        self.integrator_input = nengo.Node(size_in=dimensions)
        self.ensemble = nengo.Ensemble(**ens_args)
        nengo.Connection(self.ensemble, self.ensemble, synapse=recurrent_tau)
        nengo.Connection(self.integrator_input, self.ensemble, transform=recurrent_tau)
