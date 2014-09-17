import nengo


class Integrator(nengo.Network):
    def __init__(self, recurrent_tau, label=None, seed=None,
                 add_to_container=None, **ens_args):
        super(Integrator, self).__init__(label, seed, add_to_container)
        dimensions = ens_args.get('dimensions', 1)
        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.ensemble = nengo.Ensemble(**ens_args)
            nengo.Connection(self.ensemble, self.ensemble,
                             synapse=recurrent_tau)
            nengo.Connection(self.input, self.ensemble,
                             transform=recurrent_tau, synapse=None)
