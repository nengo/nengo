import nengo


class Integrator(nengo.Network):
    def make(self, recurrent_tau, **ens_args):
        dimensions = ens_args.get('dimensions', 1)
        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.ensemble = nengo.Ensemble(**ens_args)
            nengo.Connection(
                self.ensemble, self.ensemble, filter=recurrent_tau)
            nengo.Connection(
                self.input, self.ensemble, transform=recurrent_tau)
