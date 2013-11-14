import nengo


class Integrator(nengo.Network):
    def make(self, recurrent_tau, **ens_args):
        with self:
            self.input = nengo.Node()
            self.ensemble = nengo.Ensemble(**ens_args)
            nengo.Connection(
                self.ensemble, self.ensemble, filter=recurrent_tau)
            nengo.Connection(
                self.input, self.ensemble, transform=recurrent_tau)
