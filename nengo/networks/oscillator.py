import nengo


class Oscillator(nengo.Network):
    def make(self, recurrent_tau, frequency, **ens_args):
        with self:
            self.input = nengo.Node(label='In', dimensions=2)
            self.ensemble = nengo.Ensemble(
                label='Oscillator', dimensions=2, **ens_args)
            tA = [[1, -frequency * recurrent_tau],
                  [frequency * recurrent_tau, 1]]
            nengo.Connection(self.ensemble, self.ensemble,
                             filter=recurrent_tau, transform=tA)
            nengo.Connection(self.input, self.ensemble)
