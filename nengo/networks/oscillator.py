import nengo


class Oscillator(nengo.Network):
    def __init__(self, recurrent_tau, frequency, label=None, seed=None,
                 add_to_container=None, **ens_args):
        super(Oscillator, self).__init__(label, seed, add_to_container)
        with self:
            self.input = nengo.Node(label="In", size_in=2)
            self.ensemble = nengo.Ensemble(
                label="Oscillator", dimensions=2, **ens_args)
            tA = [[1, -frequency * recurrent_tau],
                  [frequency * recurrent_tau, 1]]
            nengo.Connection(self.ensemble, self.ensemble,
                             synapse=recurrent_tau, transform=tA)
            nengo.Connection(self.input, self.ensemble, synapse=None)
