import nengo


class Oscillator(object):
    def __init__(self, recurrent_tau, frequency, **ens_args):
        self.label = ens_args.pop('label', 'Oscillator')
        self.input = nengo.Node(size_in=2, label=self.label + '.input')
        self.ensemble = nengo.Ensemble(
            dimensions=2, label=self.label + '.ensemble', **ens_args)
        tA = [[1, -frequency * recurrent_tau],
              [frequency * recurrent_tau, 1]]
        nengo.Connection(self.ensemble, self.ensemble,
                         filter=recurrent_tau, transform=tA)
        nengo.Connection(self.input, self.ensemble)
