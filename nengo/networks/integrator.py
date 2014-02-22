import nengo


class Integrator(object):
    def __init__(self, recurrent_tau, **ens_args):
        self.label = ens_args.pop('label', 'Integrator')
        dimensions = ens_args.get('dimensions', 1)
        self.input = nengo.Node(size_in=dimensions,
                                label=self.label + '.input')
        self.ensemble = nengo.Ensemble(label=self.label + '.ensemble',
                                       **ens_args)
        nengo.Connection(self.ensemble, self.ensemble, filter=recurrent_tau)
        nengo.Connection(self.input, self.ensemble, transform=recurrent_tau)
