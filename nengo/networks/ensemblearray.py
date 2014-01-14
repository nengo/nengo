import copy

import numpy as np

import nengo


class EnsembleArray(nengo.Network):
    def make(self, neurons, n_ensembles, **ens_args):
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_args.pop('dimensions', 1)
        self.ensembles = []
        transform = np.eye(self.dimensions)

        with self:
            self.input = nengo.Node(dimensions=self.dimensions)

            for i in range(n_ensembles):
                e = nengo.Ensemble(copy.deepcopy(neurons), self.dimensions_per_ensemble, **ens_args)
                trans = transform[i * self.dimensions_per_ensemble:
                                  (i + 1) * self.dimensions_per_ensemble, :]
                nengo.Connection(self.input, e, transform=trans, filter=None)
                self.ensembles.append(e)

            self.add_output('output', function=None)

    def add_output(self, name, function):
        if function is None:
            function_d = self.dimensions_per_ensemble
        else:
            func_output = function(np.zeros(self.dimensions_per_ensemble))
            function_d = (1 if isinstance(func_output, (int, float))
                          else len(func_output))
        transform = np.eye(self.n_ensembles*function_d)

        output = nengo.Node(dimensions=self.n_ensembles*function_d, label=name)
        setattr(self, name, output)

        for i, e in enumerate(self.ensembles):
            trans = transform[:, i * function_d:(i + 1) * function_d]
            nengo.Connection(e, output, transform=trans,
                             filter=None,
                             function=function)

        return output

    @property
    def dimensions(self):
        return self.n_ensembles * self.dimensions_per_ensemble
