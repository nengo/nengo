import copy

import numpy as np

import nengo


class EnsembleArray(nengo.Network):
    def make(self, neurons, n_ensembles, dimensions_per_ensemble=1, **ens_args):
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = dimensions_per_ensemble
        self.ensembles = []
        transform = np.eye(self.dimensions)
        each_neurons = neurons.n_neurons / n_ensembles
        extra_neurons = neurons.n_neurons % n_ensembles

        with self:
            self.input = nengo.Node(dimensions=self.dimensions)

            for i in range(n_ensembles):
                ens_neurons = copy.deepcopy(neurons)
                if extra_neurons > 0:
                    ens_neurons.n_neurons = each_neurons + 1
                    extra_neurons -= 1
                else:
                    ens_neurons.n_neurons = each_neurons

                e = nengo.Ensemble(ens_neurons, dimensions_per_ensemble)
                trans = transform[i * dimensions_per_ensemble:
                                  (i + 1) * dimensions_per_ensemble, :]
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
