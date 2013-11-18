import copy
import logging

import numpy as np

from .. import objects

logger = logging.getLogger(__name__)


class EnsembleArray(object):
    """A collection of neurons that collectively represent a vector.

    Attributes
    ----------
    name
    neurons
    dimensions

    encoders
    eval_points
    intercepts
    max_rates
    radius
    seed

    connections_in : type
        description
    connections_out : type
        description
    probes : type
        description

    """

    def __init__(self, name, neurons, n_ensembles,
                 dimensions_per_ensemble=1, **ens_args):
        """
        TODO
        """
        assert n_ensembles > 0, "Number of ensembles must be positive"

        self.name = name

        # Make some empty ensembles for now
        self.ensembles = [objects.Ensemble(name+("[%d]" % i), neurons, 1)
                          for i in range(n_ensembles)]

        # Any ens_args will be set on enclosed ensembles
        for ens in self.ensembles:
            for k, v in list(ens_args.items()):
                setattr(ens, k, v)

        # Fill in the details of those ensembles here
        self.neurons = neurons
        self.dimensions_per_ensemble = dimensions_per_ensemble

        self.connections_in = []
        self.connections_out = []
        self.probes = {'decoded_output': []}

    @property
    def dimensions(self):
        return self.n_ensembles * self.dimensions_per_ensemble

    @property
    def dimensions_per_ensemble(self):
        return self._dimensions_per_ensemble

    @dimensions_per_ensemble.setter
    def dimensions_per_ensemble(self, _dimensions_per_ensemble):
        self._dimensions_per_ensemble = _dimensions_per_ensemble
        for ens in self.ensembles:
            ens.dimensions = _dimensions_per_ensemble

    @property
    def n_ensembles(self):
        return len(self.ensembles)

    @n_ensembles.setter
    def n_ensembles(self, _n_ensembles):
        raise ValueError("Cannot change number of ensembles after creation. "
                         "Please create a new EnsembleArray instead.")

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, _neurons):
        self._neurons = _neurons
        each_neurons = _neurons.n_neurons / self.n_ensembles
        extra_neurons = _neurons.n_neurons % self.n_ensembles
        for i, ens in enumerate(self.ensembles):
            # Copy and partition _neurons
            ens_neurons = copy.deepcopy(_neurons)
            if extra_neurons > 0:
                ens_neurons.n_neurons = each_neurons + 1
                extra_neurons -= 1
            else:
                ens_neurons.n_neurons = each_neurons
            ens.neurons = ens_neurons

    def connect_to(self, post, transform=1.0, **kwargs):
        connections = []
        for i, ensemble in enumerate(self.ensembles):
            c = ensemble.connect_to(post, **kwargs)
            connections.append(c)
        connection = objects.ConnectionList(connections, transform)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='decoded_output',
              sample_every=0.001, filter=0.01, dt=0.001):
        if to_probe == 'decoded_output':
            probe = objects.Probe(self.name + ".decoded_output", sample_every)
            self.connect_to(probe, filter=filter)
            self.probes['decoded_output'].append(probe)
        return probe

    def add_to_model(self, model):
        if self.name in model.objs:
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self
