import copy
import logging

import numpy as np

import nengo
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

    probes : type
        description

    """

    def __init__(self, neurons, n_ensembles,
                 dimensions_per_ensemble=1, label="EnsembleArray",
                 **ens_args):
        """
        TODO
        """
        assert n_ensembles > 0, "Number of ensembles must be positive"

        self.label = label

        # Make some empty ensembles for now
        self.ensembles = [objects.Ensemble(
            neurons, 1, label=self.label+("[%d]" % i), auto_add=False)
            for i in xrange(n_ensembles)]

        # Any ens_args will be set on enclosed ensembles
        for ens in self.ensembles:
            for k, v in ens_args.items():
                setattr(ens, k, v)

        # Fill in the details of those ensembles here
        self.neurons = neurons
        self.dimensions_per_ensemble = dimensions_per_ensemble

        self.probes = {'decoded_output': []}

        #add self to current context
        nengo.context.add_to_current(self)

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
            c = nengo.DecodedConnection(
                ensemble, post, auto_add=False, **kwargs)
            connections.append(c)
        connection = objects.ConnectionList(connections, transform)

        return connection

    def probe(self, probe):
        if probe.attr == 'decoded_output':
            self.connect_to(probe, filter=probe.filter)
            self.probes['decoded_output'].append(probe)
        return probe

    def add_to_model(self, model):
        model.objs += [self]
