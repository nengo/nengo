import logging

import numpy as np

from ..connections import ConnectionList
from .. import core
from .. import objects

logger = logging.getLogger(__name__)

class EnsembleArray(object):
    def __init__(self, name, neurons_per_ensemble, n_ensembles,
                 dimensions_per_ensemble=1, encoders=None, **kwargs):
        """
        TODO
        """
        assert n_ensembles > 0, "Number of ensembles must be positive"

        neurons, dims = neurons_per_ensemble, dimensions_per_ensemble

        encoders = np.asarray(encoders)
        enc_shape = (n_ensembles * neurons.n_neurons, dims)
        msg = "Encoder shape is %s. Should be %s" % (encoders.shape, enc_shape)
        assert encoders is None or encoders.shape == enc_shape, msg

        self.name = name
        self.input_signal = core.Signal(
            n=n_ensembles*dims, name=name+".input_signal")

        self.ensembles = [
            objects.Ensemble(name+("[%d]" % i), neurons, dims,
                             input_signal=self.input_signal[i*dims:(i+1)*dims],
                             encoders=encoders[i*dims:(i+1)*dims], **kwargs)
            for i, encoders in enumerate(encoders)]

        self.connections = []
        self.probes = []

    @property
    def n_ensembles(self):
        return len(self.ensembles)

    @property
    def dimensions_per_ensemble(self):
        return self.ensembles[0].dimensions

    @property
    def dimensions(self):
        return self.n_ensembles*self.dimensions_per_ensemble

    def connect_to(self, post, transform=1.0, **kwargs):
        if not isinstance(post, core.Signal):
            post = post.input_signal

        dims = self.dimensions_per_ensemble
        if 'function' in kwargs:
            dims = np.asarray(kwargs['function']([1] * dims)).size

        connections = []
        for i, ensemble in enumerate(self.ensembles):
            c = ensemble.connect_to(post[i*dims:(i+1)*dims], **kwargs)
            connections.append(c)
        connection = ConnectionList(connections)
        self.connections.append(connection)
        return connection

    def probe(self, to_probe='decoded_output',
              sample_every=0.001, filter=0.01, dt=0.001):
        if to_probe == 'decoded_output':
            p = probes.Probe(self.name + ".decoded_output",
                              self.dimensions, sample_every)
            c = self.connect_to(p, filter=filter)

        self.probes.append(p)
        return p, c

    def add_to_model(self, model):
        model.add(self.input_signal)
        for ensemble in self.ensembles:
            model.add(ensemble)
        for connection in self.connections:
            model.add(connection)
        for probe in self.probes:
            model.add(probe)

    def remove_from_model(self):
        raise NotImplementedError("Nope")
