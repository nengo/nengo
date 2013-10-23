"""
old_api.py: adapter for nengo_theano and [Jython] nengo-1.4

The purpose of this emulation layer is to run scripts and tests that were
written for nengo-1.4 and theano_nengo. Authors are encouraged to use
`nengo.Model` directly instead of this file for their current work.

"""

import logging
import random

import numpy as np

from . import Model
from . import LIF
from .templates.ensemble_array import EnsembleArray


logger = logging.getLogger(__name__)


class Network(object):
    def __init__(self, *args, **kwargs):
        self.model = Model(*args, **kwargs)

    @property
    def dt(self):
        return self.model.dt

    def make_input(self, name, value):
        return self.model.make_node(name, value)

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        """Generate a network array specifically.

        This function is deprecated; use for legacy code
        or non-theano API compatibility.

        """
        return self.make(
            name=name, neurons=neurons, dimensions=dimensions,
            array_size=array_size, **kwargs)

    def make(self, name, neurons, array_size, dimensions=1, **kwargs):
        if 'neuron_type' in kwargs:
            if kwargs['neuron_type'].lower() == 'lif':
                neurons = LIF(neurons)
            else:
                raise ValueError("Only LIF supported")
            del kwargs['neuron_type']
        else:
            neurons = LIF(neurons)
        return self.model.add(EnsembleArray(
            name, neurons, array_size, dimensions, **kwargs))

    def connect(self, name1, name2, func=None, pstc=0.005, **kwargs):
        if func is not None:
            kwargs['function'] = func
        return self.model.connect(name1, name2, filter=pstc, **kwargs)

    def make_probe(self, name, dt_sample, pstc):
        return self.model.probe(name, dt_sample, pstc)

    def run(self, simtime, **kwargs):
        netcopy = Network()
        netcopy.model = self.model.run(simtime, **kwargs)
        return netcopy
