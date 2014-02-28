import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.spa.rules import Rules
from nengo.utils.compat import iteritems


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    rules : object or class
        The methods on this object define connection rules using the
        spa.Rules syntax
    synapse : float
        The synapse to use for the connections
    """
    def __init__(self, rules, synapse=0.01):
        super(Cortical, self).__init__()
        self.rules = Rules(rules)
        self.synapse = synapse
        self.direct = []

    def on_add(self, spa):
        Module.on_add(self, spa)

        # parse the provided class and match it up with the spa model
        self.rules.process(spa)
        for rule in self.rules.rules:
            if len(rule.matches) > 0:
                raise TypeError('Cortical rules cannot contain match() rules')

        self.create_direct(spa)
        self.create_routed(spa)

    def create_direct(self, spa):
        """Create direct fixed inputs that never change."""
        for output, transform in iteritems(self.rules.get_outputs_direct()):
            transform = np.sum(transform, axis=1)

            with self:
                node = nengo.Node(transform)
                self.direct.append(node)
            with spa:
                nengo.Connection(node, output, synapse=None)

    def create_routed(self, spa):
        """Create routing connections from one module to another."""
        for index, route in self.rules.get_outputs_route():
            target, source = route

            if hasattr(source, 'convolve'):
                raise NotImplementedError('Cortical convolution not ' +
                                          'implemented yet')
            else:
                if source.invert:
                    raise NotImplementedError('Inverting on a communication' +
                                              ' channel not supported yet')

                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)

                if hasattr(source, 'transform'):
                    trans = source.vocab.parse(source.transform)
                    t2 = trans.get_convolution_matrix()
                    transform = np.dot(transform, t2)

                with spa:
                    nengo.Connection(source.obj, target.obj,
                                     transform=transform, synapse=self.synapse)
