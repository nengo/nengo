import numpy as np

import nengo
from nengo.spa.action_objects import Symbol, Source, Convolution
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    actions : spa.Actions
        The actions to implement
    synapse : float
        The synaptic filter to use for the connections
    neurons_cconv : int
        Number of neurons per circular convolution dimension
    """
    def __init__(self, actions, synapse=0.01, neurons_cconv=200):
        super(Cortical, self).__init__()
        self.actions = actions
        self.synapse = synapse
        self.neurons_cconv = neurons_cconv
        self._bias = None

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        # parse the provided class and match it up with the spa model
        self.actions.process(spa)
        for action in self.actions.actions:
            if action.condition is not None:
                raise NotImplementedError("Cortical actions do not support "
                                          "conditional expressions: %s." %
                                          action.condition)
            for name, effects in iteritems(action.effect.effect):
                for effect in effects.expression.items:
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(name, effect.name,
                                              effect.transform.symbol,
                                              effect.inverted)
                    elif isinstance(effect, Convolution):
                        self.add_conv_effect(name, effect)
                    else:
                        raise NotImplementedError(
                            "Subexpression '%s' from action '%s' is not "
                            "supported by the cortex." % (effect, action))

    @property
    def bias(self):
        """Return a bias node; create it if needed."""
        if self._bias is None:
            with self:
                self._bias = nengo.Node([1])
        return self._bias

    def add_direct_effect(self, target_name, value):
        """Make a fixed constant input to a module.

        Parameters
        ----------
        target_name : string
            The name of the module input to use
        value : string
            A semantic pointer to be sent to the module input
        """
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(value).v]).T

        with self.spa:
            nengo.Connection(self.bias, sink, transform=transform,
                             synapse=self.synapse)

    def add_route_effect(self, target_name, source_name, transform, inverted):
        """Connect a module output to a module input

        Parameters
        ----------
        target_name : string
            The name of the module input to affect
        source_name : string
            The name of the module output to read from.  If this output uses
            a different Vocabulary than the target, a linear transform
            will be applied to convert from one to the other.
        transform : string
            A semantic pointer to convolve with the source value before
            sending it into the target.  This transform takes
            place in the source Vocabulary.
        """
        target, target_vocab = self.spa.get_module_input(target_name)
        source, source_vocab = self.spa.get_module_output(source_name)

        t = source_vocab.parse(transform).get_convolution_matrix()
        if inverted:
            D = source_vocab.dimensions
            t = np.dot(t, np.eye(D)[-np.arange(D)])

        if target_vocab is not source_vocab:
            t = np.dot(source_vocab.transform_to(target_vocab), t)

        with self.spa:
            nengo.Connection(source, target, transform=t, synapse=self.synapse)

    def add_conv_effect(self, target_name, effect):
        source1 = effect.source1
        source2 = effect.source2

        target, target_vocab = self.spa.get_module_input(target_name)
        s1_output, s1_vocab = self.spa.get_module_output(source1.name)
        s2_output, s2_vocab = self.spa.get_module_output(source2.name)

        with self:
            channel = nengo.networks.CircularConvolution(
                self.neurons_cconv, s1_vocab.dimensions,
                invert_a=False,
                invert_b=False,
                label='cconv_%s' % str(effect))

        with self.spa:
            # compute the requested transform
            t = s1_vocab.parse(str(effect.transform)).get_convolution_matrix()
            # handle conversion between different Vocabularies
            if target_vocab is not s1_vocab:
                t = np.dot(s1_vocab.transform_to(target_vocab), t)

            nengo.Connection(channel.output, target, transform=t,
                             synapse=self.synapse)

            t1 = s1_vocab.parse(
                source1.transform.symbol).get_convolution_matrix()
            if source1.inverted:
                D = s1_vocab.dimensions
                t1 = np.dot(t1, np.eye(D)[-np.arange(D)])

            nengo.Connection(s1_output, channel.A, transform=t1,
                             synapse=self.synapse)

            t2 = s2_vocab.parse(
                source2.transform.symbol).get_convolution_matrix()
            if source2.inverted:
                D = s2_vocab.dimensions
                t2 = np.dot(t2, np.eye(D)[-np.arange(D)])
            if s1_vocab is not s2_vocab:
                t2 = np.dot(s2_vocab.transform_to(s1_vocab), t2)
            nengo.Connection(s2_output, channel.B, transform=t2,
                             synapse=self.synapse)
