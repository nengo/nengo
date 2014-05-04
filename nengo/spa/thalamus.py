import numpy as np

import nengo
from nengo.spa.action_objects import Symbol, Source, Convolution
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


class Thalamus(Module):
    """A thalamus, implementing the effects for an associated BasalGanglia

    Parameters
    ----------
    bg : spa.BasalGanglia
        The associated basal ganglia that defines the action to implement
    neurons_action : int
        Number of neurons per action to represent selection
    inhibit : float
        Strength of inhibition between actions
    synapse_inhibit : float
        Synaptic filter to apply for inhibition between actions
    synapse_bg : float
        Synaptic filter for connection between basal ganglia and thalamus
    synapse_direct : float
        Synaptic filter for direct outputs
    threshold_action : float
        Minimum value for action representation
    neurons_channel_dim : int
        Number of neurons per routing channel dimension
    subdim_channel : int
        Number of subdimensions used in routing channel
    synapse_channel : float
        Synaptic filter for channel inputs and outputs
    neurons_cconv : int
        Number of neurons per circular convolution dimension
    neurons_gate : int
        Number of neurons per gate
    threshold_gate : float
        Minimum value for gating neurons
    synapse_to-gate : float
        Synaptic filter for controlling a gate
    """
    def __init__(self, bg, neurons_action=50, threshold_action=0.2,
                 inhibit=1, synapse_inhibit=0.008, synapse_bg=0.008,
                 synapse_direct=0.01,
                 neurons_channel_dim=50, subdim_channel=16,
                 synapse_channel=0.01,
                 neurons_cconv=200,
                 neurons_gate=40, threshold_gate=0.3, synapse_to_gate=0.002):
        super(Thalamus, self).__init__()
        self.bg = bg
        self.neurons_action = neurons_action
        self.inhibit = inhibit
        self.synapse_inhibit = synapse_inhibit
        self.synapse_direct = synapse_direct
        self.threshold_action = threshold_action
        self.neurons_channel_dim = neurons_channel_dim
        self.subdim_channel = subdim_channel
        self.synapse_channel = synapse_channel
        self.neurons_gate = neurons_gate
        self.neurons_cconv = neurons_cconv
        self.threshold_gate = threshold_gate
        self.synapse_to_gate = synapse_to_gate
        self.synapse_bg = synapse_bg

        self.gates = {}     # gating ensembles per action (created as needed)
        self.channels = {}  # channels to pass transformed data between modules

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        N = self.bg.actions.count  # number of actions

        with self:
            # An EnsembleArray to store which action is selected
            self.actions = nengo.networks.EnsembleArray(
                self.neurons_action, N,
                ens_dimensions=1,
                encoders=[[1.0]] * self.neurons_action,
                intercepts=nengo.objects.Uniform(self.threshold_action, 1),
                label='actions')

            # bias input to keep the value up in the absence of basal
            # basal ganglia inhibition
            self.bias = nengo.Node(output=[1], label='bias')
            nengo.Connection(self.bias, self.actions.input,
                             transform=np.ones((N, 1)), synapse=None)

            # mutual inhibition on the actions
            nengo.Connection(self.actions.output, self.actions.input,
                             transform=(np.eye(N)-1)*self.inhibit,
                             synapse=self.synapse_inhibit)

        with spa:
            # connect basal ganglia to thalamus
            nengo.Connection(self.bg.output, self.actions.input,
                             synapse=self.synapse_bg)

        # implement the various effects
        for i, action in enumerate(self.bg.actions.actions):
            for name, effects in iteritems(action.effect.effect):
                for effect in effects.expression.items:
                    if isinstance(effect, (int, float)):
                        effect = Symbol('%g' % effect)
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(i, name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(i, name, effect.name,
                                              effect.transform.symbol,
                                              effect.inverted)
                    elif isinstance(effect, Convolution):
                        self.add_conv_effect(i, name, effect)
                    else:
                        raise NotImplementedError(
                            "Subexpression '%s' from action '%s' is not "
                            "supported by the Thalamus." % (effect, action))

    def add_direct_effect(self, index, target_name, value):
        """Cause an action to drive a particular module input to value.

        Parameters
        ----------
        index : int
            The action number that causes this effect
        target_name : string
            The name of the module input to connect to
        value : string
            A semantic pointer to be sent into the module when this action
            is active
        """
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(value).v]).T

        with self.spa:
            nengo.Connection(self.actions.ensembles[index],
                             sink, transform=transform,
                             synapse=self.synapse_direct)

    def get_gate(self, index):
        """Return the gate for an action

        The gate will be created if it does not already exist.  The gate
        neurons have no activity when the action is selected, but are
        active when the action is not selected.  This makes the gate useful
        for inhibiting ensembles that should only be active when this
        action is active.
        """
        if index not in self.gates:
            with self:
                intercepts = nengo.objects.Uniform(self.threshold_gate, 1)
                gate = nengo.Ensemble(self.neurons_gate,
                                      dimensions=1,
                                      intercepts=intercepts,
                                      label='gate[%d]' % index,
                                      encoders=[[1]] * self.neurons_gate)
                nengo.Connection(self.actions.ensembles[index], gate,
                                 synapse=self.synapse_to_gate, transform=-1)
                nengo.Connection(self.bias, gate, synapse=None)
                self.gates[index] = gate
        return self.gates[index]

    def add_route_effect(self,
                         index, target_name, source_name, transform, inverted):
        """Set an action to send source to target with the given transform

        Parameters
        ----------
        index : int
            The action number that will cause this effect
        target_name : string
            The name of the module input to affect
        source_name : string
            The name of the module output to read from.  If this output uses
            a different Vocabulary than the target, a linear transform
            will be applied to convert from one to the other.
        transform : string
            A semantic point to convolve with the source value before
            sending it into the target.  This transform takes
            place in the source Vocabulary.
        inverted : bool
            Whether to perform inverse convolution on the source.
        """
        with self:
            gate = self.get_gate(index)

            target, target_vocab = self.spa.get_module_input(target_name)
            source, source_vocab = self.spa.get_module_output(source_name)

            # build a communication channel between the source and target
            dim = target_vocab.dimensions
            subdim = self.subdim_channel
            assert dim % subdim == 0  # TODO: check this somewhere

            channel = nengo.networks.EnsembleArray(
                self.neurons_channel_dim * subdim,
                dim // subdim,
                ens_dimensions=subdim,
                label='channel_%d_%s' % (index, target_name))

            # inhibit the channel when the action is not chosen
            inhibit = [[-1]] * (self.neurons_channel_dim * subdim)
            for e in channel.ensembles:
                nengo.Connection(gate, e.neurons, transform=inhibit,
                                 synapse=self.synapse_inhibit)

        with self.spa:
            # compute the requested transform
            t = source_vocab.parse(transform).get_convolution_matrix()
            if inverted:
                D = source_vocab.dimensions
                t = np.dot(t, np.eye(D)[-np.arange(D)])
            # handle conversion between different Vocabularies
            if target_vocab is not source_vocab:
                t = np.dot(source_vocab.transform_to(target_vocab), t)

            # connect source to target
            nengo.Connection(source, channel.input, transform=t,
                             synapse=self.synapse_channel)
            nengo.Connection(channel.output, target,
                             synapse=self.synapse_channel)

    def add_conv_effect(self, index, target_name, effect):
        source1 = effect.source1
        source2 = effect.source2

        gate = self.get_gate(index)

        target, target_vocab = self.spa.get_module_input(target_name)
        s1_output, s1_vocab = self.spa.get_module_output(source1.name)
        s2_output, s2_vocab = self.spa.get_module_output(source2.name)

        with self:
            channel = nengo.networks.CircularConvolution(
                self.neurons_cconv, s1_vocab.dimensions,
                invert_a=False,
                invert_b=False,
                label='cconv_%d_%s' % (index, str(effect)))

            # inhibit the channel when the action is not chosen
            inhibit = [[-1]] * (self.neurons_cconv)
            for e in channel.product.ensembles:
                nengo.Connection(gate, e.neurons, transform=inhibit,
                                 synapse=self.synapse_inhibit)

        with self.spa:
            # compute the requested transform
            t = s1_vocab.parse(str(effect.transform)).get_convolution_matrix()
            # handle conversion between different Vocabularies
            if target_vocab is not s1_vocab:
                t = np.dot(s1_vocab.transform_to(target_vocab), t)

            nengo.Connection(channel.output, target, transform=t,
                             synapse=self.synapse_channel)

            t1 = s1_vocab.parse(
                source1.transform.symbol).get_convolution_matrix()
            if source1.inverted:
                D = s1_vocab.dimensions
                t1 = np.dot(t1, np.eye(D)[-np.arange(D)])

            nengo.Connection(s1_output, channel.A, transform=t1,
                             synapse=self.synapse_channel)

            t2 = s2_vocab.parse(
                source2.transform.symbol).get_convolution_matrix()
            if source2.inverted:
                D = s2_vocab.dimensions
                t2 = np.dot(t2, np.eye(D)[-np.arange(D)])
            if s1_vocab is not s2_vocab:
                t2 = np.dot(s2_vocab.transform_to(s1_vocab), t2)
            nengo.Connection(s2_output, channel.B, transform=t2,
                             synapse=self.synapse_channel)
