import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.spa.action_objects import Symbol, Source, Convolution
from nengo.spa.module import Module


class Thalamus(nengo.networks.Thalamus, Module):
    """A thalamus, implementing the effects for an associated basal ganglia.

    See `.spa.BasalGanglia` for information on the basal ganglia, and
    `.networks.Thalamus` for details on the underlying network.

    Parameters
    ----------
    bg : spa.BasalGanglia
        The associated basal ganglia that defines the action to implement.
    neurons_action : int, optional
        Number of neurons per action to represent the selection.
    threshold_action : float, optional
        Minimum value for action representation.
    mutual_inhibit : float, optional
        Strength of inhibition between actions.
    route_inhibit : float, optional
        Strength of inhibition for unchosen actions.
    synapse_inhibit : float, optional
        Synaptic filter to apply for inhibition between actions.
    synapse_bg : float, optional
        Synaptic filter for connection between basal ganglia and thalamus.
    synapse_direct : float, optional
        Synaptic filter for direct outputs.
    neurons_channel_dim : int, optional
        Number of neurons per routing channel dimension.
    subdim_channel : int, optional
        Number of subdimensions used in routing channel.
    synapse_channel : float, optional
        Synaptic filter for channel inputs and outputs.
    neurons_cconv : int, optional
        Number of neurons per circular convolution dimension.
    neurons_gate : int, optional
        Number of neurons per gate.
    threshold_gate : float, optional
        Minimum value for gating neurons.
    synapse_to-gate : float, optional
        Synaptic filter for controlling a gate.

    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional
        The seed used for random number generation.
    add_to_container : bool, optional
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    def __init__(
        self,
        bg,
        neurons_action=50,
        threshold_action=0.2,
        mutual_inhibit=1.0,
        route_inhibit=3.0,
        synapse_inhibit=0.008,
        synapse_bg=0.008,
        synapse_direct=0.01,
        neurons_channel_dim=50,
        subdim_channel=16,
        synapse_channel=0.01,
        neurons_cconv=200,
        neurons_gate=40,
        threshold_gate=0.3,
        synapse_to_gate=0.002,
        label=None,
        seed=None,
        add_to_container=None,
    ):

        self.bg = bg
        self.neurons_action = neurons_action
        self.mutual_inhibit = mutual_inhibit
        self.route_inhibit = route_inhibit
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
        self.spa = None

        self.gates = {}  # gating ensembles per action (created as needed)
        self.channels = {}  # channels to pass transformed data between modules

        Module.__init__(self)
        nengo.networks.Thalamus.__init__(
            self,
            dimensions=self.bg.actions.count,
            n_neurons_per_ensemble=self.neurons_action,
            mutual_inhib=self.mutual_inhibit,
            threshold=self.threshold_action,
            label=label,
            seed=seed,
            add_to_container=add_to_container,
        )

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        with spa:
            # connect basal ganglia to thalamus
            nengo.Connection(
                self.bg.output, self.actions.input, synapse=self.synapse_bg
            )

        # implement the various effects
        for i, action in enumerate(self.bg.actions.actions):
            for name, effects in action.effect.effect.items():
                for effect in effects.expression.items:
                    if isinstance(effect, (int, float)):
                        effect = Symbol("%g" % effect)
                    if isinstance(effect, Symbol):
                        self.add_direct_effect(i, name, effect.symbol)
                    elif isinstance(effect, Source):
                        self.add_route_effect(
                            i,
                            name,
                            effect.name,
                            effect.transform.symbol,
                            effect.inverted,
                        )
                    elif isinstance(effect, Convolution):
                        self.add_conv_effect(i, name, effect)
                    else:
                        raise NotImplementedError(
                            "Subexpression '%s' from action '%s' is not "
                            "supported by the Thalamus." % (effect, action)
                        )

    def add_direct_effect(self, index, target_name, value):
        """Cause an action to drive a particular module input to value.

        Parameters
        ----------
        index : int
            The action number that causes this effect.
        target_name : str
            The name of the module input to connect to.
        value : str
            A semantic pointer to be sent into the module when this action
            is active.
        """
        sink, vocab = self.spa.get_module_input(target_name)
        transform = np.array([vocab.parse(value).v]).T

        with self.spa:
            nengo.Connection(
                self.actions.ensembles[index],
                sink,
                transform=transform,
                synapse=self.synapse_direct,
            )

    def get_gate(self, index, target_name):
        """Return the gate for an action.

        The gate will be created if it does not already exist. The gate
        neurons have no activity when the action is selected, but are
        active when the action is not selected. This makes the gate useful
        for inhibiting ensembles that should only be active when this
        action is active.
        """

        target_module = self.spa.get_module(target_name)

        if index not in self.gates:
            with target_module:
                intercepts = Uniform(self.threshold_gate, 1)
                gate = nengo.Ensemble(
                    self.neurons_gate,
                    dimensions=1,
                    intercepts=intercepts,
                    label="gate[%d]" % index,
                    encoders=[[1]] * self.neurons_gate,
                )
                if not hasattr(target_module, "bias"):
                    target_module.bias = nengo.Node([1], label=target_name + " bias")
                nengo.Connection(target_module.bias, gate, synapse=None)

            with self.spa:
                nengo.Connection(
                    self.actions.ensembles[index],
                    gate,
                    synapse=self.synapse_to_gate,
                    transform=-1,
                )

            with self:
                self.gates[index] = gate
        return self.gates[index]

    def add_route_effect(self, index, target_name, source_name, transform, inverted):
        """Set an action to send source to target with the given transform.

        Parameters
        ----------
        index : int
            The action number that will cause this effect.
        target_name : str
            The name of the module input to affect.
        source_name : str
            The name of the module output to read from. If this output uses
            a different Vocabulary than the target, a linear transform
            will be applied to convert from one to the other.
        transform : str
            A semantic point to convolve with the source value before
            sending it into the target. This transform takes
            place in the source Vocabulary.
        inverted : bool
            Whether to perform inverse convolution on the source.
        """
        with self:
            gate = self.get_gate(index, target_name)

            target, target_vocab = self.spa.get_module_input(target_name)
            source, source_vocab = self.spa.get_module_output(source_name)

            target_module = self.spa.get_module(target_name)
            source_module = self.spa.get_module(source_name)

            # build a communication channel between the source and target
            dim = target_vocab.dimensions

            # Determine size of subdimension. If target module is spa.Buffer,
            # use target module's subdimension. Otherwise use default
            # (self.subdim_channel).
            # TODO: Use the ensemble properties of target module instead?
            #       - How to get these properties?
            subdim = self.subdim_channel
            if isinstance(target_module, nengo.spa.Buffer):
                subdim = target_module.state.dimensions_per_ensemble
            elif isinstance(source_module, nengo.spa.Buffer):
                subdim = source_module.state.dimensions_per_ensemble
            elif dim < subdim:
                subdim = dim
            elif dim % subdim != 0:
                subdim = 1

        with target_module:
            channel = nengo.networks.EnsembleArray(
                self.neurons_channel_dim * subdim,
                dim // subdim,
                ens_dimensions=subdim,
                radius=np.sqrt(float(subdim) / dim),
                label="channel_%d_%s" % (index, target_name),
            )

        with self.spa:
            # inhibit the channel when the action is not chosen
            inhibit = [[-self.route_inhibit]] * (self.neurons_channel_dim * subdim)
            for e in channel.ensembles:
                nengo.Connection(
                    gate, e.neurons, transform=inhibit, synapse=self.synapse_inhibit
                )

            # compute the requested transform
            t = source_vocab.parse(transform).get_convolution_matrix()
            if inverted:
                D = source_vocab.dimensions
                t = np.dot(t, np.eye(D)[-np.arange(D)])
            # handle conversion between different Vocabularies
            if target_vocab is not source_vocab:
                t = np.dot(source_vocab.transform_to(target_vocab), t)

            # connect source to target
            nengo.Connection(
                source, channel.input, transform=t, synapse=self.synapse_channel
            )
            nengo.Connection(channel.output, target, synapse=self.synapse_channel)

    def add_conv_effect(self, index, target_name, effect):
        """Set an action to combine two sources and send to target.

        Parameters
        ----------
        index : int
            The action number that will cause this effect.
        target_name : str
            The name of the module input to affect.
        effect : Convolution
            The details of the convolution to implement.
        """
        target_module = self.spa.get_module(target_name)
        cconv = nengo.spa.action_build.convolution(
            self, target_name, effect, self.neurons_cconv, self.synapse_channel
        )

        gate = self.get_gate(index, target_name)

        with target_module:
            # inhibit the convolution when the action is not chosen
            for e in cconv.product.all_ensembles:
                inhibit = -np.ones((e.n_neurons, 1)) * self.route_inhibit
                nengo.Connection(
                    gate, e.neurons, transform=inhibit, synapse=self.synapse_inhibit
                )
