import numpy as np

import nengo
from nengo.dists import Uniform
from nengo.spa.spa_ast import ConstructionContext
from nengo.spa.module import Module


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
    threshold_action : float
        Minimum value for action representation
    neurons_channel_dim : int
        Number of neurons per routing channel dimension
    synapse_channel : float
        Synaptic filter for channel inputs and outputs
    neurons_gate : int
        Number of neurons per gate
    threshold_gate : float
        Minimum value for gating neurons
    synapse_to-gate : float
        Synaptic filter for controlling a gate
    """
    def __init__(self, bg, neurons_action=50, threshold_action=0.2,
                 mutual_inhibit=1, route_inhibit=3,
                 synapse_inhibit=0.008, synapse_bg=0.008,
                 neurons_channel_dim=50,
                 synapse_channel=0.01,
                 neurons_gate=40, threshold_gate=0.3, synapse_to_gate=0.002,
                 label=None, seed=None, add_to_container=None):
        Module.__init__(self, label, seed, add_to_container)

        # bg gets also added to the top level network, thus we have to
        # circumvent the check that it gets only added once.
        self.__dict__['bg'] = bg
        self.neurons_action = neurons_action
        self.mutual_inhibit = mutual_inhibit
        self.route_inhibit = route_inhibit
        self.synapse_inhibit = synapse_inhibit
        self.threshold_action = threshold_action
        self.neurons_channel_dim = neurons_channel_dim
        self.synapse_channel = synapse_channel
        self.neurons_gate = neurons_gate
        self.threshold_gate = threshold_gate
        self.synapse_to_gate = synapse_to_gate
        self.synapse_bg = synapse_bg

        self.gates = {}     # gating ensembles per action (created as needed)
        self.channels = {}  # channels to pass transformed data between modules

        nengo.networks.Thalamus(self.bg.actions.count,
                                n_neurons_per_ensemble=self.neurons_action,
                                mutual_inhib=self.mutual_inhibit,
                                threshold=self.threshold_action,
                                net=self)
        with nengo.Network(label="routing") as self.routing:
            self.bias = nengo.Node([1], label="bias")
        self.spa = None

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        self.connect_bg(self.bg)
        self.bg.actions.construction_context = ConstructionContext(
            spa, bg=self.bg, thalamus=self)
        self.bg.actions.process()

    def construct_gate(self, index):
        """Construct a gate ensemble.

        The gate neurons have no activity when the action is selected, but are
        active when the action is not selected. This makes the gate useful for
        inhibiting ensembles that should only be active when this action is
        active.

        Parameters
        ----------
        index : int
            Index to identify the gate.

        Returns
        -------
        :class:`nengo.Ensemble`
            The constructed gate.
        """
        with self.routing:
            intercepts = Uniform(self.threshold_gate, 1)
            self.gates[index] = gate = nengo.Ensemble(
                self.neurons_gate, dimensions=1, intercepts=intercepts,
                label='gate[%d]' % index, encoders=[[1]] * self.neurons_gate)
            nengo.Connection(self.bias, gate, synapse=None)

        with self.spa:
            nengo.Connection(
                self.actions.ensembles[index], self.gates[index],
                synapse=self.synapse_to_gate, transform=-1)

        return self.gates[index]

    def construct_channel(self, target_module, target_input):
        """Construct a channel.

        Channels are an additional neural population in-between a source
        population and a target population. This allows inhibiting the channel
        without affecting the source and thus is useful in routing information.

        Parameters
        ----------
        target_module : :class:`spa.Module`
            The module that the channel will project to.
        target_vocab : :class:`spa.Vocabulary`
            The vocabulary used by the target population..

        Returns
        -------
        :class:`nengo.networks.EnsembleArray`
            The constructed channel.
        """
        if target_input[1] is not None:
            dim = target_input[1].dimensions
        else:
            dim = 1
        subdim = target_module.dim_per_ensemble
        if dim < subdim:
            subdim = dim
        elif dim % subdim != 0:
            subdim = 1

        with self.routing:
            channel = nengo.networks.EnsembleArray(
                self.neurons_channel_dim * subdim,
                dim // subdim, ens_dimensions=subdim,
                radius=np.sqrt(float(subdim) / dim), label='channel')
        with self.spa:
            nengo.Connection(
                channel.output, target_input[0], synapse=self.synapse_channel)
        return channel

    def connect_bg(self, bg):
        """Connect a basal ganglia network to this thalamus."""
        with self.spa:
            nengo.Connection(bg.output, self.input, synapse=self.synapse_bg)

    def connect_gate(self, index, channel):
        """Connect a gate to a channel for information routing.

        Parameters
        ----------
        index : int
            Index of the gate to connect.
        channel : :class:`nengo.networks.EnsembleArray`
            Channel to inhibit with the gate.
        """
        with self.spa:
            for e in channel.ensembles:
                inhibit = ([[-self.route_inhibit]] * (e.n_neurons))
                nengo.Connection(
                    self.gates[index], e.neurons, transform=inhibit,
                    synapse=self.synapse_inhibit)

    def connect_fixed(self, index, target, transform):
        """Create connection to route fixed value.

        Parameters
        ----------
        index : int
            Index of the action to connect.
        target : :class:`nengo.base.NengoObject`
            Target of the connection.
        transform : array-like
            Transform to apply to apply to the connection.
        """
        self.connect(self.actions.ensembles[index], target, transform)

    def connect(self, source, target, transform):
        """Create connection.

        The connection will use the thalamus' `synapse_channel`.
        """
        with self.spa:
            nengo.Connection(
                source, target, transform=transform,
                synapse=self.synapse_channel)
