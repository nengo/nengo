import nengo
from nengo.dists import Uniform
from nengo.exceptions import ObsoleteError
from nengo.params import Default, IntParam, NumberParam
from nengo.spa.spa_ast import ConstructionContext
from nengo.spa.module import Module
from nengo.spa.scalar import Scalar
from nengo.spa.state import State
from nengo.synapses import Lowpass, SynapseParam


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

    neurons_action = IntParam('neurons_action', default=50)
    threshold_action = NumberParam('threshold_action', default=0.2)
    mutual_inhibit = NumberParam('mutual_inhibit', default=1.)
    route_inhibit = NumberParam('route_inhibit', default=3.)
    synapse_inhibit = SynapseParam('synapse_inhibit', default=Lowpass(0.008))
    synapse_bg = SynapseParam('synapse_bg', default=Lowpass(0.008))
    neurons_channel_dim = IntParam('neurons_channel_dim', default=50)
    synapse_channel = SynapseParam('synapse_channel', default=Lowpass(0.01))
    neurons_gate = IntParam('neurons_gate', default=40)
    threshold_gate = NumberParam('threshold_gate', default=0.3)
    synapse_to_gate = SynapseParam('synapse_to_gate', default=Lowpass(0.002))

    def __init__(self, bg, neurons_action=Default, threshold_action=Default,
                 mutual_inhibit=Default, route_inhibit=Default,
                 synapse_inhibit=Default, synapse_bg=Default,
                 neurons_channel_dim=Default, synapse_channel=Default,
                 neurons_gate=Default, threshold_gate=Default,
                 synapse_to_gate=Default,
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

        nengo.networks.Thalamus(self.bg.action_count,
                                n_neurons_per_ensemble=self.neurons_action,
                                mutual_inhib=self.mutual_inhibit,
                                threshold=self.threshold_action,
                                net=self)
        self.connect_bg(self.bg)

        added = add_to_container is True or len(self.context) > 0
        if self.bg.actions is not None:
            if not added:
                raise ObsoleteError(
                    "Instantiating Thalamus with a BasalGanglia with actions "
                    "without adding Thalamus immediately to a network is not "
                    "supported anymore.")

            self.bg.actions.construction_context = ConstructionContext(
                self.parent_module, bg=self.bg, thalamus=self)
            self.bg.actions.process()

    def construct_gate(self, index, net=None, label=None):
        """Construct a gate ensemble.

        The gate neurons have no activity when the action is selected, but are
        active when the action is not selected. This makes the gate useful for
        inhibiting ensembles that should only be active when this action is
        active.

        Parameters
        ----------
        index : int
            Index to identify the gate.
        net : :class:`nengo.Network`, optional
            Network to which to add the channel. Defaults to ``self.spa``.
        label : str, optional
            Label for the gate.

        Returns
        -------
        :class:`nengo.Ensemble`
            The constructed gate.
        """
        if net is None:
            net = self.parent_module
        if label is None:
            label = 'gate[%d]' % index
        with net:
            intercepts = Uniform(self.threshold_gate, 1)
            self.gates[index] = gate = nengo.Ensemble(
                self.neurons_gate, dimensions=1, intercepts=intercepts,
                label=label, encoders=[[1]] * self.neurons_gate)
            if not hasattr(net, 'bias'):
                net.bias = nengo.Node([1], label="bias")
            nengo.Connection(net.bias, gate, synapse=None)

        nengo.Connection(
            self.actions.ensembles[index], self.gates[index],
            synapse=self.synapse_to_gate, transform=-1)

        return self.gates[index]

    def construct_channel(
            self, target_module, target_input, net=None, label=None):
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
        net : :class:`nengo.Network`, optional
            Network to which to add the channel. Defaults to ``self.spa``.
        label : str, optional
            Label for the channel.

        Returns
        -------
        :class:`nengo.networks.EnsembleArray`
            The constructed channel.
        """
        if net is None:
            net = self.parent_module
        if label is None:
            if target_module.label is not None:
                label = 'channel to ' + target_module.label
            else:
                label = 'channel'
        with net:
            if target_input[1] is None:
                channel = Scalar(label=label)
            else:
                vocab = target_input[1]
                channel = State(vocab.dimensions, vocab=vocab, label=label)
        nengo.Connection(
            channel.output, target_input[0], synapse=self.synapse_channel)
        return channel

    def connect_bg(self, bg):
        """Connect a basal ganglia network to this thalamus."""
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
        if isinstance(channel, Scalar):
            target = channel.scalar.neurons
        elif isinstance(channel, State):
            target = channel.state_ensembles.add_neuron_input()
        else:
            raise NotImplementedError()

        inhibit = ([[-self.route_inhibit]] * (target.size_in))
        nengo.Connection(
            self.gates[index], target, transform=inhibit,
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
        nengo.Connection(
            source, target, transform=transform, synapse=self.synapse_channel)
