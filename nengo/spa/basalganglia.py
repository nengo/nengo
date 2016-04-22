import nengo
from nengo.params import Default
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


class BasalGanglia(Module):
    """A Basal Ganglia, performing action selection on a set of given actions.

    Parameters
    ----------
    actions : spa.Actions
        The actions to choose between
    input_synapse : float
        The synaptic filter on all input connections
    """

    input_synapse = SynapseParam('input_synapse', default=Lowpass(0.002))

    def __init__(self, actions=None, action_count=None, input_synapse=Default,
                 label=None, seed=None, add_to_container=None):
        self.actions = actions
        if action_count is None:
            if actions is None:
                raise ValueError("One of actions or action_count required.")
            action_count = self.actions.count
        self.action_count = action_count
        self.input_synapse = input_synapse
        self._bias = None
        Module.__init__(self, label, seed, add_to_container)
        nengo.networks.BasalGanglia(dimensions=action_count, net=self)

    def connect_input(self, source, transform, index):
        nengo.Connection(
            source, self.input[index], transform=transform,
            synapse=self.input_synapse)
