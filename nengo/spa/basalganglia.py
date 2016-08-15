import nengo
from nengo.params import Default
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


class BasalGanglia(Module):
    """A basal ganglia, performing action selection on a set of given actions.

    See `.networks.BasalGanglia` for more details.

    Parameters
    ----------
    actions : Actions
        The actions to choose between.
    input_synapse : float, optional (Default: 0.002)
        The synaptic filter on all input connections.
    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
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
