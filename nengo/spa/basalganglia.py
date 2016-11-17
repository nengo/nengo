import nengo
from nengo.params import Default
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


class BasalGanglia(Module):
    """A basal ganglia, performing action selection on a set of given actions.

    See `.networks.BasalGanglia` for more details.

    Parameters
    ----------
    action_count : int
        Number of actions.
    input_synapse : float, optional (Default: 0.002)
        The synaptic filter on all input connections.
    kwargs
        Passed through to ``spa.Module``.
    """

    input_synapse = SynapseParam('input_synapse', default=Lowpass(0.002))

    def __init__(self, action_count, input_synapse=Default, **kwargs):
        super(BasalGanglia, self).__init__(**kwargs)

        self.action_count = action_count
        self.input_synapse = input_synapse

        nengo.networks.BasalGanglia(dimensions=action_count, net=self)

    def connect_input(self, source, transform, index):
        nengo.Connection(
            source, self.input[index], transform=transform,
            synapse=self.input_synapse)
