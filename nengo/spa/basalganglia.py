import nengo
from nengo.spa.module import Module


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
    def __init__(self, actions, input_synapse=0.002,
                 label=None, seed=None, add_to_container=None):
        self.actions = actions
        self.input_synapse = input_synapse
        self._bias = None
        Module.__init__(self, label, seed, add_to_container)
        nengo.networks.BasalGanglia(dimensions=self.actions.count, net=self)
        self.spa = None

    def on_add(self, spa):
        """Form the connections into the BG to compute the utilty values.

        Each action's condition variable contains the set of computations
        needed for that action's utility value, which is the input to the
        basal ganglia.
        """
        Module.on_add(self, spa)
        self.spa = spa

    def connect_input(self, source, transform, index):
        with self.spa:
            nengo.Connection(
                source, self.input[index], transform=transform,
                synapse=self.input_synapse)
