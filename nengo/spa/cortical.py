import nengo
from nengo.params import Default
from nengo.spa.spa_ast import ConstructionContext
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    actions : Actions
        The actions to implement.
    synapse : float, optional (Default: 0.01)
        The synaptic filter to use for the connections.

    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    synapse = SynapseParam('synapse', default=Lowpass(0.01))

    def __init__(
            self, actions=None, synapse=Default, label=None, seed=None,
            add_to_container=None):
        super(Cortical, self).__init__(label, seed, add_to_container)
        self.actions = actions
        self.synapse = synapse
        self.spa = None

    def on_add(self, spa):
        Module.on_add(self, spa)
        self.spa = spa

        # parse the provided class and match it up with the spa model
        if self.actions is not None:
            self.actions.construction_context = ConstructionContext(
                spa, cortical=self)
            self.actions.process()

    def connect(self, source, target, transform):
        """Create connection.

        The connection will use the cortical synapse.
        """
        with self.spa:
            nengo.Connection(
                source, target, transform=transform, synapse=self.synapse)
