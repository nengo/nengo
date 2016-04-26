import nengo
from nengo.exceptions import ObsoleteError
from nengo.params import Default
from nengo.spa.spa_ast import ConstructionContext
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


class Cortical(Module):
    """A SPA module for forming connections between other modules.

    Parameters
    ----------
    actions : spa.Actions
        The actions to implement
    synapse : float
        The synaptic filter to use for the connections
    """

    def __init__(
            self, actions=None, label=None, seed=None,
            add_to_container=None):
        super(Cortical, self).__init__(label, seed, add_to_container)
        self.actions = actions
        added = add_to_container is True or len(self.context) > 0
        if actions is not None:
            if not added:
                raise ObsoleteError(
                    "Instantiating Cortical with actions without adding it "
                    "immediately to a network is not supported anymore.")

            # parse the provided class and match it up with the spa model
            self.actions.construction_context = ConstructionContext(
                self.parent_module, cortical=self)
            self.actions.process()

    def connect(self, source, target, transform):
        nengo.Connection(source, target, transform=transform)
