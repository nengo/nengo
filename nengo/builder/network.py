import logging

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Builder
from nengo.network import Network

logger = logging.getLogger(__name__)


@Builder.register(Network)  # noqa: C901
def build_network(model, network):
    """Takes a Network object and returns a Model.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

    1) Ensembles, Nodes, Neurons
    2) Subnetworks (recursively)
    3) Connections
    4) Learning Rules
    5) Probes
    """
    def get_seed(obj, rng):
        # Generate a seed no matter what, so that setting a seed or not on
        # one object doesn't affect the seeds of other objects.
        seed = rng.randint(npext.maxint)
        return (seed if not hasattr(obj, 'seed') or obj.seed is None
                else obj.seed)

    if model.toplevel is None:
        model.toplevel = network
        model.seeds[network] = get_seed(network, np.random)
        model.seeded[network] = getattr(network, 'seed', None) is not None

    # Set config
    old_config = model.config
    model.config = network.config

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    sorted_types = sorted(network.objects, key=lambda t: t.__name__)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeded[obj] = (model.seeded[network] or
                                 getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    logger.debug("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    logger.debug("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    logger.debug("Network step 3: Building connections")
    for conn in network.connections:
        # NB: we do these in the order in which they're defined, and build the
        # learning rule in the connection builder. Because learning rules are
        # attached to connections, the connection that contains the learning
        # rule (and the learning rule) are always built *before* a connection
        # that attaches to that learning rule. Therefore, we don't have to
        # worry about connection ordering here.
        # TODO: Except perhaps if the connection being learned
        # is in a subnetwork?
        model.build(conn)

    logger.debug("Network step 4: Building probes")
    for probe in network.probes:
        model.build(probe)

    # Unset config
    model.config = old_config
    model.params[network] = None
