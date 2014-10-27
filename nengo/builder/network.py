import logging

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.network import Network
from nengo.utils.compat import is_iterable, itervalues

logger = logging.getLogger(__name__)


@Builder.register(Network)  # noqa: C901
def build_network(network, model):
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
        model.sig['common'][0] = Signal(0.0, name='Common: Zero')
        model.sig['common'][1] = Signal(1.0, name='Common: One')
        model.seeds[network] = get_seed(network, np.random)

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    sorted_types = sorted(network.objects, key=lambda t: t.__name__)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeds[obj] = get_seed(obj, rng)

    logger.info("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        Builder.build(obj, model=model, config=network.config)

    logger.info("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        Builder.build(subnetwork, model=model)

    logger.info("Network step 3: Building connections")
    for conn in network.connections:
        Builder.build(conn, model=model, config=network.config)

    logger.info("Network step 4: Building learning rules")
    for conn in network.connections:
        rule = conn.learning_rule
        if is_iterable(rule):
            for r in (itervalues(rule) if isinstance(rule, dict) else rule):
                Builder.build(r, model=model, config=network.config)
        elif rule is not None:
            Builder.build(rule, model=model, config=network.config)

    logger.info("Network step 5: Building probes")
    for probe in network.probes:
        Builder.build(probe, model=model, config=network.config)

    model.params[network] = None
