import contextlib
import logging

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Builder
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.network import Network
from nengo.node import Node
from nengo.probe import Probe
from nengo.utils.progress import Progress

logger = logging.getLogger(__name__)
nullcontext = contextlib.contextmanager(lambda: (yield))


@Builder.register(Network)  # noqa: C901
def build_network(model, network, progress=None):
    """Builds a `.Network` object into a model.

    The network builder does this by mapping each high-level object to its
    associated signals and operators one-by-one, in the following order:

    1. Ensembles, nodes, neurons
    2. Subnetworks (recursively)
    3. Connections, learning rules
    4. Probes

    Before calling any of the individual objects' build functions, random
    number seeds are assigned to objects that did not have a seed explicitly
    set by the user. Whether the seed was assigned manually or automatically
    is tracked, and the decoder cache is only used when the seed is assigned
    manually.

    Parameters
    ----------
    model : Model
        The model to build into.
    network : Network
        The network to build.
    progress : Progress, optional
        Object used to track the build progress.

        Note that this will only affect top-level networks.

    Notes
    -----
    Sets ``model.params[network]`` to ``None``.
    """
    if model.toplevel is None:
        model.toplevel = network
        seed_network(network, seeds=model.seeds, seeded=model.seeded)

        if progress is not None:
            # number of sub-objects, plus 1 to account for this network
            progress.max_steps = len(network.all_objects) + 1

            def build_callback(obj):
                if isinstance(obj, tuple(network.objects)):
                    progress.step()

            model.build_callback = build_callback

    if progress is None:
        progress = Progress()  # dummy progress

    # Set config
    old_config = model.config
    model.config = network.config

    # If this is the toplevel network, enter the decoder cache
    context = model.decoder_cache if model.toplevel is network else nullcontext()
    with context, progress:
        logger.debug("Network step 1: Building ensembles and nodes")
        for obj in network.ensembles + network.nodes:
            model.build(obj)

        logger.debug("Network step 2: Building subnetworks")
        for subnetwork in network.networks:
            model.build(subnetwork)

        logger.debug("Network step 3: Building connections")
        for conn in network.connections:
            # NB: we do these in the order in which they're defined, and build
            # the learning rule in the connection builder. Because learning
            # rules are attached to connections, the connection that contains
            # the learning rule (and the learning rule) are always built
            # *before* a connection that attaches to that learning rule.
            # Therefore, we don't have to worry about connection ordering here.
            # TODO: Except perhaps if the connection being learned
            # is in a subnetwork?
            model.build(conn)

        logger.debug("Network step 4: Building probes")
        for probe in network.probes:
            model.build(probe)

        if context is model.decoder_cache:
            model.decoder_cache.shrink()

        if model.toplevel is network:
            progress.step()
            model.build_callback = None

    # Unset config
    model.config = old_config
    model.params[network] = None


def seed_network(network, seeds, seeded, base_rng=np.random):
    """Populate seeding dictionaries for all objects in a network.

    This includes all subnetworks.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    network : Network
        The network containing all objects to set seeds for.
    seeds : {object: int}
        Pre-existing map from objects to seeds for those objects.
        Will be modified in-place, but entries will not be
        overwritten if already set.
    seeded : {object: bool}
        Pre-existing map from objects to a boolean indicating whether they
        have a fixed seed either themselves or from a parent network (True),
        or whether the seed is randomly generated (False).
        Will be modified in-place, but entries will not be
        overwritten if already set.
    base_rng : np.random.RandomState
        Random number generator to use to set the seeds.
    """

    # seed this base network
    _set_seed(seeds, network, base_rng)
    _set_seeded(seeded, network)

    # seed all sub-objects
    _seed_network(network, seeds, seeded)


def _seed_network(network, seeds, seeded):
    """Recursive helper to set seeds for all child objects and subnetworks."""
    rng = np.random.RandomState(seeds[network])

    # Put probes last so that they don't influence other seeds
    sorted_types = (Connection, Ensemble, Network, Node, Probe)
    assert all(tp in sorted_types for tp in network.objects)

    # assign seeds to all child objects
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            _set_seed(seeds, obj, rng)
            _set_seeded(seeded, obj, parent=network)

    # assign seeds to subnetwork objects
    for subnetwork in network.networks:
        _seed_network(subnetwork, seeds, seeded)


def _set_seed(seeds, obj, rng):
    # Generate a seed no matter what, so that setting a seed or not on
    # one object doesn't affect the seeds of other objects.
    seed = rng.randint(npext.maxint)

    if obj in seeds:
        return  # do not overwrite an existing seed
    elif getattr(obj, "seed", None) is not None:
        seeds[obj] = obj.seed
    else:
        seeds[obj] = seed


def _set_seeded(seeded, obj, parent=None):
    # do not overwrite an existing value, since this value says how the
    # original seed was assigned (deterministically or randomly), and if we
    # re-determine this, we might be wrong (e.g. if obj.seed has changed)
    if obj not in seeded:
        seeded[obj] = (
            getattr(obj, "seed", None) is not None
            or parent is not None
            and seeded[parent]
        )
