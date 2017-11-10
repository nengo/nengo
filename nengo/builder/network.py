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
from nengo.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)
nullcontext = contextlib.contextmanager(lambda: (yield))


@Builder.register(Network)  # noqa: C901
def build_network(model, network, progress_bar=False):
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
    progress_bar : bool or `.ProgressBar` or `.ProgressUpdater`, optional \
                   (Default: False)
        Progress bar for displaying build progress.

        If True, the default progress bar will be used.
        If False, the progress bar will be disabled.
        For more control over the progress bar, pass in a `.ProgressBar`
        or `.ProgressUpdater` instance.

        Note that this will only affect top-level networks. Subnetworks
        cannot have progress bars displayed.

    Notes
    -----
    Sets ``model.params[network]`` to ``None``.
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
        max_steps = len(network.all_objects) + 1  # +1 for top level network
        progress = ProgressTracker(max_steps, progress_bar, task="Building")

        def build_callback(obj):
            if isinstance(obj, tuple(network.objects)):
                progress.step()
        model.build_callback = build_callback
    else:
        # create noop context manager
        progress = contextlib.contextmanager(lambda: (yield))()

    # Set config
    old_config = model.config
    model.config = network.config

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    # Put probes last so that they don't influence other seeds
    sorted_types = (Connection, Ensemble, Network, Node, Probe)
    assert all(tp in sorted_types for tp in network.objects)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeded[obj] = (model.seeded[network] or
                                 getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    # If this is the toplevel network, enter the decoder cache
    context = (model.decoder_cache if model.toplevel is network
               else nullcontext())
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
