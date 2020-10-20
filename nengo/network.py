import warnings
from copy import deepcopy

from nengo.config import Config
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.exceptions import (
    ConfigError,
    NetworkContextError,
    NotAddedToNetworkWarning,
    ReadonlyError,
)
from nengo.node import Node
from nengo.params import IntParam, StringParam
from nengo.probe import Probe
from nengo.utils.threading import ThreadLocalStack


class Network:
    """A network contains ensembles, nodes, connections, and other networks.

    A network is primarily used for grouping together related
    objects and connections for visualization purposes.
    However, you can also use networks as a nice way to reuse
    network creation code.

    To group together related objects that you do not need to reuse,
    you can create a new ``Network`` and add objects in a ``with`` block.
    For example:

    .. testcode::

       network = nengo.Network()
       with network:
           with nengo.Network(label="Vision"):
               v1 = nengo.Ensemble(n_neurons=100, dimensions=2)
           with nengo.Network(label="Motor"):
               sma = nengo.Ensemble(n_neurons=100, dimensions=2)
           nengo.Connection(v1, sma)

    To reuse a group of related objects, you can create a new subclass
    of ``Network``, and add objects in the ``__init__`` method.
    For example:

    .. testcode::

       class OcularDominance(nengo.Network):
           def __init__(self):
               self.column = nengo.Ensemble(n_neurons=100, dimensions=2)

       network = nengo.Network()
       with network:
           left_eye = OcularDominance()
           right_eye = OcularDominance()
           nengo.Connection(left_eye.column, right_eye.column)

    Parameters
    ----------
    label : str, optional
        Name of the network.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting the seed makes the network's build process deterministic.
    add_to_container : bool, optional
        Determines if this network will be added to the current container.
        If None, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.

    Attributes
    ----------
    connections : list
        `.Connection` instances in this network.
    ensembles : list
        `.Ensemble` instances in this network.
    label : str
        Name of this network.
    networks : list
        `.Network` instances in this network.
    nodes : list
        `.Node` instances in this network.
    probes : list
        `.Probe` instances in this network.
    seed : int
        Random seed used by this network.
    """

    context = ThreadLocalStack(maxsize=100)  # static stack of Network objects

    label = StringParam("label", optional=True, readonly=False)
    seed = IntParam("seed", optional=True, readonly=False)

    def __init__(self, label=None, seed=None, add_to_container=None):
        self.label = label
        self.seed = seed
        self._config = self.default_config()

        self._objects = {Ensemble: [], Node: [], Connection: [], Network: [], Probe: []}
        self._ensembles = self.objects[Ensemble]
        self._nodes = self.objects[Node]
        self._connections = self.objects[Connection]
        self._networks = self.objects[Network]
        self._probes = self.objects[Probe]

        # By default, we want to add to the current context, unless there is
        # no context; i.e., we're creating a top-level network.
        if add_to_container is None:
            add_to_container = len(Network.context) > 0

        if add_to_container:
            Network.add(self)

    @staticmethod
    def add(obj):
        """Add the passed object to ``Network.context``."""
        if len(Network.context) == 0:
            raise NetworkContextError(
                f"'{obj}' must either be created inside a ``with network:`` "
                "block, or set add_to_container=False in the object's "
                "constructor."
            )
        network = Network.context[-1]
        if not isinstance(network, Network):
            raise NetworkContextError(f"Current context ({network}) is not a network")
        for cls in type(obj).__mro__:
            if cls in network.objects:
                network.objects[cls].append(obj)
                break
        else:
            raise NetworkContextError(
                f"Objects of type '{type(obj).__name__}' cannot be added to networks."
            )

    @staticmethod
    def default_config():
        """Constructs a `~.Config` object for setting defaults."""
        return Config(Connection, Ensemble, Node, Probe)

    def _all_objects(self, object_type):
        """Returns a list of all objects of the specified type."""
        # Make a copy of this network's list
        objects = list(self.objects[object_type])
        for subnet in self.networks:
            objects.extend(subnet._all_objects(object_type))
        return objects

    @property
    def all_objects(self):
        """(list) All objects in this network and its subnetworks."""
        objects = []
        for object_type in self.objects:
            objects.extend(self._all_objects(object_type))
        return objects

    @property
    def all_ensembles(self):
        """(list) All ensembles in this network and its subnetworks."""
        return self._all_objects(Ensemble)

    @property
    def all_nodes(self):
        """(list) All nodes in this network and its subnetworks."""
        return self._all_objects(Node)

    @property
    def all_networks(self):
        """(list) All networks in this network and its subnetworks."""
        return self._all_objects(Network)

    @property
    def all_connections(self):
        """(list) All connections in this network and its subnetworks."""
        return self._all_objects(Connection)

    @property
    def all_probes(self):
        """(list) All probes in this network and its subnetworks."""
        return self._all_objects(Probe)

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, _):
        raise ReadonlyError(attr="objects", obj=self)

    @property
    def ensembles(self):
        return self._ensembles

    @ensembles.setter
    def ensembles(self, _):
        raise ReadonlyError(attr="ensembles", obj=self)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, _):
        raise ReadonlyError(attr="nodes", obj=self)

    @property
    def networks(self):
        return self._networks

    @networks.setter
    def networks(self, _):
        raise ReadonlyError(attr="networks", obj=self)

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, _):
        raise ReadonlyError(attr="connections", obj=self)

    @property
    def probes(self):
        return self._probes

    @probes.setter
    def probes(self, _):
        raise ReadonlyError(attr="probes", obj=self)

    @property
    def config(self):
        """(`.Config`) Configuration for this network."""
        return self._config

    @config.setter
    def config(self, _):
        raise ReadonlyError(attr="config", obj=self)

    @property
    def n_neurons(self):
        """(int) Number of neurons in this network, including subnetworks."""
        return sum(ens.n_neurons for ens in self.all_ensembles)

    def __contains__(self, obj):
        return type(obj) in self.objects and obj in self.objects[type(obj)]

    def __enter__(self):
        Network.context.append(self)
        self._config.__enter__()
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Network.context) == 0:
            raise NetworkContextError(
                "Network.context in bad state; was empty when "
                "exiting from a 'with' block."
            )

        config = Config.context[-1]
        if config is not self._config:
            raise ConfigError(
                "Config.context in bad state; was expecting "
                f"current context to be '{self._config}' but instead got "
                f"'{config}'."
            )

        network = Network.context.pop()
        if network is not self:
            raise NetworkContextError(
                "Network.context in bad state; was expecting current context "
                f"to be '{self}' but instead got '{network}'."
            )

        self._config.__exit__(dummy_exc_type, dummy_exc_value, dummy_tb)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["label"] = self.label
        state["seed"] = self.seed
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        if len(Network.context) > 0:
            warnings.warn(NotAddedToNetworkWarning(self))

    def __str__(self):
        label_txt = (
            f'"{self.label}"'
            if self.label is not None
            else f"(unlabeled) at 0x{id(self):x}"
        )
        return f"<{type(self).__name__} {label_txt}>"

    def __repr__(self):
        label_txt = f'"{self.label}"' if self.label is not None else "(unlabeled)"
        return f"<{type(self).__name__} {label_txt} at 0x{id(self):x}>"

    def copy(self, add_to_container=None):
        with warnings.catch_warnings():
            # We warn when copying since we can't change add_to_container.
            # However, we deal with it here, so we ignore the warning.
            warnings.simplefilter("ignore", category=NotAddedToNetworkWarning)
            c = deepcopy(self)
        if add_to_container is None:
            add_to_container = len(Network.context) > 0
        if add_to_container:
            Network.add(c)
        return c
