import collections

from nengo.config import Config
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.probe import Probe


class Network(object):
    """A network contains ensembles, nodes, connections, and other networks.

    A network is primarily used for grouping together related
    objects and connections for visualization purposes.
    However, you can also use networks as a nice way to reuse
    network creation code.

    To group together related objects that you do not need to reuse,
    you can create a new ``Network`` and add objects in a ``with`` block.
    For example::

        network = nengo.Network()
        with network:
            with nengo.Network(label="Vision"):
                v1 = nengo.Ensemble(nengo.LIF(100), dimensions=2)
            with nengo.Network(label="Motor"):
                sma = nengo.Ensemble(nengo.LIF(100), dimensions=2)
            nengo.Connection(v1, sma)

    To reuse a group of related objects, you can create a new subclass
    of ``Network``, and add objects in the ``__init__`` method.
    For example::

        class OcularDominance(nengo.Network):
            def __init__(self):
                self.column = nengo.Ensemble(nengo.LIF(100), dimensions=2)
        network = nengo.Network()
        with network:
            left_eye = OcularDominance()
            right_eye = OcularDominance()
            nengo.Connection(left_eye.column, right_eye.column)

    For more information and advanced usage, please see the Nengo
    documentation at http://nengo.readthedocs.org/.

    Parameters
    ----------
    label : str, optional
        Name of the model. Defaults to None.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.
    add_to_container : bool, optional
        Determines if this Network will be added to the current container.
        Defaults to true iff currently with a Network.

    Attributes
    ----------
    label : str
        Name of the Network.
    seed : int
        Random seed used by the Network.
    ensembles : list
        List of nengo.Ensemble objects in this Network.
    nodes : list
        List of nengo.Node objects in this Network.
    connections : list
        List of nengo.Connection objects in this Network.
    networks : list
        List of nengo.BaseNetwork objects in this Network.
    """

    context = collections.deque(maxlen=100)  # static stack of Network objects

    def __init__(self, label=None, seed=None, add_to_container=None):
        self.label = label
        self.seed = seed
        self._config = self.default_config()

        self.objects = {
            Ensemble: [], Node: [], Connection: [], Network: [], Probe: [],
        }
        self.ensembles = self.objects[Ensemble]
        self.nodes = self.objects[Node]
        self.connections = self.objects[Connection]
        self.networks = self.objects[Network]
        self.probes = self.objects[Probe]

        # By default, we want to add to the current context, unless there is
        # no context; i.e., we're creating a top-level network.
        if add_to_container is None:
            add_to_container = len(Network.context) > 0

        if add_to_container:
            Network.add(self)

    @staticmethod
    def default_config():
        """Constructs a Config object for setting Nengo object defaults."""
        return Config(Connection, Ensemble, Node, Probe)

    @staticmethod
    def add(obj):
        """Add the passed object to the current Network.context."""
        if len(Network.context) == 0:
            raise RuntimeError("'%s' must either be created "
                               "inside a `with network:` block, or set "
                               "add_to_container=False in the object's "
                               "constructor." % obj)
        network = Network.context[-1]
        if not isinstance(network, Network):
            raise RuntimeError("Current context is not a network: %s" %
                               network)
        for cls in obj.__class__.__mro__:
            if cls in network.objects:
                network.objects[cls].append(obj)
                break
        else:
            raise TypeError("Objects of type '%s' cannot be added to "
                            "networks." % obj.__class__.__name__)

    def _all_objects(self, object_type):
        """Returns a list of all objects of the specified type"""
        # Make a copy of this network's list
        objects = list(self.objects[object_type])
        for subnet in self.networks:
            objects.extend(subnet._all_objects(object_type))
        return objects

    @property
    def all_objects(self):
        """All objects in this network and its subnetworks"""
        objects = []
        for object_type in self.objects:
            objects.extend(self._all_objects(object_type))
        return objects

    @property
    def all_ensembles(self):
        """All ensembles in this network and its subnetworks"""
        return self._all_objects(Ensemble)

    @property
    def all_nodes(self):
        """All nodes in this network and its subnetworks"""
        return self._all_objects(Node)

    @property
    def all_networks(self):
        """All networks in this network and its subnetworks"""
        return self._all_objects(Network)

    @property
    def all_connections(self):
        """All connections in this network and its subnetworks"""
        return self._all_objects(Connection)

    @property
    def all_probes(self):
        """All probes in this network and its subnetworks"""
        return self._all_objects(Probe)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, dummy):
        raise AttributeError("config cannot be overwritten. See help("
                             "nengo.Config) for help on modifying configs.")

    def __contains__(self, obj):
        return type(obj) in self.objects and obj in self.objects[type(obj)]

    def __enter__(self):
        Network.context.append(self)
        self._config.__enter__()
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Network.context) == 0:
            raise RuntimeError("Network.context in bad state; was empty when "
                               "exiting from a 'with' block.")

        config = Config.context[-1]
        if config is not self._config:
            raise RuntimeError("Config.context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self._config, config))

        network = Network.context.pop()
        if network is not self:
            raise RuntimeError("Network.context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self, network))

        self._config.__exit__(dummy_exc_type, dummy_exc_value, dummy_tb)

    def __str__(self):
        return "<%s %s>" % (
            self.__class__.__name__,
            '"%s"' % self.label if self.label is not None else
            "(unlabeled) at 0x%x" % id(self))

    def __repr__(self):
        return "<%s %s %s>" % (
            self.__class__.__name__,
            '"%s"' % self.label if self.label is not None else "(unlabeled)",
            "at 0x%x" % id(self))
