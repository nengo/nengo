import collections

from nengo.config import Config
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.probe import Probe
from nengo.utils.compat import with_metaclass


class NengoObjectContainer(type):
    """A metaclass for containers of Nengo objects.

    Currently, the only container is ``Network``.

    There are two primary reasons for this metaclass. The first is to
    automatically add networks to the current context; this is similar
    to the need for the ``NetworkMember`` metaclass. However, there
    are some differences with how this works in containers, so they are
    separate classes (that both call ``Network.add``).
    The second reason for this metaclass is to wrap the __init__ method
    within the network's context manager; i.e., there is an automatic
    ``with self`` inside a Network's (or Network subclass') __init__.
    This allows modelers to create Network subclasses that look like
    ordinary Python classes, while maintaining the nice property that
    all created objects are stored inside the network.
    """

    def __call__(cls, *args, **kwargs):
        inst = cls.__new__(cls, *args, **kwargs)
        add_to_container = kwargs.pop(
            'add_to_container', len(Network.context) > 0)
        inst.label = kwargs.pop('label', None)
        inst.seed = kwargs.pop('seed', None)
        with inst:
            inst.__init__(*args, **kwargs)
        # Do the __init__ before adding in case __init__ errors out
        if add_to_container:
            cls.add(inst)
        return inst


class Network(with_metaclass(NengoObjectContainer)):
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

    def __new__(cls, *args, **kwargs):
        inst = super(Network, cls).__new__(cls)
        inst._config = cls.default_config()
        inst.objects = {
            Ensemble: [], Node: [], Connection: [], Network: [], Probe: [],
        }
        inst.ensembles = inst.objects[Ensemble]
        inst.nodes = inst.objects[Node]
        inst.connections = inst.objects[Connection]
        inst.networks = inst.objects[Network]
        inst.probes = inst.objects[Probe]
        return inst

    context = collections.deque(maxlen=100)  # static stack of Network objects

    @classmethod
    def add(cls, obj):
        """Add the passed object to the current Network.context."""
        if len(cls.context) == 0:
            raise RuntimeError("'%s' must either be created "
                               "inside a `with network:` block, or set "
                               "add_to_container=False in the object's "
                               "constructor." % obj)
        network = cls.context[-1]
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

    @staticmethod
    def default_config():
        config = Config()
        config.configures(Connection)
        config.configures(Ensemble)
        config.configures(Network)
        config.configures(Node)
        config.configures(Probe)
        return config

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
        for object_type in self.objects.keys():
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
        return "%s: %s" % (
            self.__class__.__name__,
            self.label if self.label is not None else str(id(self)))

    def __repr__(self):
        return str(self)
