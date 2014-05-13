"""User-facing Network, Probe, Connection, Ensemble, and Node objects."""

import collections
import logging

import numpy as np

import nengo.decoders
import nengo.utils.numpy as npext
from nengo.config import Config, Default, is_param, Parameter
from nengo.learning_rules import LearningRule
from nengo.neurons import LIF
from nengo.utils.compat import is_callable, is_iterable, with_metaclass
from nengo.utils.distributions import Uniform
from nengo.utils.inspect import checked_call

logger = logging.getLogger(__name__)


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
        inst = cls.__new__(cls)
        add_to_container = kwargs.pop(
            'add_to_container', len(Network.context) > 0)
        if add_to_container:
            cls.add(inst)
        else:
            inst._key = None
        inst.label = kwargs.pop('label', None)
        inst.seed = kwargs.pop('seed', None)
        inst._next_key = hash(inst)
        with inst:
            inst.__init__(*args, **kwargs)
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
        obj._key = network.generate_key()
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
        return config

    def generate_key(self):
        """Returns a new key for a NengoObject to be added to this Network."""
        self._next_key += 1
        return self._next_key

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, dummy):
        raise AttributeError("config cannot be overwritten. See help("
                             "nengo.Config) for help on modifying configs.")

    def __eq__(self, other):
        return hash(self) == hash(other)

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

    def __hash__(self):
        return hash((self.__class__, id(self.config), self._key, self.label))

    def __str__(self):
        return "%s: %s" % (
            self.__class__.__name__,
            self.label if self.label is not None else str(self._key))

    def __repr__(self):
        return str(self)


class NetworkMember(type):
    """A metaclass used to add instances of derived classes to networks.

    Inheriting from this class means that Network.add will be invoked after
    initializing the object, unless add_to_container=False is passed to the
    derived class constructor.
    """

    def __call__(cls, *args, **kwargs):
        """Override default __call__ behavior so that Network.add is called."""
        inst = cls.__new__(cls)
        add_to_container = kwargs.pop('add_to_container', True)
        if add_to_container:
            Network.add(inst)
        else:
            inst._key = None
        inst.__init__(*args, **kwargs)
        return inst


class NengoObject(with_metaclass(NetworkMember)):
    """A base class for Nengo objects.

    This defines some functions that the Network requires
    for correct operation. In particular, list membership
    and object comparison require each object to have a unique ID.
    """

    def __hash__(self):
        if self._key is None:
            return super(NengoObject, self).__hash__()
        return hash((self.__class__, self._key))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        if hasattr(self, 'label') and self.label is not None:
            return "%s: %s" % (self.__class__.__name__, self.label)
        elif hasattr(self, '_key'):
            return "%s: key=%d" % (self.__class__.__name__, self._key)
        else:
            return "%s: id=%d" % (self.__class__.__name__, id(self))

    def __repr__(self):
        return str(self)

    def __setattr__(self, name, val):
        if val is Default:
            val = Config.default(type(self), name)
        super(NengoObject, self).__setattr__(name, val)

    @classmethod
    def param_list(cls):
        """Returns a list of parameter names that can be set."""
        return (attr for attr in dir(cls) if is_param(getattr(cls, attr)))

    @property
    def params(self):
        """Returns a list of parameter names that can be set."""
        return self.param_list()


class Neurons(object):
    """A wrapper around Ensemble for making connections directly to neurons.

    This should only ever be used in the ``Ensemble.neurons`` property,
    as a way to signal to Connection that the connection should be made
    directly to the neurons rather than to the Ensemble's decoded value.

    Does not currently support any other view-like operations.
    """
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __getitem__(self, key):
        return ObjView(self, key)

    @property
    def label(self):
        return "%s.neurons" % self.ensemble.label

    @property
    def size_in(self):
        return self.ensemble.n_neurons

    @property
    def size_out(self):
        return self.ensemble.n_neurons


class Ensemble(NengoObject):
    """A group of neurons that collectively represent a vector.

    Parameters
    ----------
    n_neurons : int
        The number of neurons.
    dimensions : int
        The number of representational dimensions.
    radius : int, optional
        The representational radius of the ensemble.
    encoders : ndarray (`n_neurons`, `dimensions`), optional
        The encoders, used to transform from representational space
        to neuron space. Each row is a neuron's encoder, each column is a
        representational dimension.
    intercepts : Distribution or ndarray (`n_neurons`), optional
        The point along each neuron's encoder where its activity is zero. If
        e is the neuron's encoder, then the activity will be zero when
        dot(x, e) <= c, where c is the given intercept.
    max_rates : Distribution or ndarray (`n_neurons`), optional
        The activity of each neuron when dot(x, e) = 1, where e is the neuron's
        encoder.
    eval_points : ndarray (n_eval_points, `dimensions`) or int, optional
        The evaluation points used for decoder solving, spanning the interval
        (-radius, radius) in each dimension. If an int is provided, this
        sets the number of evaluation points to be drawn from a hypersphere.
        If None, then a heuristic is used to determine the number of
        evaluation points.
    neuron_type : Neurons, optional
        The model that simulates all neurons in the ensemble.
    seed : int, optional
        The seed used for random number generation.
    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    """

    radius = Parameter(default=1.0)
    encoders = Parameter(default=None)
    intercepts = Parameter(default=Uniform(-1.0, 1.0))
    max_rates = Parameter(default=Uniform(200, 400))
    eval_points = Parameter(default=None)
    seed = Parameter(default=None)
    label = Parameter(default="Ensemble")
    bias = Parameter(default=None)
    gain = Parameter(default=None)
    neuron_type = Parameter(default=LIF())
    probeable = Parameter(default=['decoded_output',
                                   'neuron_output',
                                   'spikes',
                                   'voltage'])

    def __init__(self, n_neurons, dimensions, radius=Default, encoders=Default,
                 intercepts=Default, max_rates=Default, eval_points=Default,
                 neuron_type=Default, seed=Default, label=Default):

        self.dimensions = dimensions
        if self.dimensions <= 0:
            raise ValueError(
                "Number of dimensions (%d) must be positive" % dimensions)

        self.n_neurons = n_neurons
        if self.n_neurons <= 0:
            raise ValueError(
                "Number of neurons (%d) must be positive." % n_neurons)

        self.radius = radius
        self.encoders = encoders
        self.intercepts = intercepts
        self.max_rates = max_rates
        self.label = label
        self.eval_points = eval_points
        self.neuron_type = neuron_type
        self.seed = seed
        self.probeable = Default

    def __getitem__(self, key):
        return ObjView(self, key)

    @property
    def neurons(self):
        return Neurons(self)

    @neurons.setter
    def neurons(self, dummy):
        raise AttributeError("neurons cannot be overwritten.")

    @property
    def size_in(self):
        return self.dimensions

    @property
    def size_out(self):
        return self.dimensions


class Node(NengoObject):
    """Provides arbitrary data to Nengo objects.

    Nodes can accept input, and perform arbitrary computations
    for the purpose of controlling a Nengo simulation.
    Nodes are typically not part of a brain model per se,
    but serve to summarize the assumptions being made
    about sensory data or other environment variables
    that cannot be generated by a brain model alone.

    Nodes can also be used to test models by providing specific input signals
    to parts of the model, and can simplify the input/output interface of a
    Network when used as a relay to/from its internal Ensembles
    (see networks.EnsembleArray for an example).

    Parameters
    ----------
    output : callable or array_like
        Function that transforms the Node inputs into outputs, or
        a constant output value.
    size_in : int, optional
        The number of input dimensions.
    size_out : int, optional
        The size of the output signal.
        Optional; if not specified, it will be determined based on
        the values of ``output`` and ``size_in``.
    label : str, optional
        A name for the node. Used for debugging and visualization.

    Attributes
    ----------
    label : str
        The name of the node.
    output : callable or array_like
        The given output.
    size_in : int
        The number of input dimensions.
    size_out : int
        The number of output dimensions.
    """

    output = Parameter(default=None)
    size_in = Parameter(default=0)
    size_out = Parameter(default=None)
    label = Parameter(default="Node")
    probeable = Parameter(default=['output'])

    def __init__(self, output=Default,  # noqa: C901
                 size_in=Default, size_out=Default, label=Default):
        self.output = output
        self.label = label
        self.size_in = size_in
        self.size_out = size_out
        self.probeable = Default

        if self.output is not None and not is_callable(self.output):
            self.output = npext.array(self.output, min_dims=1, copy=False)

        if self.output is not None:
            if self.size_in != 0 and not is_callable(self.output):
                raise TypeError("output must be callable if size_in != 0")
            if isinstance(self.output, np.ndarray):
                shape_out = self.output.shape
            elif self.size_out is None and is_callable(self.output):
                t, x = np.asarray(0.0), np.zeros(self.size_in)
                args = [t, x] if self.size_in > 0 else [t]
                value, invoked = checked_call(self.output, *args)
                if not invoked:
                    raise TypeError("output function '%s' must accept "
                                    "%d arguments" % (self.output, len(args)))
                shape_out = (0,) if value is None else np.asarray(value).shape
            else:
                shape_out = (self.size_out,)  # assume `size_out` is correct

            if len(shape_out) > 1:
                raise ValueError(
                    "Node output must be a vector (got array shape %s)" %
                    (shape_out,))

            size_out_new = shape_out[0] if len(shape_out) == 1 else 1

            if self.size_out is not None and self.size_out != size_out_new:
                raise ValueError(
                    "Size of Node output (%d) does not match `size_out` (%d)" %
                    (size_out_new, self.size_out))

            self.size_out = size_out_new
        else:  # output is None
            self.size_out = self.size_in

    def __getitem__(self, key):
        return ObjView(self, key)


class Connection(NengoObject):
    """Connects two objects together.

    TODO: Document slice syntax here and in the transform parameter.

    Parameters
    ----------
    pre : Ensemble or Neurons or Node
        The source Nengo object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    label : string
        A descriptive label for the connection.
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but not including `transform`.
    eval_points : (n_eval_points, pre_size) array_like or int
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    synapse : float, optional
        Post-synaptic time constant (PSTC) to use for filtering.
    transform : (post_size, pre_size) array_like, optional
        Linear transform mapping the pre output to the post input.
    solver : Solver
        Instance of a Solver class to compute decoders or weights
        (see `nengo.decoders`). If solver.weights is True, a full
        connection weight matrix is computed instead of decoders.
    function : callable, optional
        Function to compute using the pre population (pre must be Ensemble).
    modulatory : bool, optional
        Specifies whether the connection is modulatory (does not physically
        connect to post, for use by learning rules), or not (default).
    eval_points : (n_eval_points, pre_size) array_like or int, optional
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    learning_rule : LearningRule or list of LearningRule, optional
        Methods of modifying the connection weights during simulation.

    Attributes
    ----------
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but before applying the `transform`.
    function : callable
        The given function.
    function_size : int
        The output dimensionality of the given function. Defaults to 0.
    label : str
        A human-readable connection label for debugging and visualization.
        Incorporates the labels of the pre and post objects.
    learning_rule : list of LearningRule
        The given learning rules. If given a single LearningRule, this will be
        a list with a single element.
    post : Ensemble or Neurons or Node or Probe
        The given pre object.
    pre : Ensemble or Neurons or Node
        The given pre object.
    transform : (post_size, pre_size) array_like
        The given transform.
    transform_full : (post_size, pre_size) array_like
        The given transform, after padding with zeros according to the sliced
        dimensionality of pre and post
        (see the documentation for the transform parameter).
    """

    synapse = Parameter(default=0.005)
    _transform = Parameter(default=1.0)
    solver = Parameter(default=nengo.decoders.LstsqL2())
    _function = Parameter(default=(None, 0))
    modulatory = Parameter(default=False)
    eval_points = Parameter(default=None)
    probeable = Parameter(default=['signal'])

    def __init__(self, pre, post, synapse=Default, transform=1.0,
                 solver=Default,
                 function=None, modulatory=Default, eval_points=Default,
                 learning_rule=[]):
        if not isinstance(pre, ObjView):
            pre = ObjView(pre)
        if not isinstance(post, ObjView):
            post = ObjView(post)
        self._pre = pre.obj
        self._post = post.obj
        self._preslice = pre.slice
        self._postslice = post.slice
        self.probeable = Default

        self.synapse = synapse
        self.modulatory = modulatory
        self.learning_rule = learning_rule

        # don't check shapes until we've set all parameters
        self._skip_check_shapes = True

        self.solver = solver
        if isinstance(self._pre, Neurons):
            self.eval_points = None
            self._function = (None, 0)
        elif isinstance(self._pre, Node):
            self.eval_points = None
            self.function = function
        elif isinstance(self._pre, (Ensemble, Node)):
            self.eval_points = eval_points
            self.function = function
            if self.solver.weights and not isinstance(self._post, Ensemble):
                raise ValueError("Cannot specify weight solver "
                                 "when 'post' is not an Ensemble")
        else:
            raise ValueError("Objects of type '%s' cannot serve as 'pre'" %
                             self._pre.__class__.__name__)

        if not isinstance(self._post, (Ensemble, Neurons, Node, Probe)):
            raise ValueError("Objects of type '%s' cannot serve as 'post'" %
                             self._post.__class__.__name__)

        # check that shapes match up
        self._skip_check_shapes = False

        # set after `function` for correct padding
        self.transform = transform

    def _pad_transform(self, transform):
        """Pads the transform with zeros according to the pre/post slices."""
        if self._preslice == slice(None) and self._postslice == slice(None):
            # Default case when unsliced objects are passed to __init__
            return transform

        # Get the required input/output sizes for the new transform
        out_dims, in_dims = self._required_transform_shape()

        # Leverage numpy's slice syntax to determine sizes of slices
        pre_sliced_size = np.asarray(np.zeros(in_dims)[self._preslice]).size
        post_sliced_size = np.asarray(np.zeros(out_dims)[self._postslice]).size

        # Check that the given transform matches the pre/post slices sizes
        self._check_transform(transform, (post_sliced_size, pre_sliced_size))

        # Cast scalar transforms to the identity
        if transform.ndim == 0:
            # following assertion should be guaranteed by _check_transform
            assert pre_sliced_size == post_sliced_size
            transform = transform*np.eye(pre_sliced_size)

        # Create the new transform matching the pre/post dimensions
        new_transform = np.zeros((out_dims, in_dims))
        rows_transform = np.array(new_transform[self._postslice])
        rows_transform[:, self._preslice] = transform
        new_transform[self._postslice] = rows_transform
        # Note: the above is a little obscure, but we do it so that lists of
        #  indices can specify selections of rows and columns, rather than
        #  just individual items

        # Note: Calling _check_shapes after this, is (or, should be) redundant
        return new_transform

    def _check_shapes(self):
        if not self._skip_check_shapes:
            self._check_transform(self.transform_full,
                                  self._required_transform_shape())

    def _required_transform_shape(self):
        if (isinstance(self._pre, (Ensemble, Node))
                and self.function is not None):
            in_dims = self.function_size
        else:
            in_dims = self._pre.size_out

        out_dims = self._post.size_in
        return out_dims, in_dims

    def _check_transform(self, transform, required_shape):
        in_src = self._pre.__class__.__name__
        out_src = self._post.__class__.__name__
        # This is a bit of a hack as probe sizes aren't known until build time.
        if out_src == "Probe":
            return
        out_dims, in_dims = required_shape
        if transform.ndim == 0:
            # check input dimensionality matches output dimensionality
            if in_dims != out_dims:
                raise ValueError("%s output size (%d) not equal to "
                                 "%s input size (%d)" %
                                 (in_src, in_dims, out_src, out_dims))
        else:
            # check input dimensionality matches transform
            if in_dims != transform.shape[1]:
                raise ValueError("%s output size (%d) not equal to "
                                 "transform input size (%d)" %
                                 (in_src, in_dims, transform.shape[1]))

            # check output dimensionality matches transform
            if out_dims != transform.shape[0]:
                raise ValueError("Transform output size (%d) not equal to "
                                 "%s input size (%d)" %
                                 (transform.shape[0], out_src, out_dims))

    @property
    def label(self):
        label = "%s->%s" % (self._pre.label, self._post.label)
        if self.function is not None:
            return "%s:%s" % (label, self.function.__name__)
        return label

    @property
    def dimensions(self):
        return self._required_transform_shape()[1]

    @property
    def function(self):
        return self._function[0]

    @property
    def function_size(self):
        return self._function[1]

    @function.setter
    def function(self, _function):
        if _function is not None:
            if not isinstance(self._pre, (Node, Ensemble)):
                raise ValueError("'function' can only be set if 'pre' "
                                 "is an Ensemble or Node")
            if not is_callable(_function):
                raise TypeError("function '%s' must be callable" % _function)
            x = (self.eval_points[0] if is_iterable(self.eval_points) else
                 np.zeros(self._pre.size_out))
            value, invoked = checked_call(_function, x)
            if not invoked:
                raise TypeError("function '%s' must accept a single "
                                "np.array argument" % _function)
            size_out = np.asarray(value).size
        else:
            size_out = 0

        self._function = (_function, size_out)
        self._check_shapes()

    @property
    def pre(self):
        return self._pre

    @property
    def post(self):
        return self._post

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = _transform
        self.transform_full = self._pad_transform(np.asarray(_transform))
        self._check_shapes()

    @property
    def learning_rule(self):
        return self._learning_rule

    @learning_rule.setter
    def learning_rule(self, _learning_rule):
        try:
            # This is done to convert generators to lists, and to copy the list
            _learning_rule = list(_learning_rule)
        except TypeError:
            # Not given an iterable
            _learning_rule = [_learning_rule]
        for lr in _learning_rule:
            if not isinstance(lr, LearningRule):
                raise ValueError("Argument '%s' is not a learning rule." % lr)
            if type(self.pre).__name__ not in lr.modifies:
                raise ValueError("Learning rule '%s' cannot be applied to "
                                 "connection with pre of type '%s'"
                                 % (lr, type(self.pre).__name__))

        self._learning_rule = _learning_rule


class Probe(NengoObject):
    """A probe is an object that receives data from the simulation.

    This is to be used in any situation where you wish to gather simulation
    data (spike data, represented values, neuron voltages, etc.) for analysis.

    Probes cannot directly affect the simulation.

    TODO: Example usage for each object.

    Parameters
    ----------
    target : Ensemble, Node, Connection
        The Nengo object to connect to the probe.
    attr : str, optional
        The quantity to probe. Refer to the target's ``probeable`` list for
        details. Defaults to the first element in the list.
    sample_every : float, optional
        Sampling period in seconds.
    conn_args : dict, optional
        Optional keyword arguments to pass to the Connection created for this
        probe. For example, passing ``synapse=pstc`` will filter the data.
    """

    def __init__(self, target, attr=None, sample_every=None, **conn_args):
        if not hasattr(target, 'probeable') or len(target.probeable) == 0:
            raise TypeError(
                "Type '%s' is not probeable" % target.__class__.__name__)

        conn_args.setdefault('synapse', None)

        # We'll use the first in the list as default
        self.attr = attr if attr is not None else target.probeable[0]

        if self.attr not in target.probeable:
            raise ValueError(
                "'%s' is not probeable for '%s'" % (self.attr, target))

        self.target = target
        self.label = "Probe(%s.%s)" % (target.label, self.attr)
        self.sample_every = sample_every
        self.conn_args = conn_args

    @property
    def size_in(self):
        return self.target.size_out


class ObjView(object):
    """Container for a slice with respect to some object.

    This is used by the __getitem__ of Neurons, Node, and Ensemble, in order
    to pass slices of those objects to Connect. This is a notational
    convenience for creating transforms. See Connect for details.

    Does not currently support any other view-like operations.
    """

    def __init__(self, obj, key=slice(None)):
        self.obj = obj
        if isinstance(key, int):
            # single slices of the form [i] should be cast into
            # slice objects for convenience
            if key == -1:
                # special case because slice(-1, 0) gives the empty list
                key = slice(key, None)
            else:
                key = slice(key, key+1)
        self.slice = key
