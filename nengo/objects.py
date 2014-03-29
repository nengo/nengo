import collections
import logging
import os
import pickle

import numpy as np

import nengo.utils.numpy as npext
from nengo.utils.compat import (
    is_callable, is_iterable, with_metaclass)
from nengo.utils.distributions import Uniform

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

    To grouping together related objects that you do not need to reuse,
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
        inst.objects = {Ensemble: [], Node: [], Connection: [], Network: []}
        inst.ensembles = inst.objects[Ensemble]
        inst.nodes = inst.objects[Node]
        inst.connections = inst.objects[Connection]
        inst.networks = inst.objects[Network]
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

    def generate_key(self):
        """Returns a new key for a NengoObject to be added to this Network."""
        self._next_key += 1
        return self._next_key

    def save(self, fname, fmt=None):
        """Save this model to a file.

        So far, Pickle is the only implemented format.
        """
        if fmt is None:
            fmt = os.path.splitext(fname)[1]

        # Default to pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            logger.info("Saved %s successfully.", fname)

    @classmethod
    def load(cls, fname, fmt=None):
        """Load a model from a file.

        So far, Pickle is the only implemented format.
        """
        if fmt is None:
            fmt = os.path.splitext(fname)[1]

        # Default to pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)

        raise IOError("Could not load %s" % fname)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __enter__(self):
        Network.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Network.context) == 0:
            raise RuntimeError("Network.context in bad state; was empty when "
                               "exiting from a 'with' block.")

        network = Network.context.pop()

        if network is not self:
            raise RuntimeError("Network.context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self, network))

    def __hash__(self):
        return hash((self._key, self.label))

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
        inst.__init__(*args, **kwargs)
        if add_to_container:
            Network.add(inst)
        else:
            inst._key = None
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
        else:
            return "%s: key=%d" % (self.__class__.__name__, self._key)

    def __repr__(self):
        return str(self)


class Ensemble(NengoObject):
    """A group of neurons that collectively represent a vector.

    Parameters
    ----------
    name : str
        An arbitrary name for the new ensemble.
    neurons : Neurons
        The neuron model. This object defines both the
        number and type of neurons.
    dimensions : int
        The number of representational dimensions
    **kwargs
        Can contain any attribute.

    Attributes
    ----------
    dimensions : int
        The number of representational dimensions.
    encoders : ndarray (`n_neurons`, `dimensions`)
        The encoders, used to transform from representational space
        to neuron space. Each row is a neuron's encoder, each column is a
        representational dimension.
    eval_points : ndarray (n_eval_points, `dimensions`) or int
        The evaluation points used for decoder solving, spanning the interval
        (-radius, radius) in each dimension. If an int is provided, this
        sets the number of evaluation points to be drawn from a hypersphere.
        If None, then a heuristic is used to determine the number of
        evaluation points.
    n_neurons
    neurons
    radius : float
        The representational radius of the ensemble.
    seed : int
        The seed used for random number generation.
    """

    def __init__(self, neurons, dimensions, radius=1.0, encoders=None,
                 intercepts=Uniform(-1.0, 1.0), max_rates=Uniform(200, 400),
                 eval_points=None, seed=None, label="Ensemble"):
        if dimensions <= 0:
            raise ValueError(
                "Number of dimensions (%d) must be positive" % dimensions)

        self.dimensions = dimensions  # Must be set before neurons
        self.neurons = neurons
        self.radius = radius
        self.encoders = encoders
        self.intercepts = intercepts
        self.max_rates = max_rates
        self.label = label
        self.eval_points = eval_points
        self.seed = seed

        # Set up probes
        self.probes = {'decoded_output': [], 'spikes': [], 'voltages': []}

    def __getitem__(self, key):
        return ObjView(self, key)

    @property
    def n_neurons(self):
        """The number of neurons in the ensemble.

        Returns
        -------
        ~ : int
        """
        return self.neurons.n_neurons

    def probe(self, probe, **kwargs):
        """Probe a signal in this ensemble.

        Parameters
        ----------
        to_probe : {'decoded_output'}, optional
            The signal to probe.
        sample_every : float, optional
            The sampling period, in seconds.
        filter : float, optional
            The low-pass filter time constant of the probe, in seconds.

        Returns
        -------
        probe : Probe
            The new Probe object.
        """
        if probe.attr == 'decoded_output':
            Connection(self, probe, filter=probe.filter, **kwargs)
        elif probe.attr == 'spikes':
            Connection(self.neurons, probe, filter=probe.filter,
                       transform=np.eye(self.n_neurons), **kwargs)
        elif probe.attr == 'voltages':
            Connection(self.neurons.voltage, probe, filter=None, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe


class Node(NengoObject):
    """Provides arbitrary data to Nengo objects.

    It can also accept input, and perform arbitrary computations
    for the purpose of controlling a Nengo simulation.
    Nodes are typically not part of a brain model per se,
    but serve to summarize the assumptions being made
    about sensory data or other environment variables
    that cannot be generated by a brain model alone.
    Nodes are also useful to test models in various situations.


    Parameters
    ----------
    name : str
        An arbitrary name for the object.
    output : callable or array_like
        Function that transforms the Node inputs into outputs, or
        a constant output value. If output is None, this Node simply acts
        as a placeholder and will be optimized out of the final model.
    size_in : int, optional
        The number of input dimensions.
    size_out : int, optional
        The size of the output signal.
        Optional; if not specified, it will be determined based on
        the values of ``output`` and ``size_in``.

    Attributes
    ----------
    name : str
        The name of the object.
    size_in : int
        The number of input dimensions.
    size_out : int
        The number of output dimensions.
    """

    def __init__(self, output=None, size_in=0, size_out=None, label="Node"):
        if output is not None and not is_callable(output):
            output = npext.array(output, min_dims=1, copy=False)
        self.output = output
        self.label = label
        self.size_in = size_in

        if output is not None:
            if isinstance(output, np.ndarray):
                shape_out = output.shape
            elif size_out is None and is_callable(output):
                t, x = np.asarray(0.0), np.zeros(size_in)
                args = [t, x] if size_in > 0 else [t]
                try:
                    result = output(*args)
                except TypeError:
                    raise TypeError(
                        "The function '%s' provided to '%s' takes %d "
                        "argument(s), where a function for this type "
                        "of node is expected to take %d argument(s)" % (
                            output.__name__, self,
                            output.__code__.co_argcount, len(args)))
                shape_out = np.asarray(result).shape
            else:  # callable and size_out is not None
                shape_out = (size_out,)  # assume `size_out` is correct

            if len(shape_out) > 1:
                raise ValueError(
                    "Node output must be a vector (got array shape %s)" %
                    (shape_out,))

            size_out_new = shape_out[0] if len(shape_out) == 1 else 1
            if size_out is not None and size_out != size_out_new:
                raise ValueError(
                    "Size of Node output (%d) does not match `size_out` (%d)" %
                    (size_out_new, size_out))

            size_out = size_out_new
        else:  # output is None
            size_out = size_in

        self.size_out = size_out

        # Set up probes
        self.probes = {'output': []}

    def __getitem__(self, key):
        return ObjView(self, key)

    def probe(self, probe, **kwargs):
        """TODO"""
        if probe.attr == 'output':
            Connection(self, probe, filter=probe.filter, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe


class Connection(NengoObject):
    """Connects two objects together.

    Attributes
    ----------
    pre : Ensemble or Neurons or Node
        The source object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    label : string
        A descriptive label for the connection.
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but not including `transform`.
    decoder_solver : callable
        Function to compute decoders (see `nengo.decoders`).
    eval_points : (n_eval_points, pre_size) array_like or int
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    filter : float
        Post-synaptic time constant (PSTC) to use for filtering.
    function : callable
        Function to compute using the pre population (pre must be Ensemble).
    probes : dict
        description TODO
    transform : (post_size, pre_size) array_like
        Linear transform mapping the pre output to the post input.
    weight_solver : callable
        Function to compute a full connection weight matrix. Similar to
        `decoder_solver`, but more general. See `nengo.decoders`.
    """

    def __init__(self, pre, post, filter=0.005, transform=1.0,
                 modulatory=False, **kwargs):
        if not isinstance(pre, ObjView):
            pre = ObjView(pre)
        if not isinstance(post, ObjView):
            post = ObjView(post)
        self._pre = pre.obj
        self._post = post.obj
        self._preslice = pre.slice
        self._postslice = post.slice
        self.probes = {'signal': []}

        self.filter = filter
        self.modulatory = modulatory

        # don't check shapes until we've set all parameters
        self._skip_check_shapes = True

        if isinstance(self._pre, Ensemble):
            if isinstance(self._post, Ensemble):
                self.weight_solver = kwargs.pop('weight_solver', None)
            else:
                self.weight_solver = None
            self.decoder_solver = kwargs.pop('decoder_solver', None)
            self.eval_points = kwargs.pop('eval_points', None)
            self.function = kwargs.pop('function', None)
        elif not isinstance(self._pre, (Neurons, Node)):
            raise ValueError("Objects of type '%s' cannot serve as 'pre'" %
                             self._pre.__class__.__name__)
        else:
            self.decoder_solver = None
            self.eval_points = None
            self._function = (None, 0)

        if not isinstance(self._post, (Ensemble, Neurons, Node, Probe)):
            raise ValueError("Objects of type '%s' cannot serve as 'post'" %
                             self._post.__class__.__name__)

        # check that we've used all user-provided arguments
        if len(kwargs) > 0:
            raise TypeError("__init__() received an unexpected keyword "
                            "argument '%s'" % next(iter(kwargs)))

        self.transform = transform  # set after `function` for correct padding

        # check that shapes match up
        self._skip_check_shapes = False
        self._check_shapes()

    def _check_pre_ensemble(self, prop_name):
        if not isinstance(self._pre, Ensemble):
            raise ValueError("'%s' can only be set if 'pre' is an Ensemble" %
                             prop_name)

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
        if isinstance(self._pre, Ensemble) and self.function is not None:
            in_dims = self._function[1]
        elif isinstance(self._pre, Ensemble):
            in_dims = self._pre.dimensions
        elif isinstance(self._pre, Neurons):
            in_dims = self._pre.n_neurons
        else:  # Node
            in_dims = self._pre.size_out

        if isinstance(self._post, Ensemble):
            out_dims = self._post.dimensions
        elif isinstance(self._post, Neurons):
            out_dims = self._post.n_neurons
        elif isinstance(self._post, Probe):
            out_dims = in_dims
        else:  # Node
            out_dims = self._post.size_in

        return (out_dims, in_dims)

    def _check_transform(self, transform, required_shape):
        in_src = self._pre.__class__.__name__
        out_src = self._post.__class__.__name__
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
        label = "%s>%s" % (self._pre.label, self._post.label)
        if self.function is not None:
            return "%s:%s" % (label, self.function.__name__)
        return label

    @property
    def dimensions(self):
        return self._required_transform_shape()[1]

    @property
    def function(self):
        return self._function[0]

    @function.setter
    def function(self, _function):
        if _function is not None:
            self._check_pre_ensemble('function')
            x = (self.eval_points[0] if is_iterable(self.eval_points) else
                 np.zeros(self._pre.dimensions))
            size = np.asarray(_function(x)).size
        else:
            size = 0

        self._function = (_function, size)
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


class Neurons(object):

    def __init__(self, n_neurons, bias=None, gain=None, label=None):
        self.n_neurons = n_neurons
        self.bias = bias
        self.gain = gain
        if label is None:
            label = "<%s%d>" % (self.__class__.__name__, id(self))
        self.label = label

        self.probes = {'output': []}

    def __str__(self):
        return "%s(%s, %dN)" % (
            self.__class__.__name__, self.label, self.n_neurons)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return ObjView(self, key)

    def rates(self, x, gain, bias):
        raise NotImplementedError("Neurons must provide rates")

    def gain_bias(self, max_rates, intercepts):
        raise NotImplementedError("Neurons must provide gain_bias")

    def probe(self, probe):
        self.probes[probe.attr].append(probe)

        if probe.attr == 'output':
            Connection(self, probe, filter=probe.filter)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)
        return probe


class Probe(object):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    Parameters
    ----------
    name : str
        An arbitrary name for the object.
    sample_every : float
        Sampling period in seconds.
    """
    DEFAULTS = {
        Ensemble: 'decoded_output',
        Node: 'output',
    }

    def __init__(self, target, attr=None, sample_every=None, filter=None,
                 **kwargs):
        if attr is None:
            try:
                attr = self.DEFAULTS[target.__class__]
            except KeyError:
                for k in self.DEFAULTS:
                    if issubclass(target.__class__, k):
                        attr = self.DEFAULTS[k]
                        break
                else:
                    raise TypeError("Type '%s' has no default probe." %
                                    target.__class__.__name__)
        self.attr = attr
        self.label = "Probe(%s.%s)" % (target.label, attr)
        self.sample_every = sample_every
        self.filter = filter

        # Probes add themselves to an object through target.probe in order to
        # be built into the model.
        target.probe(self, **kwargs)

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)


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
            key = slice(key, key+1)
        self.slice = key
