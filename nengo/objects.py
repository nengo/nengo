import collections
import logging

import numpy as np
import six

import nengo
import nengo.decoders
import nengo.utils

logger = logging.getLogger(__name__)


class NengoObject(object):
    """Base class for all non-builtin model objects.

    Inheriting from this class means that self.add_to_network(network) will be
    invoked after initializing the object, where 'network' is the Network
    object associated with the current context. This callback can then be used
    to automatically add the object to the current network.

    If add_to_network=False is passed to __init__, then the current Network
    will not be determined and the callback will not be invoked.

    initialize(*args, **kwargs) can be overridden to avoid writing
    super(cls, self).__init__(*args, **kwargs) in the subclass's __init__.
    """

    def __init__(self, *args, **kwargs):
        add_to_network = kwargs.pop("add_to_network", True)
        self.initialize(*args, **kwargs)
        if add_to_network:
            if not len(nengo.context):
                raise RuntimeError("Object '%s' must either be created "
                                   "inside a `with network:` block, or set "
                                   "add_to_network=False in the object's "
                                   "constructor." % self)

            network = nengo.context[-1]
            if not isinstance(network, Network):
                raise RuntimeError("Current context is not a network: %s" %
                                   network)
            self.add_to_network(network)

    def initialize(self, *args, **kwargs):
        """Hook for subclass initialization; invoked by __init__."""
        pass

    def add_to_network(self, network):
        """Callback for nengo.context.add_to_network(self) after __init__."""
        pass


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


class Uniform(NengoObject):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.uniform(low=self.low, high=self.high, size=n)


class Gaussian(NengoObject):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.std

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.normal(loc=self.mean, scale=self.std, size=n)


class Neurons(NengoObject):

    def __init__(self, *args, **kwargs):
        # Neurons are added to Ensembles in order to be built into the model.
        # We additionally enforce add_to_network=False in order to allow
        # Neurons to be created when there is no context. This permits the user
        # to do things like pass Neurons to a Network constructor.
        kwargs["add_to_network"] = False
        super(Neurons, self).__init__(*args, **kwargs)

    def initialize(self, n_neurons, bias=None, gain=None, label=None):
        self.n_neurons = n_neurons
        self.bias = bias
        self.gain = gain
        if label is None:
            label = "<%s%d>" % (self.__class__.__name__, id(self))
        self.label = label

        self.probes = {'output': []}

    def __str__(self):
        r = self.__class__.__name__ + "("
        r += self.label if hasattr(self, 'label') else "id " + str(id(self))
        r += ", %dN)" if hasattr(self, 'n_neurons') else ")"
        return r

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return ObjView(self, key)

    def rates(self, x):
        raise NotImplementedError("Neurons must provide rates")

    def probe(self, probe):
        self.probes[probe.attr].append(probe)

        if probe.attr == 'output':
            nengo.Connection(self, probe, filter=probe.filter)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)
        return probe


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
    eval_points : ndarray (n_eval_points, `dimensions`)
        The evaluation points used for decoder solving.
    n_neurons
    neurons
    radius : float
        The representational radius of the ensemble.
    seed : int
        The seed used for random number generation.
    """

    EVAL_POINTS = 500

    def initialize(self, neurons, dimensions, radius=1.0, encoders=None,
                   intercepts=Uniform(-1.0, 1.0), max_rates=Uniform(200, 400),
                   eval_points=None, seed=None, label="Ensemble"):
        if dimensions <= 0:
            raise ValueError(
                'Number of dimensions (%d) must be positive' % dimensions)

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

    def add_to_network(self, model):
        model.ensembles.append(self)

    def __getitem__(self, key):
        return ObjView(self, key)

    def __str__(self):
        return "Ensemble: " + self.label

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

    def initialize(self, output=None, size_in=0, size_out=None, label="Node"):
        if output is not None and not isinstance(output, collections.Callable):
            output = np.asarray(output)
        self.output = output
        self.label = label
        self.size_in = size_in

        if output is not None:
            if isinstance(output, np.ndarray):
                shape_out = output.shape
            elif size_out is None and isinstance(output, collections.Callable):
                t, x = np.asarray(0.0), np.zeros(size_in)
                args = [t, x] if size_in > 0 else [t]
                try:
                    result = output(*args)
                except TypeError:
                    raise TypeError(
                        ("The function '%s' provided to '%s' takes %d "
                         "argument(s), where a function for this type "
                         "of node is expected to take %d argument(s)")
                        % (output.__name__, self,
                           output.__code__.co_argcount, len(args)))
                shape_out = np.asarray(result).shape
            else:  # callable and size_out is not None
                shape_out = (size_out,)  # assume `size_out` is correct

            if len(shape_out) > 1:
                raise ValueError(
                    "Node output must be a vector (got array shape %s)"
                    % str(shape_out))

            size_out_new = shape_out[0] if len(shape_out) == 1 else 1
            if size_out is not None and size_out != size_out_new:
                raise ValueError(
                    "Size of Node output (%d) does not match `size_out` (%d)"
                    % (size_out_new, size_out))

            size_out = size_out_new
        else:  # output is None
            size_out = size_in

        self.size_out = size_out

        # Set up probes
        self.probes = {'output': []}

    def add_to_network(self, model):
        model.nodes.append(self)

    def __str__(self):
        return "Node: " + self.label

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
    eval_points : array_like, shape (n_eval_points, pre_size)
        Points at which to evaluate `function` when computing decoders.
    filter : float
        Post-synaptic time constant (PSTC) to use for filtering.
    function : callable
        Function to compute using the pre population (pre must be Ensemble).
    probes : dict
        description TODO
    transform : array_like, shape (post_size, pre_size)
        Linear transform mapping the pre output to the post input.
    """

    def initialize(self, pre, post, filter=0.005, transform=1.0,
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

        if isinstance(self._pre, Ensemble):
            self.decoder_solver = kwargs.pop(
                'decoder_solver', nengo.decoders.lstsq_L2)
            self.eval_points = kwargs.pop('eval_points', None)
            self.function = kwargs.pop('function', None)
        elif not isinstance(self._pre, (Neurons, Node)):
            raise ValueError("Objects of type '%s' cannot serve as 'pre'" %
                             (self._pre.__class__.__name__))
        else:
            self.decoder_solver = None
            self.eval_points = None
            self._function = (None, 0)

        if not isinstance(self._post, (Ensemble, Neurons, Node, Probe)):
            raise ValueError("Objects of type '%s' cannot serve as 'post'" %
                             (self._post.__class__.__name__))

        # check that we've used all user-provided arguments
        if len(kwargs) > 0:
            raise TypeError("__init__() got an unexpected keyword argument '"
                            + next(iter(kwargs)) + "'")

        # check that shapes match up
        self.transform = transform
        self._check_shapes(check_in_init=True)

    def add_to_network(self, model):
        model.connections.append(self)

    def _check_pre_ensemble(self, prop_name):
        if not isinstance(self._pre, Ensemble):
            raise ValueError("'%s' can only be set if 'pre' is an Ensemble" %
                             (prop_name))

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
        new_transform[self._postslice, self._preslice] = transform

        # Note: Calling _check_shapes after this, is (or, should be) redundant
        return new_transform

    def _check_shapes(self, check_in_init=False):
        if not check_in_init and nengo.utils.in_stack(self.__init__):
            return  # skip automatic checks if we're in the init function
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

    def __str__(self):
        return "%s (%s)" % (self.label, self.__class__.__name__)

    def __repr__(self):
        return str(self)

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
            x = (self.eval_points[0] if self.eval_points is not None else
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


class Probe(NengoObject):
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

    def initialize(self, target, attr=None, sample_every=None, filter=None,
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
                    raise TypeError("Type " + target.__class__.__name__
                                    + " has no default probe.")
        self.attr = attr
        self.label = "Probe(%s.%s)" % (target.label, attr)
        self.sample_every = sample_every
        self.filter = filter

        # Probes add themselves to an object through target.probe in order to
        # be built into the model.
        target.probe(self, **kwargs)


class Network(NengoObject):
    """A network contains ensembles, nodes, connections, and other networks.

    # TODO: Example usage.

    Parameters
    ----------
    label : str, optional
        Name of the model. Defaults to Model.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.

    Attributes
    ----------
    label : str
        Name of the model
    seed : int
        Random seed used by the model.
    """

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop("label", "Model")

        # TODO: Only pass a seed to the simulator
        self.seed = kwargs.pop("seed", None)

        if not isinstance(self.label, six.string_types):
            raise ValueError("Label '%s' must be str or unicode." %
                             self.label)

        self.ensembles = []
        self.nodes = []
        self.connections = []
        self.networks = []

        super(Network, self).__init__(
            add_to_network=len(nengo.context) > 0, *args, **kwargs)

        with self:
            self.make(*args, **kwargs)

    def make(self, *args, **kwargs):
        """Hook for subclass network creation inside of a `with self:` block.

        This is as a convenience to obtain working space to create the
        network's objects within the context of self.
        """
        pass

    def add_to_network(self, model):
        model.networks.append(self)

    def __enter__(self):
        nengo.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        try:
            model = nengo.context.pop()
        except IndexError:
            raise RuntimeError("Network context in bad state; was empty when "
                               "exiting from a 'with' block.")
        if not model is self:
            raise RuntimeError("Network context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self, model))
