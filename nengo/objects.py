import collections
import copy
import logging
import numpy as np

import nengo
import nengo.decoders

logger = logging.getLogger(__name__)


def _in_stack(function):
    """Check whether the given function is in the call stack"""
    import inspect
    codes = [record[0].f_code for record in inspect.stack()]
    return function.__code__ in codes


class Uniform(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.uniform(low=self.low, high=self.high, size=n)


class Gaussian(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.std

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.normal(loc=self.mean, scale=self.std, size=n)


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
        r = self.__class__.__name__ + "("
        r += self.label if hasattr(self, 'label') else "id " + str(id(self))
        r += ", %dN)" if hasattr(self, 'n_neurons') else ")"
        return r

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return ObjView(self, key)

    def default_encoders(self, dimensions, rng):
        raise NotImplementedError("Neurons must provide default_encoders")

    def rates(self, x):
        raise NotImplementedError("Neurons must provide rates")

    def set_gain_bias(self, max_rates, intercepts):
        raise NotImplementedError("Neurons must provide set_gain_bias")

    def probe(self, probe):
        self.probes[probe.attr].append(probe)

        if probe.attr == 'output':
            nengo.Connection(self, probe, filter=probe.filter)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)
        return probe

    def add_to_model(self, model):
        model.objs.append(self)


class Ensemble(object):
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

    def __init__(self, neurons, dimensions, radius=1.0, encoders=None,
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

        # objects created at build time
        self._scaled_encoders = None  # encoders * neuron-gains / radius

        # add self to current context
        nengo.context.add_to_current(self)

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

    @property
    def neurons(self):
        """The neurons that make up the ensemble.

        Returns
        -------
        ~ : Neurons
        """
        return self._neurons

    @neurons.setter
    def neurons(self, _neurons):
        if isinstance(_neurons, int):
            logger.warning(("neurons should be an instance of a Neuron type, "
                            "not an int. Defaulting to LIF."))
            _neurons = nengo.LIF(_neurons)

        _neurons.dimensions = self.dimensions
        self._neurons = _neurons

    def activities(self, eval_points=None):
        """Determine the neuron firing rates at the given points.

        Parameters
        ----------
        eval_points : array_like (n_points, `self.dimensions`), optional
            The points at which to measure the firing rates
            (``None`` uses `self.eval_points`).

        Returns
        -------
        activities : array (n_points, `self.n_neurons`)
            Firing rates (in Hz) for each neuron at each point.
        """
        if eval_points is None:
            eval_points = self.eval_points

        return self.neurons.rates(
            np.dot(eval_points, self.encoders.T / self.radius))

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

    def add_to_model(self, model):
        model.objs.append(self)


class Node(object):
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
        if output is not None and not isinstance(output, collections.Callable):
            output = np.asarray(output)
        self.output = output
        self.label = label
        self._size_in = size_in

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

        self._size_out = size_out

        # Set up probes
        self.probes = {'output': []}

        # add self to current context
        nengo.context.add_to_current(self)

    def __str__(self):
        return "Node: " + self.label

    def __getitem__(self, key):
        return ObjView(self, key)

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.items():
                if k == 'output':
                    try:
                        rval.__dict__[k] = copy.deepcopy(v, memo)
                    except TypeError:
                        # XXX some callable things aren't serializable
                        #     is it worth crashing over?
                        #     .... we're going to guess not.
                        rval.__dict__[k] = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval

    @property
    def size_in(self):
        return self._size_in

    @property
    def size_out(self):
        return self._size_out

    def probe(self, probe, **kwargs):
        """TODO"""
        if probe.attr == 'output':
            Connection(self, probe, filter=probe.filter, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe

    def add_to_model(self, model):
        model.objs.append(self)


class Connection(object):
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

    _decoders = None
    _eval_points = None
    _function = (None, 0)  # (handle, n_outputs)
    _transform = None

    def __init__(self, pre, post,
                 filter=0.005, transform=1.0, modulatory=False, **kwargs):
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
        self.transform = transform
        self.modulatory = modulatory

        if isinstance(self.pre, Ensemble):
            self.decoder_solver = kwargs.pop(
                'decoder_solver', nengo.decoders.lstsq_L2)
            self.eval_points = kwargs.pop('eval_points', None)
            self.function = kwargs.pop('function', None)
        elif not isinstance(self.pre, (Neurons, Node)):
            raise ValueError("Objects of type '%s' cannot serve as 'pre'" %
                             (self.pre.__class__.__name__))

        if not isinstance(self.post, (Ensemble, Neurons, Node, Probe)):
            raise ValueError("Objects of type '%s' cannot serve as 'post'" %
                             (self.post.__class__.__name__))

        # check that we've used all user-provided arguments
        if len(kwargs) > 0:
            raise TypeError("__init__() got an unexpected keyword argument '"
                            + next(iter(kwargs)) + "'")

        # check that shapes match up
        self._check_shapes(check_in_init=True)

        # add self to current context
        nengo.context.add_to_current(self)

    def _check_pre_ensemble(self, prop_name):
        if not isinstance(self.pre, Ensemble):
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
        if not check_in_init and _in_stack(self.__init__):
            return  # skip automatic checks if we're in the init function
        self._check_transform(self.transform_full,
                              self._required_transform_shape())

    def _required_transform_shape(self):
        if isinstance(self.pre, Ensemble) and self.function is not None:
            in_dims = self._function[1]
        elif isinstance(self.pre, Ensemble):
            in_dims = self.pre.dimensions
        elif isinstance(self.pre, Neurons):
            in_dims = self.pre.n_neurons
        else:  # Node
            in_dims = self.pre.size_out

        if isinstance(self.post, Ensemble):
            out_dims = self.post.dimensions
        elif isinstance(self.post, Neurons):
            out_dims = self.post.n_neurons
        elif isinstance(self.post, Probe):
            out_dims = in_dims
        else:  # Node
            out_dims = self.post.size_in

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
        label = "%s>%s" % (self.pre.label, self.post.label)
        if self.function is not None:
            return "%s:%s" % (label, self.function.__name__)
        return label

    @property
    def dimensions(self):
        return self._required_transform_shape()[1]

    @property
    def eval_points(self):
        if self._eval_points is None:
            return self.pre.eval_points
        return self._eval_points

    @eval_points.setter
    def eval_points(self, _eval_points):
        if _eval_points is not None:
            _eval_points = np.asarray(_eval_points)
            if _eval_points.ndim == 1:
                _eval_points.shape = 1, _eval_points.shape[0]
        self._eval_points = _eval_points

    @property
    def function(self):
        return self._function[0]

    @function.setter
    def function(self, _function):
        if _function is not None:
            self._check_pre_ensemble('function')
            x = (self._eval_points[0] if self._eval_points is not None else
                 np.zeros(self.pre.dimensions))
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

    def add_to_model(self, model):
        model.connections.append(self)


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
    dimensions : int, optional
        Number of dimensions.
    """
    DEFAULTS = {
        Ensemble: 'decoded_output',
        Node: 'output',
    }

    def __init__(self, target, attr=None,
                 sample_every=None, filter=None, dimensions=None, **kwargs):
        self.target = target
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
        self.label = "Probe(" + target.label + "." + attr + ")"
        self.sample_every = sample_every
        self.dimensions = dimensions  # None?
        self.filter = filter
        self.kwargs = kwargs

        target.probe(self, **kwargs)

        # add self to current context
        nengo.context.add_to_current(self)

    def add_to_model(self, model):
        model.probed[id(self)] = self


class Network(object):

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop("label", "Network")
        self.objects = []
        with self:
            self.make(*args, **kwargs)

        # add self to current context
        nengo.context.add_to_current(self)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            if not isinstance(obj, nengo.Connection):
                obj.label = self.label + '.' + obj.label
            model.add(obj)

    def __enter__(self):
        nengo.context.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()


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
