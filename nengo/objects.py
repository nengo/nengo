import collections
import copy
import logging
import numpy as np

import nengo
import nengo.decoders
from nengo.nonlinearities import Neurons

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

        # add self to current context
        nengo.context.add_to_current(self)

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
        The size of the output signal

    Attributes
    ----------
    name : str
        The name of the object.
    size_in : int
        The number of input dimensions.
    """

    def __init__(self, output=None, size_in=0, size_out=None, label="Node"):
        if output is not None and not isinstance(output, collections.Callable):
            output = np.asarray(output)
        self.output = output
        self.label = label
        self._size_in = size_in

        if size_out is None:
            if isinstance(output, collections.Callable):
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
                size_out = np.asarray(result).size
            elif isinstance(output, np.ndarray):
                size_out = output.size
        self._size_out = size_out

        # Set up probes
        self.probes = {'output': []}

        # add self to current context
        nengo.context.add_to_current(self)

    def __str__(self):
        return "Node: " + self.label

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
        self._pre = pre
        self._post = post
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

    def _check_shapes(self, check_in_init=False):
        if not check_in_init and _in_stack(self.__init__):
            return  # skip automatic checks if we're in the init function

        in_dims, in_src = self._get_input_dimensions()
        out_dims, out_src = self._get_output_dimensions()

        if self.transform.ndim == 0:
            # check input dimensionality matches output dimensionality
            if (in_dims is not None and out_dims is not None
                    and in_dims != out_dims):
                raise ValueError("%s output size (%d) not equal to "
                                 "post %s (%d)" %
                                 (in_src, in_dims, out_src, out_dims))
        else:
            # check input dimensionality matches transform
            if in_dims is not None and in_dims != self.transform.shape[1]:
                raise ValueError("%s output size (%d) not equal to "
                                 "transform input size (%d)" %
                                 (in_src, in_dims, self.transform.shape[1]))

            # check output dimensionality matches transform
            if out_dims is not None and out_dims != self.transform.shape[0]:
                raise ValueError("Transform output size (%d) not equal to "
                                 "post %s (%d)" %
                                 (self.transform.shape[0], out_src, out_dims))

    def _get_input_dimensions(self):
        if isinstance(self.pre, Ensemble):
            if self.function is not None:
                dims, src = self._function[1], "Function"
            else:
                dims, src = self.pre.dimensions, "Pre population"
        elif isinstance(self.pre, Neurons):
            dims, src = self.pre.n_neurons, "Neurons"
        elif isinstance(self.pre, Node):
            dims, src = self.pre.size_out, "Node"
        return dims, src

    def _get_output_dimensions(self):
        if isinstance(self.post, Ensemble):
            dims, src = self.post.dimensions, "population dimensions"
        elif isinstance(self.post, Neurons):
            dims, src = self.post.n_neurons, "number of neurons"
        elif isinstance(self.post, Node):
            dims, src = self.post.size_in, "node input size"
        else:
            dims, src = None, str(self.post)
        return dims, src

    def __str__(self):
        return self.label + " (" + self.__class__.__name__ + ")"

    def __repr__(self):
        return str(self)

    @property
    def label(self):
        label = self.pre.label + ">" + self.post.label
        if self.function is not None:
            return label + ":" + self.function.__name__
        return label

    @property
    def dimensions(self):
        return self._get_input_dimensions()[0]

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
        self._transform = np.asarray(_transform)
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
        self.sig = None  # XXX temp, until better probes

        target.probe(self, **kwargs)

        # add self to current context
        nengo.context.add_to_current(self)

    @property
    def dt(self):
        return self.sample_every

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
