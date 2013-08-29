import copy
import logging
import numpy as np

import nengo
import nengo.decoders

logger = logging.getLogger(__name__)


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
        assert self._scaled_encoders is not None, (
            "Cannot get neuron activities before ensemble has been built")
        if eval_points is None:
            eval_points = self.eval_points

        return self.neurons.rates(
            np.dot(eval_points, self._scaled_encoders.T))
            # note: this assumes that self.encoders has already been
            # processed in the build function (i.e., had the radius
            # and gain mixed in)

    def probe(self, probe):
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

        self.probes[probe.attr].append(probe)

        if probe.attr == 'decoded_output':
            Connection(self, probe, filter=probe.filter)
        elif probe.attr == 'spikes':
            Connection(self.neurons, probe, filter=probe.filter,
                       transform=np.eye(self.n_neurons))
        elif probe.attr == 'voltages':
            Connection(self.neurons.voltage, probe, filter=None)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)
        return probe

    def add_to_model(self, model):
        model.objs += [self]


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
    output : callable
        Function that transforms the Node inputs into outputs.
    dimensions : int, optional
        The number of input dimensions.

    Attributes
    ----------
    name : str
        The name of the object.
    dimensions : int
        The number of input dimensions.
    """

    def __init__(self, output=None, dimensions=0, label="Node"):
        self.output = output
        self.label = label
        self.dimensions = dimensions

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

    def probe(self, probe):
        """TODO"""
        self.probes[probe.attr].append(probe)
        if probe.attr == 'output':
            Connection(self, probe, filter=probe.filter)
        return probe

    def add_to_model(self, model):
        model.objs += [self]


class Connection(object):
    """A Connection connects two objects together.

    Attributes
    ----------
    name
    pre
    post

    filter : type
        description
    transform

    probes : type
        description

    """

    def __init__(self, pre, post,
                 filter=0.005, transform=1.0, modulatory=False, **kwargs):
        self.pre = pre
        self.post = post
        self.probes = {'signal': []}

        self.filter = filter
        self.transform = transform
        self.modulatory = modulatory
        self.learning_rule = kwargs.pop('learning_rule', None)

        if isinstance(self.pre, Ensemble):
            self.decoders = kwargs.pop('decoders', None)
            self.decoder_solver = kwargs.pop(
                'decoder_solver', nengo.decoders.lstsq_L2)
            self.eval_points = kwargs.pop('eval_points', None)
            self.function = kwargs.pop('function', None)

        if len(kwargs) > 0:
            raise TypeError("__init__() got an unexpected keyword argument '"
                            + next(iter(kwargs)) + "'")

        # add self to current context
        nengo.context.add_to_current(self)

    def __str__(self):
        return self.label + " (" + self.__class__.__name__ + ")"

    def __repr__(self):
        return str(self)

    @property
    def label(self):
        label = self.pre.label + ">" + self.post.label
        if hasattr(self, 'function') and self.function is not None:
            return label + ":" + self.function.__name__
        return label

    @property
    def learning_rule(self):
        return self._learning_rule

    @learning_rule.setter
    def learning_rule(self, _learning_rule):
        if _learning_rule is not None:
            _learning_rule.connection = self
        self._learning_rule = _learning_rule

    def add_to_model(self, model):
        model.connections.append(self)

    @property
    def decoders(self):
        return None if self._decoders is None else self._decoders.T

    @decoders.setter
    def decoders(self, _decoders):
        if _decoders is not None and self.function is not None:
            logger.warning("Setting decoders on a connection with a specified "
                           "function. May not actually compute that function.")

        if _decoders is not None:
            _decoders = np.asarray(_decoders)
            if _decoders.shape[0] != self.pre.n_neurons:
                msg = ("Decoders axis 0 must be %d; in this case it is "
                       "%d. (shape=%s)" % (self.pre.n_neurons,
                                           _decoders.shape[0],
                                           _decoders.shape))
                raise ValueError(msg)

        self._decoders = None if _decoders is None else _decoders.T

    @property
    def dimensions(self):
        if self.decoders is not None:
            return self.decoders.shape[1]

        if not hasattr(self, 'function') or self.function is None:
            return self.pre.dimensions

        if self._eval_points is not None:
            val = self._eval_points[0]
        else:
            val = np.zeros(self.pre.dimensions)
        return np.array(self.function(val)).size

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

    Attributes
    ----------
    sample_rate
    """

    def __init__(self, target, attr,
                 sample_every=0.001, filter=None, dimensions=None):
        self.target = target
        self.attr = attr
        self.label = "Probe(" + target.label + "." + attr + ")"
        self.sample_every = sample_every
        self.dimensions = dimensions  # None?
        self.filter = filter

        target.probe(self)

        # add self to current context
        nengo.context.add_to_current(self)

    @property
    def sample_rate(self):
        """TODO"""
        return 1.0 / self.sample_every

    def add_to_model(self, model):
        model.probed[(self.target, self.attr)] = self


class Network(object):

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop("label", "Network")
        self.objects = []
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
