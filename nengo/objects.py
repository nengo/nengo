import copy
import inspect
import logging
import numpy as np

from . import decoders
from . import nonlinearities

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
    neurons : Nonlinearity
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

    def __init__(self, name, neurons, dimensions, **kwargs):
        self.name = name
        self.neurons = neurons
        self.dimensions = dimensions

        if 'decoder_noise' in kwargs:
            raise NotImplementedError('decoder_noise')

        if 'noise' in kwargs or 'noise_frequency' in kwargs:
            raise NotImplementedError('noise')

        self.encoders = kwargs.get('encoders', None)
        self.eval_points = kwargs.get('eval_points', None)
        self.intercepts = kwargs.get('intercepts', Uniform(-1.0, 1.0))
        self.max_rates = kwargs.get('max_rates', Uniform(200, 400))
        self.radius = kwargs.get('radius', 1.0)
        self.seed = kwargs.get('seed', None)

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'decoded_output': [], 'spikes': [], 'voltages': []}

        # objects created at build time
        self._scaled_encoders = None  # encoders * neuron-gains / radius

    def __str__(self):
        return "Ensemble: " + self.name

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
        ~ : Nonlinearity
        """
        return self._neurons

    @neurons.setter
    def neurons(self, _neurons):
        if isinstance(_neurons, int):
            logger.warning(("neurons should be an instance of a nonlinearity, "
                            "not an int. Defaulting to LIF."))
            _neurons = nonlinearities.LIF(_neurons)

        # Give a better name if name is default
        if _neurons.name.startswith("<LIF"):
            _neurons.name = self.name + "." + _neurons.__class__.__name__
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
            #note: this assumes that self.encoders has already been
            #processed in the build function (i.e., had the radius
            #and gain mixed in)

    def connect_to(self, post, **kwargs):
        """Connect this ensemble to another object.

        Parameters
        ----------
        post : model object
            The connection's target destination.
        **kwargs : optional
            Arguments for the new DecodedConnection.

        Returns
        -------
        connection : DecodedConnection
            The new connection object.
        """

        connection = DecodedConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='decoded_output', sample_every=0.001, filter=0.01):
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
        if to_probe == 'decoded_output':
            probe = Probe(self.name + '.decoded_output', sample_every)
            self.connect_to(probe, filter=filter)
            self.probes['decoded_output'].append(probe)

        elif to_probe == 'spikes':
            probe = Probe(self.name + '.spikes', sample_every)
            connection = NonlinearityConnection(
                self.neurons, probe, filter=filter,
                transform=np.eye(self.n_neurons))
            self.connections_out.append(connection)
            if hasattr(probe, 'connections_in'):
                probe.connections_in.append(connection)
            self.probes['spikes'].append(probe)

        elif to_probe == 'voltages':
            probe = Probe(self.name + '.voltages', sample_every, self.n_neurons)
            connection = SignalConnection(
                self.neurons.voltage, probe, filter=None)
            self.connections_out.append(connection)
            if hasattr(probe, 'connections_in'):
                probe.connections_in.append(connection)
            self.probes['voltages'].append(probe)

        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % to_probe)
        return probe

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self


class PassthroughNode(object):
    def __init__(self, name, dimensions=1):
        self.name = name
        self.dimensions = dimensions

        self.connections_out = []
        self.probes = {'output': []}

    def connect_to(self, post, **kwargs):
        connection = SignalConnection(self, post, **kwargs)
        self.connections_out += [connection]

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        if filter is not None and filter > 0:
            logger.warning("Filter set on constant. Usually accidental.")

        if to_probe == 'output':
            p = Probe(self.name + ".output", sample_every)
            self.connect_to(p, filter=filter)
            self.probes['output'].append(p)
        return p


class ConstantNode(object):
    def __init__(self, name, output):
        self.name = name
        self.output = output

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'output': []}

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, _output):
        self._output = np.asarray(_output)
        if self._output.shape == ():
            self._output.shape = (1,)

    def __str__(self):
        return "Constant Node: " + self.name

    def connect_to(self, post, **kwargs):
        connection = SignalConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        """TODO"""
        if filter is not None and filter > 0:
            logger.warning("Filter set on constant. Usually accidental.")

        if to_probe == 'output':
            p = Probe(self.name + ".output", sample_every)
            self.connect_to(p, filter=filter)
            self.probes['output'].append(p)
        return p

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self


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

    def __init__(self, name, output, dimensions=1):
        assert callable(output), "Use ConstantNode for constant nodes."

        self.name = name
        self.output = output
        self.dimensions = dimensions

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'output': []}

    def __str__(self):
        return "Node: " + self.name

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

    def connect_to(self, post, **kwargs):
        """TODO"""
        connection = NonlinearityConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        """TODO"""
        if to_probe == 'output':
            p = Probe(self.name + ".output", sample_every)
            self.connect_to(p, filter=filter)
            self.probes['output'].append(p)
        return p

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self


class SignalConnection(object):
    """A SimpleConnection connects two Signals (or objects with signals)
    via a transform and a filter.

    Attributes
    ----------
    name
    filter : type
        description
    transform

    probes : type
        description

    """
    def __init__(self, pre, post, **kwargs):
        self.pre = pre
        self.post = post

        self.filter = kwargs.get('filter', 0.005)
        self.transform = kwargs.get('transform', 1.0)

        self.probes = {'signal': []}

    def __str__(self):
        return self.name + " (" + self.__class__.__name__ + ")"

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self.pre.name + ">" + self.post.name

    @property
    def transform(self):
        """TODO"""
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = np.asarray(_transform)

    def add_to_model(self, model):
        model.connections.append(self)


class NonlinearityConnection(SignalConnection):
    """A NonlinearityConnection connects a nonlinearity to a Signal
    (or objects with those) via a transform and a filter.

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
    pass

class DecodedConnection(SignalConnection):
    """A DecodedConnection connects an ensemble to a Signal
    via a set of decoders, a transform, and a filter.

    Attributes
    ----------
    name
    pre
    post

    decoders
    eval_points
    filter : type
        description
    function : type
        description
    transform

    probes : type
        description

    """
    def __init__(self, pre, post, **kwargs):
        SignalConnection.__init__(self, pre, post, **kwargs)

        self.decoders = kwargs.get('decoders', None)
        self.decoder_solver = kwargs.get('decoder_solver',
                                         decoders.least_squares)
        self.eval_points = kwargs.get('eval_points', None)
        self.function = kwargs.get('function', None)
        # self.modulatory = kwargs.get('modulatory', False)
        if 'modulatory' in kwargs:
            raise NotImplementedError('modulatory')

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
                       "%d. (shape=%s)" % (self.pre.n_neurons, _decoders.shape[0], _decoders.shape))
                raise builder.ShapeMismatch(msg)

        self._decoders = None if _decoders is None else _decoders.T

    @property
    def dimensions(self):
        if self.function is None:
            return self.pre.dimensions
        else:
            if self._eval_points is not None:
                val = self._eval_points[0]
            else:
                val = np.ones(self.pre.dimensions)
            return np.array(self.function(val)).size

    @property
    def eval_points(self):
        if self._eval_points is None:
            # OK because ensembles always build first
            return self.pre.eval_points
        return self._eval_points

    @eval_points.setter
    def eval_points(self, _eval_points):
        if _eval_points is not None:
            _eval_points = np.asarray(_eval_points)
            if len(_eval_points.shape) == 1:
                _eval_points.shape = (1, _eval_points.shape[0])
        self._eval_points = _eval_points

    @property
    def name(self):
        name = self.pre.name + ">" + self.post.name
        if self.function is not None:
            return name + ":" + self.function.__name__
        return name


class ConnectionList(object):
    """A connection made up of several other connections."""
    def __init__(self, connections, transform=1.0):
        self.connections = connections
        self.transform = transform
        self.probes = {}

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

    Attributes
    ----------
    connections_in : list
        List of incoming connections.
    sample_rate
    """
    def __init__(self, name, sample_every, dimensions=None):
        self.name = "Probe(" + name + ")"
        self.sample_every = sample_every
        self.dimensions = dimensions ##None?

        self.connections_in = []

    @property
    def sample_rate(self):
        """TODO"""
        return 1.0 / self.sample_every

    def add_to_model(self, model):
        model.signal_probes.append(self)
