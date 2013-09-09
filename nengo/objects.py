import copy
import inspect
import logging
import numpy as np

from . import connections
from . import core
from . import decoders

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
    """A collection of neurons that collectively represent a vector.

    Attributes
    ----------
    name
    neurons
    dimensions

    encoders
    eval_points
    intercepts
    max_rates
    radius
    seed

    connections_in : type
        description
    connections_out : type
        description
    probes : type
        description

    """

    EVAL_POINTS = 500

    def __init__(self, name, neurons, dimensions, **kwargs):
        """Note: kwargs can contain any attribute; see class docstring."""
        self.name = name
        self.neurons = neurons
        self.dimensions = dimensions

        if 'decoder_noise' in kwargs:
            raise NotImplementedError('decoder_noise')

        if 'noise' in kwargs or 'noise_frequency' in kwargs:
            raise NotImplementedError('noise')

        self.encoders = kwargs.get('encoders', None)
        self.intercepts = kwargs.get('intercepts', Uniform(-1.0, 1.0))
        self.max_rates = kwargs.get('max_rates', Uniform(200, 300))
        self.radius = kwargs.get('radius', 1.0)
        self.seed = kwargs.get('seed', np.random.randint(2**31-1))

        # Order matters here
        self.eval_points = kwargs.get('eval_points', None)

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'decoded_output': []}

    def __str__(self):
        return "Ensemble: " + self.name

    @property
    def dimensions(self):
        """TODO"""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, _dimensions):
        self._dimensions = _dimensions
        self.eval_points = None  # Invalidate possibly cached eval_points

    @property
    def encoders(self):
        """TODO"""
        return self._encoders

    @encoders.setter
    def encoders(self, _encoders):
        if _encoders is not None:
            _encoders = np.asarray(_encoders)
            enc_shape = (self.neurons.n_neurons, self.dimensions)
            if _encoders.shape != enc_shape:
                msg = ("Encoder shape is %s. Should be (n_neurons, dimensions);"
                       " in this case %s." % (_encoders.shape, enc_shape))
                raise core.ShapeMismatch(msg)
        self._encoders = _encoders

    @property
    def eval_points(self):
        """TODO"""
        if self._eval_points is None:
            self._eval_points = decoders.sample_hypersphere(
                self.dimensions, Ensemble.EVAL_POINTS, self.rng) * self.radius
        return self._eval_points

    @eval_points.setter
    def eval_points(self, _eval_points):
        if _eval_points is not None:
            _eval_points = np.asarray(_eval_points)
            if len(_eval_points.shape) == 1:
                _eval_points.shape = (1, _eval_points.shape[0])
        self._eval_points = _eval_points

    @property
    def n_neurons(self):
        """TODO"""
        return self.neurons.n_neurons

    @property
    def neurons(self):
        """TODO"""
        return self._neurons


    @neurons.setter
    def neurons(self, _neurons):
        if isinstance(_neurons, int):
            logger.warning(("neurons should be an instance of a nonlinearity, "
                            "not an int. Defaulting to LIF."))
            _neurons = core.LIF(neurons)

        # Give a better name if name is default
        if _neurons.name.startswith("<LIF"):
            _neurons.name = self.name + "." + _neurons.__class__.__name__
        self._neurons = _neurons

    @property
    def radius(self):
        """TODO"""
        return self._radius

    @radius.setter
    def radius(self, _radius):
        self._radius = np.asarray(_radius)
        self.eval_points = None  # Invalidate possibly cached eval_points

    @property
    def seed(self):
        """TODO"""
        return self._seed

    @seed.setter
    def seed(self, _seed):
        self._seed = _seed
        self.rng = np.random.RandomState(self._seed)
        self.eval_points = None  #Invalidate possibly cached eval_points

    def activities(self, eval_points=None):
        if eval_points is None:
            eval_points = self.eval_points

        return self.neurons.rates(
            np.dot(self.encoder.weights, eval_points).T)

    def connect_to(self, post, **kwargs):
        connection = connections.DecodedNeuronConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='decoded_output', sample_every=0.001, filter=0.01):
        if to_probe == 'decoded_output':
            probe = Probe(self.name + '.decoded_output', sample_every)
            self.connect_to(probe, filter=filter)
            self.probes['decoded_output'].append(probe)
        return probe

    def build(self, model, dt, signal=None):
        # Set up signal
        if signal is None:
            self.signal = core.Signal(n=self.dimensions,
                                      name=self.name + ".signal")
            model.add(self.signal)
        else:
            # Assume that a provided signal is already in the model
            self.signal = signal
            self.dimensions = self.signal.size

        # Set up neurons
        max_rates = self.max_rates
        if hasattr(max_rates, 'sample'):
            max_rates = max_rates.sample(self.neurons.n_neurons, rng=self.rng)
        intercepts = self.intercepts
        if hasattr(intercepts, 'sample'):
            intercepts = intercepts.sample(self.neurons.n_neurons, rng=self.rng)
        self.neurons.set_gain_bias(max_rates, intercepts)
        model.add(self.neurons)

        # Set up encoder
        if self.encoders is None:
            encoders = decoders.sample_hypersphere(self.dimensions,
                                                   self.neurons.n_neurons,
                                                   self.rng, surface=True).T
        else:
            encoders = np.asarray(self.encoders, copy=True)
            norm = np.sum(encoders * encoders, axis=1)[:, np.newaxis]
            encoders /= np.sqrt(norm)
        encoders /= np.asarray(self.radius)
        encoders *= self.neurons.gain[:, np.newaxis]
        self.encoder = core.Encoder(self.signal, self.neurons, encoders)
        model.add(self.encoder)

        # Set up probes, but don't build them (done explicitly later)
        for probe in self.probes['decoded_output']:
            probe.dimensions = self.dimensions


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
        connection = connections.SignalConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        if filter is not None and filter > 0:
            logger.warning("Filter set on constant. Ignoring.")

        if to_probe == 'output':
            p = core.Probe(None, sample_every)
            self.probes['output'].append(p)
        return p

    def build(self, model, dt):
        # Set up signal
        self.signal = core.Constant(self.output.size, self.output,
                                    name=self.name)
        model.add(self.signal)

        # Set up probes
        for probe in self.probes['output']:
            probe.sig = self.signal
            model.add(probe)


class Node(object):
    """Provides arbitrary data to Nengo objects.

    It can also accept input, and perform arbitrary computations
    for the purpose of controlling a Nengo simulation.
    Nodes are typically not part of a brain model per se,
    but serve to summarize the assumptions being made
    about sensory data or other environment variables
    that cannot be generated by a brain model alone.
    Nodes are also useful to test models in various situations.

    Attributes
    ----------
    name : type
        description
    output
    dimensions : type
        description

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
        connection = connections.DecodedConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connections_in.append(connection)
        return connection

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        if to_probe == 'output':
            p = Probe(self.name + ".output", sample_every)
            self.connect_to(p, filter=filter)
            self.probes['output'].append(p)
        return p

    def build(self, model, dt):
        # Set up signals
        self.signal = core.Signal(self.dimensions,
                                  name=self.name + ".signal")
        model.add(self.signal)

        # Set up non-linearity
        n_out = np.array(self.output(np.ones(self.dimensions))).size
        self.nonlinear = core.Direct(n_in=self.dimensions,
                                     n_out=n_out,
                                     fn=self.output,
                                     name=self.name + ".Direct")
        model.add(self.nonlinear)

        # Set up encoder
        self.encoder = core.Encoder(self.signal, self.nonlinear,
                                    weights=np.eye(self.dimensions))
        model.add(self.encoder)

        # Set up probes
        for probe in self.probes['output']:
            probe.dimensions = self.nonlinear.output_signal.n


class Probe(object):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    """
    def __init__(self, name, sample_every, dimensions=None):
        self.name = "Probe(" + name + ")"
        self.sample_every = sample_every
        self.dimensions = None

        self.connections_in = []

    @property
    def sample_rate(self):
        return 1.0 / self.sample_every

    def build(self, model, dt):
        # Set up signal
        self.signal = core.Signal(n=self.dimensions, name=self.name)
        model.add(self.signal)

        # Set up probe
        self.probe = core.Probe(self.signal, self.sample_every)
        model.add(self.probe)
