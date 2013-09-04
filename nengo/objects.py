import copy
import inspect
import logging
import numpy as np

from . import connections
from . import core
from . import decoders

logger = logging.getLogger(__name__)


def sample_unit_signal(dimensions, num_samples, rng):
    """Generate sample points uniformly distributed within the sphere.

    Returns float array of sample points: dimensions x num_samples

    """
    samples = rng.randn(num_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, np.newaxis]

    # generate magnitudes for vectors from uniform distribution
    scale = rng.rand(num_samples, 1) ** (1.0 / dimensions)

    # scale sample points
    samples *= scale

    return samples.T


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
                msg = ("Encoder shape must be (n_neurons, dimensions); "
                       "in this case %s." % str(enc_shape))
                raise core.ShapeMismatchError(msg)
        self._encoders = _encoders

    @property
    def eval_points(self):
        """TODO"""
        if self._eval_points is None:
            self._eval_points = sample_unit_signal(
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

        # We needed this for the EnsembleArray template, as it would
        # pass the same neurons object to each ensemble. But, this should
        # be done at the template level, as it's not obvious that the
        # passed in neuron object would become a copy.
        # _neurons = copy.deepcopy(_neurons)

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
        connection = connections.DecodedConnection(self, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connection_in.append(connection)
        return connection

    def probe(self, to_probe='decoded_output', sample_every=0.001, filter=0.01):
        if to_probe == 'decoded_output':
            probe = probes.Probe(self.name + '.decoded_output', sample_every)
            self.connect_to(probe, filter=filter)
            self.probes['decoded_output'].append(probe)
        return probe

    def build(self, model, dt, input_signal=None):
        # Set up input_signal
        if input_signal is None:
            self.input_signal = core.Signal(n=self.dimensions,
                                            name=self.name + ".input_signal")
            model.add(self.input_signal)
        else:
            # Assume that a provided input_signal is already in the model
            self.input_signal = input_signal

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
            encoders = self.rng.randn(self.neurons.n_neurons, self.dimensions)
        else:
            encoders = np.asarray(self.encoders, copy=True)
        norm = np.sum(encoders * encoders, axis=1)[:, np.newaxis]
        encoders /= np.sqrt(norm)
        encoders /= np.asarray(self.radius)
        encoders *= self.neurons.gain[:, np.newaxis]
        self.encoder = core.Encoder(self.input_signal, self.neurons, encoders)
        model.add(self.encoder)

        # Set up probes, but don't build them (done explicitly later)
        for probe in self.probes['decoded_output']:
            probe.dimensions = self.dimensions


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
        self.name = name
        self.output = output
        self.dimensions = dimensions

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'output': []}

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, _output):
        if callable(_output):
            self._output = _output
        else:
            self._output = np.asarray(_output)
            if self._output.shape == ():
                self._output.shape = (1,)

    def connect_to(self, post, **kwargs):
        connection = connections.SimpleConnection(self.signal, post, **kwargs)
        self.connections_out.append(connection)
        if hasattr(post, 'connections_in'):
            post.connection_in.append(connection)
        return connection

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        if to_probe == 'output':
            p = probes.Probe(self.name + ".output", sample_every)
            self.connect_to(p, filter=filter)
            self.probes['output'].append(p)
        return p

    def build(self, model, dt):
        if callable(self.output):
            # Set up input_signal
            self.input_signal = core.Signal(self.dimensions,
                                            name=self.name + ".input")
            model.add(self.input_signal)

            # Set up non-linearity
            n_out = np.array(self.output(np.ones(self.dimensions))).size
            self.function = core.Direct(n_in=self.dimensions,
                                        n_out=n_out,
                                        fn=self.output,
                                        name=self.name + ".Direct")
            model.add(self.function)
            self.signal = self.function.output_signal

            # Set up encoder
            self.encoder = core.Encoder(self.input_signal, self.function,
                                        weights=np.asarray([[1]]))
            model.add(self.encoder)

            # Set up transform
            self.transform = core.Transform(1.0, self.signal, self.signal)
            model.add(self.transform)
        else:
            self.signal = core.Constant(n=self.output.size, value=self.output,
                                        name=self.name)
            model.add(self.signal)

        # Set up probes
        for probe in self.probes['output']:
            probe.dimensions = self.signal.size


class Probe(object):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    """
    def __init__(self, name, sample_every, dimensions=None):
        self.name = name
        self.sample_every = sample_every
        self.dimensions = None

        self.connections_in = []

    @property
    def sample_rate(self):
        return 1.0 / self.sample_every

    def build(self, model, dt):
        # Set up input_signal
        self.input_signal = core.Signal(n=self.dimensions,
                                        name="Probe(" + self.name + ")")
        model.add(self.input_signal)

        # Set up probe
        self.probe = core.Probe(self.input_signal, self.sample_every)
        model.add(self.probe)
