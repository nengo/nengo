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
    logger.debug("Randomly generating %d eval points", num_samples)
    samples = rng.randn(num_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, None]

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

    """

    EVAL_POINTS = 500

    def __init__(self, name, neurons, dimensions,
                 radius=1.0, encoders=None,
                 max_rates=Uniform(200, 300), intercepts=Uniform(-1.0, 1.0),
                 decoder_noise=None, eval_points=None,
                 noise=None, noise_frequency=None, seed=None,
                 input_signal=None):
        """
        TODO
        """

        # Error for things not implemented yet or don't make sense
        if decoder_noise is not None:
            raise NotImplementedError('decoder_noise')
        if noise is not None or noise_frequency is not None:
            raise NotImplementedError('noise')

        self.seed = np.random.randint(2**31-1) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)

        self.name = name
        self.radius = radius

        if eval_points is None:
            eval_points = sample_unit_signal(
                dimensions, Ensemble.EVAL_POINTS, self.rng) * radius
        self.eval_points = eval_points

        # Set up input signal
        self.input_signal = (core.Signal(n=dimensions,
                                         name=name + ".input_signal")
                             if input_signal is None else input_signal)

        # Set up neurons
        if isinstance(neurons, int):
            logger.warning(("neurons should be an instance of a nonlinearity, "
                            "not an int. Defaulting to LIF."))
            neurons = LIF(neurons)
        neurons = copy.deepcopy(neurons)
        neurons.name = name + "." + neurons.__class__.__name__

        if hasattr(max_rates, 'sample'):
            max_rates = max_rates.sample(neurons.n_neurons, rng=self.rng)
        if hasattr(intercepts, 'sample'):
            intercepts = intercepts.sample(neurons.n_neurons, rng=self.rng)

        neurons.set_gain_bias(max_rates, intercepts)
        self.neurons = neurons

        # Set up the encoders
        if encoders is None:
            encoders = self.rng.randn(neurons.n_neurons, dimensions)
        else:
            encoders = np.asarray(encoders)
            if encoders.shape == ():
                encoders.shape = (1,)
            if encoders.shape == (dimensions,):
                encoders = np.tile(encoders, (neurons.n_neurons, 1))
        norm = np.sum(encoders * encoders, axis=1)[:, np.newaxis]
        encoders /= np.sqrt(norm)
        self.encoders = encoders

        # Set up connections and probes
        self.connections = []
        self.probes = []
        self.probeable = (
            'decoded_output',  # Default
            'spikes',
        )

    @property
    def dimensions(self):
        return self.input_signal.size

    @property
    def n_neurons(self):
        return self.neurons.n_neurons

    @property
    def encoders(self):
        # NB: Copy is super necessary!
        encoders = np.copy(self.encoder.weights)

        # Undo calculations done for speed reasons
        encoders /= self.neurons.gain[:, np.newaxis]
        encoders *= np.asarray(self.radius)
        return encoders

    @encoders.setter
    def encoders(self, weights):
        if hasattr(self, 'encoder') and self.encoder is not None:
            logger.warning("Encoder being overwritten on %s. If encoder has "
                           "already been added to a model, it must be "
                           "removed manually.", self.name)

        # Do some calculations ahead of time
        weights /= np.asarray(self.radius)
        weights *= self.neurons.gain[:, np.newaxis]
        self.encoder = core.Encoder(self.input_signal, self.neurons, weights)

    @property
    def eval_points(self):
        return self._eval_points

    @eval_points.setter
    def eval_points(self, points):
        points = np.array(points)
        if len(points.shape) == 1:
            points.shape = (1, eval_points.shape[0])
        self._eval_points = points

    def activities(self, eval_points=None):
        if eval_points is None:
            eval_points = self.eval_points

        return self.neurons.rates(
            np.dot(self.encoder.weights, eval_points).T)

    def connect_to(self, post, **kwargs):
        connection = connections.DecodedConnection(self, post, **kwargs)
        self.connections.append(connection)
        return connection

    def probe(self, to_probe='decoded_output',
              sample_every=0.001, filter=0.01, dt=0.001):
        if to_probe == 'decoded_output':
            p = Probe(self.name + '.decoded_output',
                      self.dimensions, sample_every)
            c = self.connect_to(p, filter=filter)

        self.probes.append(p)
        return p, c

    def add_to_model(self, model):
        model.add(self.neurons)
        model.add(self.encoder)
        model.add(self.input_signal)
        for connection in self.connections:
            model.add(connection)
        for probe in self.probes:
            model.add(probe)

    def remove_from_model(self, model):
        model.remove(self.neurons)
        model.remove(self.encoder)
        model.remove(self.input_signal)
        for connection in self.connections:
            model.remove(connection)
        for probe in self.probes:
            model.remove(probe)


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
        Name of this node. Must be unique in the network.
    output : function, list of floats, dict, optional
        The output that should be generated by this node.

        If ``output`` is a function, it will be called on each timestep;
        if it accepts a single parameter, it will be given
        the current time of the simulation.

        If ``output`` is a list of floats, that list will be
        used as constant output.

        If ``output`` is a dict, the output defines a piece-wise constant
        function in which the keys define when the value changes,
        and the values define what the value changes to.

    Attributes
    ----------
    name : str
        A unique name that identifies the node.
    metadata : dict
        An editable dictionary that modelers can use to store
        extra information about a network.

    """

    def __init__(self, name, output, input):
        self.name = name

        if type(input) != core.Signal:
            input = input.signal
        self.input_signal = input

        if callable(output):
            n_out = np.array(output(np.ones(input.size))).size
            self.function = core.Direct(n_in=input.size,
                                        n_out=n_out,
                                        fn=output,
                                        name=name + ".Direct")
            self.encoder = core.Encoder(input, self.function,
                                        weights=np.asarray([[1]]))
            self.transform = core.Transform(1.0, self.signal, self.signal)
        else:
            output = np.asarray(output)
            if output.shape == ():
                output.shape = (1,)
            self._signal = core.Constant(n=output.size, value=output, name=name)

        # Set up connections and probes
        self.connections = []
        self.probes = []

    @property
    def signal(self):
        try:
            return self.function.output_signal
        except AttributeError:
            return self._signal

    def connect_to(self, post, **kwargs):
        connection = connections.SimpleConnection(self.signal, post, **kwargs)
        self.connections.append(connection)
        return connection

    def probe(self, to_probe='output',
              sample_every=0.001, filter=0.0, dt=0.001):
        if to_probe == 'output':
            c = None
            if filter <= dt:
                p = RawProbe(self.signal, sample_every)
            else:
                p = Probe(self.name + ".output", self.signal.n, sample_every)
                c = self.connect_to(p, filter=filter, dt=dt)

        self.probes.append(p)
        return p, c

    def add_to_model(self, model):
        if hasattr(self, 'function'):
            model.add(self.function)
            model.add(self.encoder)
            model.add(self.transform)
        else:
            model.add(self._signal)
        if not self.input_signal in model.signals:
            model.add(self.input_signal)
        for connection in self.connections:
            model.add(connection)
        for probe in self.probes:
            model.add(probe)

    def remove_from_model(self, model):
        if hasattr(self, 'function'):
            model.remove(self.function)
            model.remove(self.encoder)
            model.remove(self.transform)
        else:
            model.remove(self._signal)
        model.remove(self.input_signal)
        for connection in self.connections:
            model.remove(connection)
        for probe in self.probes:
            model.remove(probe)


class RawProbe(object):
    """A raw probe is a wrapper around `nengo.core.Probe`.

    This wrapper is necessary because `nengo.Model` expects
    the `nengo.core.Probe` object to be `Probe.probe`.

    """
    def __init__(self, signal, sample_every):
        self.probe = core.Probe(signal, sample_every)

    @property
    def sample_every(self):
        return self.probe.dt

    @property
    def sample_rate(self):
        return 1.0 / self.probe.dt

    def add_to_model(self, model):
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.probe)


class Probe(object):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    Parameters
    ==========
    probed : Nengo object
        The object being probed.

    """
    def __init__(self, name, n_in, sample_every):
        self.input_signal = core.Signal(n=n_in, name="Probe(" + name + ")")
        self.probe = core.Probe(self.input_signal, sample_every)

    @property
    def name(self):
        return self.input_signal.name

    @property
    def sample_every(self):
        return self.probe.dt

    @property
    def sample_rate(self):
        return 1.0 / self.probe.dt

    def add_to_model(self, model):
        model.add(self.input_signal)
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.input_signal)
        model.remove(self.probe)
