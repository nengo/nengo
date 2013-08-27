import inspect
import logging
import numpy as np

from . import core
from . import decoders
from . import probes

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

### High-level objects

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


class Network(object):
    def __init__(self, name, seed, parent):
        self.model = model
        self.name = name

    def add(self, obj):
        pass

    def get(self, target, default=None):
        pass

    def remove(self, target):
        pass

    def connect(self, *args, **kwargs):
        pass

    def connect_neurons(self, *args, **kwargs):
        pass

    def make_alias(self, alias, target):
        pass

    def make_ensemble(self, *args, **kwargs):
        pass

    def make_network(self, *args, **kwargs):
        pass

    def make_node(self, *args, **kwargs):
        pass

    def probe(self, target, sample_every=None, static=False):
        pass


class Ensemble(object):
    """A collection of neurons that collectively represent a vector.

    """

    EVAL_POINTS = 500

    def __init__(self, name, neurons, dimensions,
                 radius=1.0, encoders=None,
                 max_rates=Uniform(200, 300), intercepts=Uniform(-1.0, 1.0),
                 decoder_noise=None, eval_points=None,
                 noise=None, noise_frequency=None, seed=None):
        # Error for things not implemented yet or don't make sense
        if decoder_noise is not None:
            raise NotImplementedError('decoder_noise')
        if noise is not None or noise_frequency is not None:
            raise NotImplementedError('noise')

        self.seed = np.random.randint(2**31-1) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)

        if isinstance(neurons, int):
            logger.warning(("neurons should be an instance of a nonlinearity, "
                          "not an int. Defaulting to LIF."))
            neurons = LIF(neurons)
        neurons.name = name + "." + neurons.__class__.__name__

        if hasattr(max_rates, 'sample'):
            max_rates = max_rates.sample(neurons.n_neurons, rng=self.rng)
        if hasattr(intercepts, 'sample'):
            intercepts = intercepts.sample(neurons.n_neurons, rng=self.rng)

        if eval_points is None:
            eval_points = sample_unit_signal(
                dimensions, Ensemble.EVAL_POINTS, self.rng) * radius

        if encoders is None:
            logger.debug("Randomly generating encoders, shape=(%d, %d)",
                         neurons.n_neurons, dimensions)
            encoders = self.rng.randn(neurons.n_neurons, dimensions)
            norm = np.sum(encoders * encoders, axis=1)[:, None]
            encoders /= np.sqrt(norm)
        encoders /= radius

        self.name = name
        self.radius = radius
        self.eval_points = eval_points

        # The essential components of an ensemble are:
        self.input_signal = core.Signal(n=dimensions,
                                        name=name + ".input_signal")

        neurons.set_gain_bias(max_rates, intercepts)
        self.neurons = neurons

        # Set up the encoders
        encoders *= self.neurons.gain[:, None]
        self.encoders = core.Encoder(self.input_signal, self.neurons, encoders)

        # Set up probes
        self.probes = []
        self.probeable = (
            'decoded_output',  # Default
            'spikes',
        )

    @property
    def dimensions(self):
        return self.input_signal.n

    @property
    def eval_points(self):
        return self._eval_points

    @eval_points.setter
    def eval_points(self, points):
        points = np.array(points)
        if len(points.shape) == 1:
            points.shape = [1, eval_points.shape[0]]
        self._eval_points = points

    @property
    def n_neurons(self):
        return self.neurons.n_neurons

    def _add_decoded_output(self, model=None):
        if not hasattr(self, 'decoded_output'):
            dt = 0.001 if model is None else model.dt

            self.decoded_output = core.Signal(
                n=self.dimensions, name=self.name + ".decoded_output")
            activites = self.activities() * dt
            targets = self.eval_points.T
            self.decoders = core.Decoder(
                sig=self.decoded_output, pop=self.neurons,
                weights=decoders.solve_decoders(activites, targets))
            self.transform = core.Transform(
                1.0, self.decoded_output, self.decoded_output)
            if model is not None:
                model.add(self.decoded_output)
                model.add(self.decoders)
                model.add(self.transform)

    def activities(self, eval_points=None):
        if eval_points is None:
            eval_points = self.eval_points

        return self.neurons.rates(
            np.dot(self.encoders.weights, eval_points).T)

    def probe(self, to_probe, dt_sample, filter=None, model=None):


        if to_probe == '':
            to_probe = 'decoded_output'

        if to_probe == 'decoded_output':
            self._add_decoded_output(model)
            if filter is not None and filter > dt_sample:
                logger.debug("Creating filtered probe")
                dt = 0.001 if model is None else model.dt
                p = probes.FilteredProbe(
                    self.decoded_output, dt_sample, filter, dt)
            else:
                logger.debug("Creating raw probe")
                p = probes.RawProbe(self.decoded_output, dt_sample)

        self.probes.append(p)
        if model is not None:
            model.add(p)
        return p

    def add_to_model(self, model):
        model.add(self.neurons)
        model.add(self.encoders)
        model.add(self.input_signal)
        if hasattr(self, 'decoded_output'):
            model.add(self.decoded_output)
            model.add(self.decoders)
            model.add(self.transform)
        for probe in self.probes:
            model.add(probe)

    def remove_from_model(self, model):
        model.remove(self.neurons)
        model.remove(self.encoders)
        model.remove(self.input_signal)
        if hasattr(self, 'decoded_output'):
            model.remove(self.decoded_output)
            model.remove(self.decoders)
            model.remove(self.transform)
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
            self.signal = self.function.output_signal
            self.transform = core.Transform(1.0, self.signal, self.signal)
        else:
            if type(output) == float:
                output = [output]

            if type(output) == list:
                self.signal = core.Constant(n=len(output),
                                            value=[float(n) for n in output],
                                            name=name)

    def probe(self, model):
        pass

    def add_to_model(self, model):
        if hasattr(self, 'function'):
            model.add(self.function)
        if hasattr(self, 'encoder'):
            model.add(self.encoder)
        if hasattr(self, 'transform'):
            model.add(self.transform)
        # model.add(self.input_signal)  # Should already be in network
        model.add(self.signal)

    def remove_from_model(self, model):
        raise NotImplementedError
