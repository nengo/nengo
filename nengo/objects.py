
import inspect
import logging
import numpy as np

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

def filter_coefs(pstc, dt):
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


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
        self.input_signal = Signal(n=dimensions, name=name + ".input_signal")

        neurons.set_gain_bias(max_rates, intercepts)
        self.neurons = neurons

        # Set up the encoders
        encoders *= self.neurons.gain[:, None]
        self.encoders = Encoder(self.input_signal, self.neurons, encoders)

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

            self.decoded_output = Signal(n=self.dimensions,
                                         name=self.name + ".decoded_output")
            activites = self.activities() * dt
            targets = self.eval_points.T
            self.decoders = Decoder(
                sig=self.decoded_output, pop=self.neurons,
                weights=decoders.solve_decoders(activites, targets))
            self.transform = Transform(
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
        from .probes import FilteredProbe, RawProbe

        if to_probe == '':
            to_probe = 'decoded_output'

        if to_probe == 'decoded_output':
            self._add_decoded_output(model)
            if filter is not None and filter > dt_sample:
                logger.debug("Creating filtered probe")
                dt = 0.001 if model is None else model.dt
                p = FilteredProbe(self.decoded_output, dt_sample, filter, dt)
            else:
                logger.debug("Creating raw probe")
                p = RawProbe(self.decoded_output, dt_sample)

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

        if type(input) != Signal:
            input = input.signal
        self.input_signal = input

        if callable(output):
            n_out = np.array(output(np.ones(input.size))).size
            self.function = Direct(n_in=input.size,
                                   n_out=n_out,
                                   fn=output,
                                   name=name + ".Direct")
            self.encoder = Encoder(input, self.function,
                                   weights=np.asarray([[1]]))
            self.signal = self.function.output_signal
            self.transform = Transform(1.0, self.signal, self.signal)
        else:
            if type(output) == float:
                output = [output]

            if type(output) == list:
                self.signal = Constant(n=len(output),
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


"""
Low-level objects
=================

These classes are used to describe a Nengo model to be simulated.
Model is the input to a *simulator* (see e.g. simulator.py).

"""


random_weight_rng = np.random.RandomState(12345)

"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
    pass


class SignalView(object):
    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)
        if name is not None:
            self._name = name

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def __repr__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    @property
    def dtype(self):
        return np.dtype(self.base._dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise TODO('reshape of strided view')

    def transpose(self, neworder=None):
        raise TODO('transpose')

    def __getitem__(self, item):
        # -- copy the shape and strides
        shape = list(self.shape)
        elemstrides = list(self.elemstrides)
        offset = self.offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            elemstrides = self.elemstrides[1:]
            offset = self.offset + item * self.elemstrides[0]
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            if self.base is self:
                return '<anon%d>' % id(self)
            else:
                return 'View(%s)' % self.base.name

    @name.setter
    def name(self, value):
        self._name = value

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'base': self.base.name,
            'shape': list(self.shape),
            'elemstrides': list(self.elemstrides),
            'offset': self.offset,
        }


class Signal(SignalView):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1, dtype=np.float64, name=None):
        self.n = n
        self._dtype = dtype
        if name is not None:
            self._name = name
        if assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", " + str(self.n) + "D)"
        except AttributeError:
            return "Signal (id " + str(id(self)) + ", " + str(self.n) + "D)"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return (self.n,)

    @property
    def elemstrides(self):
        return (1,)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

    def add_to_model(self, model):
        model.signals.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'n': self.n,
            'dtype': str(self.dtype),
        }


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return "Probing " + str(self.sig)

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.probes.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'dt': self.dt,
        }


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value, name=None):
        Signal.__init__(self, n, name=name)
        self.value = np.asarray(value)
        # TODO: change constructor to get n from value
        assert self.value.size == n

    def __str__(self):
        if self.name is not None:
            return "Constant(" + self.name + ")"
        return "Constant(id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'value': self.value.tolist(),
        }


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        alpha = np.asarray(alpha)
        if hasattr(outsig, 'value'):
            raise TypeError('transform destination is constant')

        name = insig.name + ">" + outsig.name + ".tf_alpha"

        self.alpha_signal = Constant(n=alpha.size, value=alpha, name=name)
        self.insig = insig
        self.outsig = outsig
        if self.alpha_signal.size == 1:
            if self.insig.shape != self.outsig.shape:
                raise ShapeMismatch()
        else:
            if self.alpha_signal.shape != (
                    self.outsig.shape + self.insig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.insig.shape,
                        self.outsig.shape,
                        )

    def __str__(self):
        return ("Transform (id " + str(id(self)) + ")"
                " from " + str(self.insig) + " to " + str(self.outsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.add(self.alpha_signal)
        model.transforms.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'insig': self.insig.name,
            'outsig': self.outsig.name,
        }


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        if hasattr(newsig, 'value'):
            raise TypeError('filter destination is constant')
        alpha = np.asarray(alpha)

        name = oldsig.name + ">" + newsig.name + ".f_alpha"

        self.alpha_signal = Constant(n=alpha.size, value=alpha, name=name)
        self.oldsig = oldsig
        self.newsig = newsig

        if self.alpha_signal.size == 1:
            if self.oldsig.shape != self.newsig.shape:
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )
        else:
            if self.alpha_signal.shape != (
                    self.newsig.shape + self.oldsig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )

    def __str__(self):
        return ("Filter (id " + str(id(self)) + ")"
                " from " + str(self.oldsig) + " to " + str(self.newsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.add(self.alpha_signal)
        model.filters.add(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'oldsig': self.oldsig.name,
            'newsig': self.newsig.name,
        }


class Encoder(object):
    """A linear transform from a signal to a population"""
    def __init__(self, sig, pop, weights=None):
        self.sig = sig
        self.pop = pop
        if weights is None:
            weights = random_weight_rng.randn(pop.n_in, sig.size)
        else:
            weights = np.asarray(weights)
            if weights.shape != (pop.n_in, sig.size):
                raise ValueError('weight shape', weights.shape)

        name = sig.name + ".encoders"
        self.weights_signal = Constant(n=weights.size, value=weights, name=name)

    def __str__(self):
        return ("Encoder (id " + str(id(self)) + ")"
                " of " + str(self.sig) + " to " + str(self.pop))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.encoders.add(self)
        model.signals.add(self.weights_signal)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'pop': self.pop.name,
            'weights': self.weights.tolist(),
        }


class Decoder(object):
    """A linear transform from a population to a signal"""
    def __init__(self, pop, sig, weights=None):
        self.pop = pop
        self.sig = sig
        if weights is None:
            weights = random_weight_rng.randn(sig.size, pop.n_out)
        else:
            weights = np.asarray(weights)
            if weights.shape != (sig.size, pop.n_out):
                raise ValueError('weight shape', weights.shape)
        name = sig.name + ".decoders"
        self.weights_signal = Constant(n=weights.size, value=weights, name=name)

    def __str__(self):
        return ("Decoder (id " + str(id(self)) + ")"
                " of " + str(self.pop) + " to " + str(self.sig))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.decoders.add(self)
        model.signals.add(self.weights_signal)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'pop': self.pop.name,
            'sig': self.sig.name,
            'weights': self.weights.tolist(),
        }


### Nonlinearities

class Nonlinearity(object):
    def __str__(self):
        return "Nonlinearity (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.nonlinearities.add(self)
        model.signals.add(self.bias_signal)
        model.signals.add(self.input_signal)
        model.signals.add(self.output_signal)


class Direct(Nonlinearity):
    def __init__(self, n_in, n_out, fn, name=None):
        if name is None:
            name = "<Direct%d>" % id(self)
        self.name = name

        self.input_signal = Signal(n_in, name=name + '.input')
        self.output_signal = Signal(n_out, name=name + '.output')
        self.bias_signal = Constant(n=n_in,
                                    value=np.zeros(n_in),
                                    name=name + '.bias')

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __str__(self):
        return "Direct (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def fn(self, J):
        return J

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'fn': inspect.getsource(self.fn),
        }



class LIF(Nonlinearity):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, upsample=1,
                name=None):
        if name is None:
            name = "<LIF%d>" % id(self)
        self.input_signal = Signal(n_neurons, name=name + '.input')
        self.output_signal = Signal(n_neurons, name=name + '.output')
        self.bias_signal = Constant(n=n_neurons,
                                    value=np.zeros(n_neurons),
                                    name=name + '.bias')

        self._name = name
        self.n_neurons = n_neurons
        self.upsample = upsample
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = None
        self.bias = None

    def __str__(self):
        return "LIF (id " + str(id(self)) + ", " + str(self.n_neurons) + "N)"

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.input_signal.name = value + '.input'
        self.output_signal.name = value + '.output'
        self.bias_signal.name = value + '.bias'

    @property
    def bias(self):
        return self.bias_signal.value

    @bias.setter
    def bias(self, value):
        self.bias_signal.value[...] = value

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        """
        logging.debug("Setting gain and bias on %s", self.name)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked[:] = (v > 1) * 1.0

        old = np.seterr(all='ignore')
        try:

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time

    def rates(self, J_without_bias):
        """Return LIF firing rates for current J in Hz

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
        """
        old = np.seterr(all='ignore')
        J = J_without_bias + self.bias
        try:
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'n_neurons': self.n_neurons,
            'upsample': self.upsample,
            'tau_rc': self.tau_rc,
            'tau_ref': self.tau_ref,
            'gain': self.gain.tolist(),
        }


class LIFRate(LIF):
    def __init__(self, n_neurons):
        LIF.__init__(self, n_neurons)
        self.input_signal = Signal(n_neurons)
        self.output_signal = Signal(n_neurons)
        self.bias_signal = Constant(n=n_neurons, value=np.zeros(n_neurons))

    def __str__(self):
        return "LIFRate (id " + str(id(self)) + ", " + str(self.n_neurons) + "N)"

    def __repr__(self):
        return str(self)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons
