import copy
import inspect
import logging
import numpy as np

from . import connections
from . import core
from . import decoders
import simulator

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
    dimensions
    encoders
    eval_points
    n_neurons
    neurons
    radius
    seed
    """

    EVAL_POINTS = 500

    def __init__(self, name, neurons, dimensions, **kwargs):
        self.name = name
        self.neurons = neurons
        self.dimensions = dimensions

        if neurons.n_neurons <= 0:
            raise ValueError('number of neurons (%d) must be positive' %
                             neurons.n_neurons)

        if dimensions <= 0:
            raise ValueError('number of dimensions (%d) must be positive' %
                             dimensions)

        if 'decoder_noise' in kwargs:
            raise NotImplementedError('decoder_noise')

        if 'noise' in kwargs or 'noise_frequency' in kwargs:
            raise NotImplementedError('noise')

        self.encoders = kwargs.get('encoders', None)
        self.intercepts = kwargs.get('intercepts', Uniform(-1.0, 1.0))
        self.max_rates = kwargs.get('max_rates', Uniform(200, 400))
        self.radius = kwargs.get('radius', 1.0)
        self.seed = kwargs.get('seed', np.random.randint(2**31-1))

        # Order matters here
        self.eval_points = kwargs.get('eval_points', None)

        # Set up connections and probes
        self.connections_in = []
        self.connections_out = []
        self.probes = {'decoded_output': [], 'spikes': [], 'voltages': []}

    def __str__(self):
        return "Ensemble: " + self.name

    @property
    def dimensions(self):
        """The number of representational dimensions.

        Returns
        -------
        ~ : int
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, _dimensions):
        self._dimensions = _dimensions
        self.eval_points = None  # Invalidate possibly cached eval_points

    @property
    def encoders(self):
        """The encoders, used to transform from representational space
        to neuron space.

        Each row is a neuron's encoder, each column is a representational
        dimension.

        Returns
        -------
        ~ : ndarray (`n_neurons`, `dimensions`)
        """
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
        """The evaluation points used for decoder solving.

        Returns
        -------
        ~ : ndarray (n_eval_points, `dimensions`)
        """
        if self._eval_points is None:
            self._eval_points = decoders.sample_hypersphere(
                self.dimensions, Ensemble.EVAL_POINTS, self.rng) * self.radius
        return self._eval_points

    @eval_points.setter
    def eval_points(self, _eval_points):
        if _eval_points is not None:
            _eval_points = np.asarray(_eval_points)
            if _eval_points.ndim == 1:
                _eval_points.shape = (-1, 1)
        self._eval_points = _eval_points

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
            _neurons = core.LIF(_neurons)

        # Give a better name if name is default
        if _neurons.name.startswith("<LIF"):
            _neurons.name = self.name + "." + _neurons.__class__.__name__
        self._neurons = _neurons

    @property
    def radius(self):
        """The representational radius of the ensemble.

        Returns
        -------
        ~ : float
        """
        return self._radius

    @radius.setter
    def radius(self, _radius):
        self._radius = np.asarray(_radius)
        self.eval_points = None  # Invalidate possibly cached eval_points

    @property
    def seed(self):
        """The seed used for random number generation.

        Returns
        -------
        ~ : int
        """
        return self._seed

    @seed.setter
    def seed(self, _seed):
        self._seed = _seed
        self.rng = np.random.RandomState(self._seed)
        self.eval_points = None  #Invalidate possibly cached eval_points

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
            np.dot(eval_points, self.encoders.T))
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
        connection = connections.DecodedConnection(self, post, **kwargs)
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
            connection = connections.NonlinearityConnection(
                self.neurons, probe, filter=None,
                transform=np.eye(self.n_neurons))
            self.connections_out.append(connection)
            if hasattr(probe, 'connections_in'):
                probe.connections_in.append(connection)
            self.probes['spikes'].append(probe)

        elif to_probe == 'voltages':
            probe = Probe(self.name + '.voltages', sample_every, self.n_neurons)
            connection = connections.SignalConnection(
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

    def build(self, model, dt, signal=None):
        """Prepare this ensemble for simulation.

        Called automatically by `model.build`.
        """
        # assert self in model.objs.values(), "Not added to model"

        # Set up signal
        if signal is None:
            self.signal = core.Signal(n=self.dimensions,
                                      name=self.name + ".signal")
            model.add(self.signal)
        else:
            # Assume that a provided signal is already in the model
            self.signal = signal
            self.dimensions = self.signal.size
            
        #reset input signal to 0 each timestep
        model._operators += [simulator.Reset(self.signal)]

        # Set up neurons
        max_rates = self.max_rates
        if hasattr(max_rates, 'sample'):
            max_rates = max_rates.sample(self.neurons.n_neurons, rng=self.rng)
        intercepts = self.intercepts
        if hasattr(intercepts, 'sample'):
            intercepts = intercepts.sample(self.neurons.n_neurons, rng=self.rng)
        #intercepts *= self.radius
        self.neurons.set_gain_bias(max_rates, intercepts)
        model.add(self.neurons)

        # Set up encoder
        if self.encoders is None:
            self.encoders = decoders.sample_hypersphere(
                self.dimensions, self.neurons.n_neurons,
                self.rng, surface=True)
        else:
            self.encoders = np.asarray(self.encoders, dtype=float).copy()
            norm = np.sum(self.encoders * self.encoders, axis=1)[:, np.newaxis]
            self.encoders /= np.sqrt(norm)
        self.encoders /= np.asarray(self.radius)
        self.encoders *= self.neurons.gain[:, np.newaxis]
        model._operators += [simulator.DotInc(core.Constant(self.encoders),
                    self.signal, self.neurons.input_signal)]

        # Set up probes, but don't build them (done explicitly later)
        # Note: Have to set it up here because we only know these things (dimensions,
        #       n_neurons) at build time.
        for probe in self.probes['decoded_output']:
            probe.dimensions = self.dimensions
        for probe in self.probes['spikes']:
            probe.dimensions = self.n_neurons
        for probe in self.probes['voltages']:
            probe.dimensions = self.n_neurons


class PassthroughNode(object):
    def __init__(self, name, dimensions=1):
        self.name = name
        self.dimensions = dimensions

        self.connections_out = []
        self.probes = {'output': []}

    def connect_to(self, post, **kwargs):
        connection = connections.SignalConnection(self, post, **kwargs)
        self.connections_out += [connection]

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self

    def build(self, model, dt):
        self.signal = core.Signal(n=self.dimensions,
                                  name=self.name + ".signal")
        model.add(self.signal)

        # Set up probes
        for probe in self.probes['output']:
            probe.sig = self.signal
            model.add(probe)

    def probe(self, to_probe='output', sample_every=0.001, filter=None):
        if to_probe == 'output':
            p = core.Probe(None, sample_every)
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

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self

    def build(self, model, dt):
        # Set up signal
        self.signal = core.Constant(self.output,
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
        connection = connections.NonlinearityConnection(self, post, **kwargs)
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

    def build(self, model, dt):
        """TODO"""
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
        model._operators += [simulator.DotInc(core.Constant(
            np.eye(self.dimensions)), self.signal, self.nonlinear.input_signal)]

        # Set up probes
        for probe in self.probes['output']:
            probe.dimensions = self.nonlinear.output_signal.n


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

    def build(self, model, dt):
        """TODO"""

        # Set up signal
        self.signal = core.Signal(n=self.dimensions, name=self.name)
        model.add(self.signal)

        # Set up probe
        self.probe = core.Probe(self.signal, self.sample_every)
        model.add(self.probe)
