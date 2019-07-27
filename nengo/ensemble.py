from nengo.base import NengoObject, ObjView, ProcessParam
from nengo.dists import DistOrArrayParam, Uniform, UniformHypersphere
from nengo.exceptions import ReadonlyError
from nengo.neurons import LIF, NeuronTypeParam, Direct
from nengo.params import BoolParam, Default, IntParam, NumberParam


class Ensemble(NengoObject):
    """A group of neurons that collectively represent a vector.

    Parameters
    ----------
    n_neurons : int
        The number of neurons.
    dimensions : int
        The number of representational dimensions.

    radius : int, optional
        The representational radius of the ensemble.
    encoders : Distribution or (n_neurons, dimensions) array_like, optional
        The encoders used to transform from representational space
        to neuron space. Each row is a neuron's encoder; each column is a
        representational dimension.
    intercepts : Distribution or (n_neurons,) array_like, optional
        The point along each neuron's encoder where its activity is zero. If
        ``e`` is the neuron's encoder, then the activity will be zero when
        ``dot(x, e) <= c``, where ``c`` is the given intercept.
    max_rates : Distribution or (n_neurons,) array_like, optional
        The activity of each neuron when the input signal ``x`` is magnitude 1
        and aligned with that neuron's encoder ``e``;
        i.e., when ``dot(x, e) = 1``.
    eval_points : Distribution or (n_eval_points, dims) array_like, optional
        The evaluation points used for decoder solving, spanning the interval
        (-radius, radius) in each dimension, or a distribution from which
        to choose evaluation points.
    n_eval_points : int, optional
        The number of evaluation points to be drawn from the ``eval_points``
        distribution. If None, then a heuristic is used to determine
        the number of evaluation points.
    neuron_type : `~nengo.neurons.NeuronType`, optional
        The model that simulates all neurons in the ensemble
        (see `~nengo.neurons.NeuronType`).
    gain : Distribution or (n_neurons,) array_like
        The gains associated with each neuron in the ensemble. If None, then
        the gain will be solved for using ``max_rates`` and ``intercepts``.
    bias : Distribution or (n_neurons,) array_like
        The biases associated with each neuron in the ensemble. If None, then
        the gain will be solved for using ``max_rates`` and ``intercepts``.
    noise : Process, optional
        Random noise injected directly into each neuron in the ensemble
        as current. A sample is drawn for each individual neuron on
        every simulation step.
    normalize_encoders : bool, optional
        Indicates whether the encoders should be normalized.
    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional
        The seed used for random number generation.

    Attributes
    ----------
    bias : Distribution or (n_neurons,) array_like or None
        The biases associated with each neuron in the ensemble.
    dimensions : int
        The number of representational dimensions.
    encoders : Distribution or (n_neurons, dimensions) array_like
        The encoders, used to transform from representational space
        to neuron space. Each row is a neuron's encoder, each column is a
        representational dimension.
    eval_points : Distribution or (n_eval_points, dims) array_like
        The evaluation points used for decoder solving, spanning the interval
        (-radius, radius) in each dimension, or a distribution from which
        to choose evaluation points.
    gain : Distribution or (n_neurons,) array_like or None
        The gains associated with each neuron in the ensemble.
    intercepts : Distribution or (n_neurons) array_like or None
        The point along each neuron's encoder where its activity is zero. If
        ``e`` is the neuron's encoder, then the activity will be zero when
        ``dot(x, e) <= c``, where ``c`` is the given intercept.
    label : str or None
        A name for the ensemble. Used for debugging and visualization.
    max_rates : Distribution or (n_neurons,) array_like or None
        The activity of each neuron when ``dot(x, e) = 1``,
        where ``e`` is the neuron's encoder.
    n_eval_points : int or None
        The number of evaluation points to be drawn from the ``eval_points``
        distribution. If None, then a heuristic is used to determine
        the number of evaluation points.
    n_neurons : int or None
        The number of neurons.
    neuron_type : NeuronType
        The model that simulates all neurons in the ensemble
        (see ``nengo.neurons``).
    noise : Process or None
        Random noise injected directly into each neuron in the ensemble
        as current. A sample is drawn for each individual neuron on
        every simulation step.
    radius : int
        The representational radius of the ensemble.
    seed : int or None
        The seed used for random number generation.
    """

    probeable = ("decoded_output", "input", "scaled_encoders")

    n_neurons = IntParam("n_neurons", low=1)
    dimensions = IntParam("dimensions", low=1)
    radius = NumberParam("radius", default=1.0, low=1e-10)
    encoders = DistOrArrayParam(
        "encoders",
        default=UniformHypersphere(surface=True),
        sample_shape=("n_neurons", "dimensions"),
    )
    intercepts = DistOrArrayParam(
        "intercepts",
        default=Uniform(-1.0, 1.0),
        optional=True,
        sample_shape=("n_neurons",),
    )
    max_rates = DistOrArrayParam(
        "max_rates",
        default=Uniform(200, 400),
        optional=True,
        sample_shape=("n_neurons",),
    )
    eval_points = DistOrArrayParam(
        "eval_points", default=UniformHypersphere(), sample_shape=("*", "dimensions")
    )
    n_eval_points = IntParam("n_eval_points", default=None, optional=True)
    neuron_type = NeuronTypeParam("neuron_type", default=LIF())
    gain = DistOrArrayParam(
        "gain", default=None, optional=True, sample_shape=("n_neurons",)
    )
    bias = DistOrArrayParam(
        "bias", default=None, optional=True, sample_shape=("n_neurons",)
    )
    noise = ProcessParam("noise", default=None, optional=True)
    normalize_encoders = BoolParam("normalize_encoders", default=True, optional=True)

    def __init__(
        self,
        n_neurons,
        dimensions,
        radius=Default,
        encoders=Default,
        intercepts=Default,
        max_rates=Default,
        eval_points=Default,
        n_eval_points=Default,
        neuron_type=Default,
        gain=Default,
        bias=Default,
        noise=Default,
        normalize_encoders=Default,
        label=Default,
        seed=Default,
    ):
        super().__init__(label=label, seed=seed)
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.radius = radius
        self.encoders = encoders
        self.intercepts = intercepts
        self.max_rates = max_rates
        self.n_eval_points = n_eval_points
        self.eval_points = eval_points
        self.bias = bias
        self.gain = gain
        self.neuron_type = neuron_type
        self.noise = noise
        self.normalize_encoders = normalize_encoders

    def __getitem__(self, key):
        return ObjView(self, key)

    def __len__(self):
        return self.dimensions

    @property
    def neurons(self):
        """A direct interface to the neurons in the ensemble."""
        return Neurons(self)

    @neurons.setter
    def neurons(self, dummy):
        raise ReadonlyError(attr="neurons", obj=self)

    @property
    def size_in(self):
        """The dimensionality of the ensemble."""
        return self.dimensions

    @property
    def size_out(self):
        """The dimensionality of the ensemble."""
        return self.dimensions


class Neurons:
    """An interface for making connections directly to an ensemble's neurons.

    This should only ever be accessed through the ``neurons`` attribute of an
    ensemble, as a way to signal to `~nengo.Connection` that the connection
    should be made directly to the neurons rather than to the ensemble's
    decoded value, e.g.::

        nengo.Connection(a.neurons, b.neurons)
    """

    def __init__(self, ensemble):
        self._ensemble = ensemble

    def __getitem__(self, key):
        return ObjView(self, key)

    def __len__(self):
        return self.ensemble.n_neurons

    def __repr__(self):
        return "<Neurons at 0x%x of %r>" % (id(self), self.ensemble)

    def __str__(self):
        return "<Neurons of %s>" % self.ensemble

    def __eq__(self, other):
        return self.ensemble is other.ensemble

    def __hash__(self):
        return hash(self.ensemble) + 1  # +1 to avoid collision with ensemble

    @property
    def ensemble(self):
        """(Ensemble) The ensemble these neurons are part of."""
        return self._ensemble

    @property
    def probeable(self):
        """(tuple) Signals that can be probed in the neuron population."""
        return ("output", "input") + self.ensemble.neuron_type.probeable

    @property
    def size_in(self):
        """(int) The number of neurons in the population."""
        if isinstance(self.ensemble.neuron_type, Direct):
            # This will prevent users from connecting/probing Direct neurons
            # (since there aren't actually any neurons being simulated).
            return 0
        return self.ensemble.n_neurons

    @property
    def size_out(self):
        """(int) The number of neurons in the population."""
        if isinstance(self.ensemble.neuron_type, Direct):
            # This will prevent users from connecting/probing Direct neurons
            # (since there aren't actually any neurons being simulated).
            return 0
        return self.ensemble.n_neurons
