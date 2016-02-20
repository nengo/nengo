import weakref

from nengo.base import NengoObject, ObjView
from nengo.dists import DistOrArrayParam, Uniform, UniformHypersphere
from nengo.neurons import LIF, NeuronTypeParam, Direct
from nengo.params import (
    Default, IntParam, NumberParam, StringParam)
from nengo.processes import ProcessParam


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
    encoders : Distribution or ndarray (`n_neurons`, `dimensions`), optional
        The encoders, used to transform from representational space
        to neuron space. Each row is a neuron's encoder, each column is a
        representational dimension.
    intercepts : Distribution or ndarray (`n_neurons`), optional
        The point along each neuron's encoder where its activity is zero. If
        e is the neuron's encoder, then the activity will be zero when
        dot(x, e) <= c, where c is the given intercept.
    max_rates : Distribution or ndarray (`n_neurons`), optional
        The activity of each neuron when dot(x, e) = 1, where e is the neuron's
        encoder.
    eval_points : Distribution or ndarray (`n_eval_points`, `dims`), optional
        The evaluation points used for decoder solving, spanning the interval
        (-radius, radius) in each dimension, or a distribution from which to
        choose evaluation points. Default: ``UniformHypersphere``.
    n_eval_points : int, optional
        The number of evaluation points to be drawn from the `eval_points`
        distribution. If None (the default), then a heuristic is used to
        determine the number of evaluation points.
    neuron_type : Neurons, optional
        The model that simulates all neurons in the ensemble.
    noise : Process, optional
        Random noise injected directly into each neuron in the ensemble
        as current. A sample is drawn for each individual neuron on
        every simulation step.
    seed : int, optional
        The seed used for random number generation.
    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    """

    n_neurons = IntParam('n_neurons', default=None, low=1)
    dimensions = IntParam('dimensions', default=None, low=1)
    radius = NumberParam('radius', default=1.0, low=1e-10)
    neuron_type = NeuronTypeParam('neuron_type', default=LIF())
    encoders = DistOrArrayParam('encoders',
                                default=UniformHypersphere(surface=True),
                                sample_shape=('n_neurons', 'dimensions'))
    intercepts = DistOrArrayParam('intercepts',
                                  default=Uniform(-1.0, 1.0),
                                  optional=True,
                                  sample_shape=('n_neurons',))
    max_rates = DistOrArrayParam('max_rates',
                                 default=Uniform(200, 400),
                                 optional=True,
                                 sample_shape=('n_neurons',))
    n_eval_points = IntParam('n_eval_points', default=None, optional=True)
    eval_points = DistOrArrayParam('eval_points',
                                   default=UniformHypersphere(),
                                   sample_shape=('*', 'dimensions'))
    bias = DistOrArrayParam('bias',
                            default=None,
                            optional=True,
                            sample_shape=('n_neurons',))
    gain = DistOrArrayParam('gain',
                            default=None,
                            optional=True,
                            sample_shape=('n_neurons',))
    noise = ProcessParam('noise', default=None, optional=True)
    seed = IntParam('seed', default=None, optional=True)
    label = StringParam('label', default=None, optional=True)

    def __init__(self, n_neurons, dimensions, radius=Default, encoders=Default,
                 intercepts=Default, max_rates=Default, eval_points=Default,
                 n_eval_points=Default, neuron_type=Default, gain=Default,
                 bias=Default, noise=Default, seed=Default, label=Default):

        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.radius = radius
        self.encoders = encoders
        self.intercepts = intercepts
        self.max_rates = max_rates
        self.label = label
        self.n_eval_points = n_eval_points
        self.eval_points = eval_points
        self.bias = bias
        self.gain = gain
        self.neuron_type = neuron_type
        self.noise = noise
        self.seed = seed
        self._neurons = Neurons(self)

    def __getitem__(self, key):
        return ObjView(self, key)

    def __len__(self):
        return self.dimensions

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, dummy):
        raise AttributeError("neurons cannot be overwritten.")

    @property
    def probeable(self):
        return ["decoded_output", "input"]

    @property
    def size_in(self):
        return self.dimensions

    @property
    def size_out(self):
        return self.dimensions


class Neurons(object):
    """A wrapper around Ensemble for making connections directly to neurons.

    This should only ever be used in the ``Ensemble.neurons`` property,
    as a way to signal to Connection that the connection should be made
    directly to the neurons rather than to the Ensemble's decoded value.

    Does not currently support any other view-like operations.
    """
    def __init__(self, ensemble):
        self._ensemble = weakref.ref(ensemble)

    def __getitem__(self, key):
        return ObjView(self, key)

    def __len__(self):
        return self.ensemble.n_neurons

    def __repr__(self):
        return "<Neurons at 0x%x of %r>" % (id(self), self.ensemble)

    def __str__(self):
        return "<Neurons of %s>" % self.ensemble

    @property
    def ensemble(self):
        return self._ensemble()

    @property
    def size_in(self):
        if isinstance(self.ensemble.neuron_type, Direct):
            # This will prevent users from connecting/probing Direct neurons
            # (since there aren't actually any neurons being simulated).
            return 0
        return self.ensemble.n_neurons

    @property
    def size_out(self):
        if isinstance(self.ensemble.neuron_type, Direct):
            # This will prevent users from connecting/probing Direct neurons
            # (since there aren't actually any neurons being simulated).
            return 0
        return self.ensemble.n_neurons

    @property
    def probeable(self):
        return ('output', 'input') + self.ensemble.neuron_type.probeable
