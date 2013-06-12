import random
import warnings

import numpy as np

from . import nonlinear as nl
from . import simulator_objects as so


class Uniform(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def sample(self, n):
        return [random.uniform(self.low, self.high) for _ in xrange(n)]

class Gaussian(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.std

    def sample(self, n):
        return [random.gauss(self.mean, self.std) for _ in xrange(n)]


class Network(object):
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

    Attributes
    ----------
    name : str
        The name of the ensemble (must be unique).
    metadata : dict
        An editable dictionary used to store miscellaneous information
        about this ensemble.
    properties : dict
        A read-only dictionary used to store miscellaneous information
        about this ensemble that is automatically generated.
    neurons : a Neuron model (see `nengo.nonlinear`)
        Information about the neurons in this ensemble.
    dimensions : int
        The number of dimensions represented by this ensemble.
    rates : vector of ``neurons`` floats
        The maximum firing rates of all of the neurons in this ensemble.
    intercepts : vector of ``neurons`` floats
        The x-intercepts of the tuning curves of all of the neurons
        in this ensemble
    encoders : 2D matrix of floats
        The encoding vectors of all of the neurons in this ensemble.
    seed : int
        The random seed used to generate this ensemble.
    noise : dict
        Information about the noise that will be injected into
        this ensemble; contains 'current', which is the amplitude of
        the current to inject, 'frequency', which is the sampling rate.

    """
    def __init__(self, name, neurons, dimensions,
                 radius=1.0, encoders=None,
                 max_rates=Uniform(50, 100), intercepts=Uniform(-1, 1),
                 mode='spiking', decoder_noise=None,
                 eval_points=None, noise=None, noise_frequency=None,
                 decoder_sign=None, seed=None):
        # Error for things not implemented yet or don't make sense
        if decoder_noise is not None:
            raise NotImplementedError('decoder_noise')
        if eval_points is not None:
            raise NotImplementedError('eval_points')
        if noise is not None or noise_frequency is not None:
            raise NotImplementedError('noise')
        if mode != 'spiking':
            raise NotImplementedError('mode')
        if decoder_sign is not None:
            raise NotImplementedError('decoder_sign')

        if isinstance(neurons, int):
            warnings.warn("neurons should be an instance of a nonlinearity, "
                          "not an int. Defaulting to LIF.")
            neurons = nl.LIF(neurons)

        # Warn if called with weird sets of arguments
        # if neurons.gain is not None and neurons.bias is None:
        #     warnings.warn("gain is set, but bias is not. Ignoring gain.")
        # if neurons.bias is not None and neurons.gain is None:
        #     warnings.warn("bias is set, but gain is not. Ignoring bias.")
        # if neurons.gain is not None and neurons.bias is not None:
        #     if max_rates != Uniform(50, 100):
        #         warnings.warn("gain and bias are set. Ignoring max_rates.")
        #     if intercepts != Uniform(-1, 1):
        #         warnings.warn("gain and bias are set. Ignoring intercepts.")

        # Look at arguments and expand those that need expanding
        if hasattr(max_rates, 'sample'):
            max_rates = max_rates.sample(neurons.n_neurons)
        if hasattr(intercepts, 'sample'):
            intercepts = intercepts.sample(neurons.n_neurons)

        # Store things on the ensemble that will be necessary for
        # later calculations or organization
        self.name = name
        self.radius = radius

        # The essential components of an ensemble are:
        #  self.sig - the signal (vector) being represented
        #  self.nl - the nonlinearity (neuron model) representing the signal
        #  self.enc - the encoders that map the signal into the population

        # Set up the signal
        self.sig = so.Signal(n=dimensions)

        # Set up the neurons
        neurons.set_gain_bias(max_rates, intercepts)
        self.nl = neurons

        # Set up the encoders
        self.enc = so.Encoder(self.sig, self.nl, encoders)

    def __str__(self):
        return ("Ensemble (id " + str(id(self)) + "): \n"
                "    " + str(self.nl) + "\n"
                "    " + str(self.sig) + "\n"
                "    " + str(self.enc))

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.nonlinearity(self.nl)
        model.signals.add(self.sig)
        model.encoders.add(self.enc)
        model.signals.add(self.enc.weights_signal)

    @property
    def dimensions(self):
        return self.sig.n

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
        if callable(output):
            self.sig = so.Signal()
            self.nl = nl.Direct(n_in=1, n_out=1, fn=output)
            self.enc = so.Encoder(input, self.nl, weights=np.asarray([[1]]))
            self.nl.input_signal.name = name + '.input'
            self.nl.bias_signal.name = name + '.bias'
            self.nl.output_signal.name = name + '.output'
        else:
            if type(output) == list:
                self.sig = so.Constant(n=len(output),
                                       value=[float(n) for n in output])
            else:
                self.sig = so.Constant(n=1, value=float(output))

    def __str__(self):
        if hasattr(self, 'nl'):
            return ("Function node (id " + str(id(self)) + "): \n"
                    "    " + str(self.nl) + "\n"
                    "    " + str(self.sig) + "\n"
                    "    " + str(self.enc))
        else:
            return ("Constant node (id " + str(id(self)) + "):  \n"
                    "    " + str(self.sig))

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        if hasattr(self, 'nl'):
            model.nonlinearity(self.nl)
        if hasattr(self, 'enc'):
            model.encoders.add(self.enc)
            model.signals.add(self.enc.weights_signal)
        model.signals.add(self.sig)


class Connection(object):
    """Describes a connection between two Nengo objects.

    The connection encapsulates a lot of information that Nengo needs
    to compute a biologically plausible connection between two networks
    that implements some mathematical function.
    Alternatively, the connection could bypass this logic and just store
    a set of connection weights between two Ensembles.

    Attributes
    ----------
    pre : Nengo object
        The Nengo object on the presynaptic side of this connection.
    post : Nengo object
        The Nengo object on the postsynaptic side of this connection.
    transform : 2D matrix of floats
        If the connection operates in vector (state) space,
        ``transform`` is a two-dimensional array of floats
        that represents the linear transformation
        between ``pre`` and ``post``.
    weights : 2D matrix of floats
        If the connection operates in neuron space,
        ``weights`` is a two-dimensional array of floats
        that represents the connection weights
        between ``pre`` neurons and ``post`` neurons.
    decoders : 2D matrix of floats
        If the connection operates in vector space,
        it will have a set of decoders defined that
        maps the neural activity to a vector representation.
    filter : dict
        A dictionary describing the filter that is applied to
        presynaptic spikes before being communicated to ``post``.
    function : function
        The function that this connection implements.
    learning_rule : dict
        A dictionary describing a learning rule that
        modifies connection's decoders, weights,
        or both during a simulation.
    modulatory : bool
        A boolean indicating if the connection is modulatory.

        Modulatory connections do not impart current in ``post``.
        Instead, it can be used by ``post`` to do other operations
        (e.g., modulate learning).

    See Also
    --------
    Model.connect : Helper to make connections
    Model.connect_neurons : Helper to make direct connections

    """

    def __init__(self, pre, post, transform=1.0, weights=None, decoders=None,
                 filter=None, function=None, learning_rule=None,
                 modulatory=False):
        if weights is not None:
            raise NotImplementedError()
        if decoders is not None:
            raise NotImplementedError()
        if filter is not None:
            raise NotImplementedError()
        if learning_rule is not None:
            raise NotImplementedError()

        # if function is None:
        #     function = lambda x: x

        self.pre = pre
        self.post = post

        if isinstance(self.pre, Ensemble):
            self.decoder = so.Decoder(self.pre.nl, self.pre.sig)
            self.decoder.desired_function = function
            self.transform = so.Transform(np.asarray(transform),
                                          self.pre.sig,
                                          self.post.sig)

        elif isinstance(self.pre, Node):
            if function is None:
                self.transform = so.Transform(np.asarray(transform),
                                              self.pre.sig,
                                              self.post.sig)
                # alpha = np.asarray(transform)
                # if alpha.size == 1:
                #     self.transform = so.Transform(alpha, self.pre.sig,
                #                                   self.post.sig)
                # else:
                #     raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __str__(self):
        ret = "Connection (id " + str(id(self)) + "): \n"
        if hasattr(self, 'decoder'):
            return ret + ("    " + str(self.decoder) + "\n"
                          "    " + str(self.transform))
        else:
            return ret + "    " + str(self.transform)

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self.pre.name + ">" + self.post.name

    def add_to_model(self, model):
        if hasattr(self, 'decoder'):
            model.decoders.add(self.decoder)
            model.signals.add(self.decoder.weights_signal)
        model.transforms.add(self.transform)
        model.signals.add(self.transform.alpha_signal)
