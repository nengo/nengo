import collections
import inspect
import logging
import numpy as np

import nengo
import nengo.decoders
from nengo.nonlinearities import Neurons

logger = logging.getLogger(__name__)


def _in_stack(function):
    """Check whether the given function is in the call stack"""
    # TODO: Move to a generic utilities file.
    codes = [record[0].f_code for record in inspect.stack()]
    return function.__code__ in codes


class NengoObject(object):

    def __init__(self, uid=None, add_to_model=True):
        if add_to_model:
            nengo.context.add_to_current(self)

    def to_dict(self):
        d = {}
        for key in sorted(self.__dict__):
            if not key.startswith("_"):
                value = self.__dict__[key]
                if isinstance(value, NengoObject):
                    value = value.to_dict()
                d[key] = value
        return d

    def add_to_model(self, model):
        raise NotImplemented("Nengo objects must implement add_to_model.")


class Sampler(NengoObject):

    def __init__(self):
        super(Sampler, self).__init__(add_to_model=False)

    def __eq__(self, other):
        if not isinstance(other, Sampler):
            raise ValueError("Cannot compare type '%s' with '%s'." % (
                self.__class__.__name__, other.__class__.__name__))
        return self.__dict__ == other.__dict__

    def sample(self, *args, **kwargs):
        raise NotImplemented("sample function not implemented.")


class Uniform(Sampler):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        super(Uniform, self).__init__()

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.uniform(low=self.low, high=self.high, size=n)


class Gaussian(Sampler):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super(Gaussian, self).__init__()

    def sample(self, n, rng=None):
        rng = np.random if rng is None else rng
        return rng.normal(loc=self.mean, scale=self.std, size=n)


class PythonFunction(NengoObject):

    def __init__(self, fn, n_in, n_out, label=None):
        self.fn = fn
        self.n_in = n_in
        self.n_out = n_out
        if label is None:
            label = "<Direct%d>" % id(self)
        self.label = label
        super(PythonFunction, self).__init__(add_to_model=False)

    @property
    def n_args(self):
        return 2 if self.n_in > 0 else 1


class Neurons(NengoObject):

    def __init__(self, n_neurons, label=None):
        self.n_neurons = n_neurons
        if label is None:
            label = "<%s%d>" % (self.__class__.__name__, id(self))
        self.label = label
        self.probes = {"output": []}
        super(Neurons, self).__init__(add_to_model=False)

    def __str__(self):
        return "%s(%s, %dN)" % (
            self.__class__.__name__, self.label, self.n_neurons)

    def __repr__(self):
        return str(self)

    def rates(self, x):
        raise NotImplementedError("Neurons must provide rates.")

    def probe(self, probe, **kwargs):
        if probe.attr == "output":
            nengo.Connection(self, probe, filter=probe.filter, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable." % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe


class Direct(Neurons):

    def __init__(self, n_neurons=None, label=None):
        # n_neurons is ignored, but accepted to maintain compatibility
        # with other neuron types
        super(Direct, self).__init__(0, label=label)

    def rates(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        return None, None

# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class _LIFBase(Neurons):

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, label=None):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        super(_LIFBase, self).__init__(n_neurons, label=label)

    def rates_from_current(self, J):
        """LIF firing rates in Hz for input current (incl. bias)"""
        old = np.seterr(divide="ignore", invalid="ignore")
        try:
            j = J - 1    # because we're using log1p instead of log
            r = 1. / (self.tau_ref + self.tau_rc * np.log1p(1. / j))
            # NOTE: There is a known bug in numpy that np.log1p(inf) returns
            #   NaN instead of inf: https://github.com/numpy/numpy/issues/4225
            r[j <= 0] = 0
        finally:
            np.seterr(**old)
        return r

    def rates(self, x, gain, bias):
        """LIF firing rates in Hz for vector space

        Parameters
        ---------
        x: ndarray of any shape
            vector-space inputs
        """
        J = gain * x + bias
        return self.rates_from_current(J)

    def gain_bias(self, max_rates, intercepts):
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
        logging.debug("Setting gain and bias on %s", self.label)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        gain = (1 - x) / (intercepts - 1.0)
        bias = 1 - gain * intercepts
        return gain, bias


class LIFRate(_LIFBase):

    def math(self, dt, J):
        """Compute rates for input current (incl. bias)"""
        return dt * self.rates_from_current(J)


class LIF(_LIFBase):

    def __init__(self, n_neurons, upsample=1, **kwargs):
        self.upsample = upsample
        super(LIF, self).__init__(n_neurons, **kwargs)

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # update voltage using Euler's method
        dV = (dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < 0] = 0  # clip values below zero

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike (if v > 1 set spiked = 1, else 0)
        spiked[:] = (voltage > 1)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - 1) / dV[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0
        refractory_time[spiked > 0] = self.tau_ref + spiketime


class Ensemble(NengoObject):
    """A group of neurons that collectively represent a vector.

    Parameters
    ----------
    name : str
        An arbitrary name for the new ensemble.
    neurons : Neurons
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

    def __init__(self, neurons, dimensions, radius=1.0, encoders=None,
                 intercepts=Uniform(-1.0, 1.0), max_rates=Uniform(200, 400),
                 eval_points=None, seed=None, label="Ensemble"):
        if dimensions <= 0:
            raise ValueError(
                "Number of dimensions (%d) must be positive." % dimensions)

        self.dimensions = dimensions  # Must be set before neurons
        self.neurons = neurons
        self.radius = radius
        self.encoders = encoders
        self.intercepts = intercepts
        self.max_rates = max_rates
        self.label = label
        self.eval_points = eval_points
        self.seed = seed

        # Set up probes
        self.probes = {"decoded_output": [], "spikes": [], "voltages": []}
        super(Ensemble, self).__init__()

    def __str__(self):
        return "Ensemble: %s" % self.label

    def probe(self, probe, **kwargs):
        if probe.attr == "decoded_output":
            Connection(self, probe, filter=probe.filter, **kwargs)
        elif probe.attr == "spikes":
            Connection(self.neurons, probe, filter=probe.filter,
                       transform=np.eye(self.neurons.n_neurons), **kwargs)
        elif probe.attr == "voltages":
            Connection(self.neurons.voltage, probe, filter=None, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable." % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe

    def add_to_model(self, model):
        model.ensembles.append(self)


class Node(NengoObject):
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
    output : callable or array_like
        Function that transforms the Node inputs into outputs, or
        a constant output value. If output is None, this Node simply acts
        as a placeholder and will be optimized out of the final model.
    size_in : int, optional
        The number of input dimensions.
    size_out : int, optional
        The size of the output signal

    Attributes
    ----------
    name : str
        The name of the object.
    size_in : int
        The number of input dimensions.
    """

    def __init__(self, output=None, size_in=0, size_out=None, label="Node"):
        if output is not None and not isinstance(output, collections.Callable):
            output = np.asarray(output)
        self.output = output
        self.label = label
        self.size_in = size_in

        if size_out is None:
            if isinstance(output, collections.Callable):
                t, x = np.asarray(0.0), np.zeros(size_in)
                args = [t, x] if size_in > 0 else [t]
                try:
                    result = output(*args)
                except TypeError:
                    raise TypeError(
                        "The function '%s' provided to '%s' takes %d "
                        "argument(s), where a function for this type "
                        "of node is expected to take %d argument(s)." % (
                        output.__name__,
                        self,
                        output.__code__.co_argcount,
                        len(args)))
                size_out = np.asarray(result).size
            elif isinstance(output, np.ndarray):
                size_out = output.size
        self.size_out = size_out

        # Set up probes
        self.probes = {"output": []}
        super(Node, self).__init__()

    def __str__(self):
        return "Node: " + self.label

    def probe(self, probe, **kwargs):
        if probe.attr == "output":
            Connection(self, probe, filter=probe.filter, **kwargs)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable." % probe.attr)

        self.probes[probe.attr].append(probe)
        return probe

    def add_to_model(self, model):
        model.nodes.append(self)


class Connection(NengoObject):
    """Connects two objects together.

    Attributes
    ----------
    pre : Ensemble or Neurons or Node
        The source object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    label : string
        A descriptive label for the connection.
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but not including `transform`.
    decoder_solver : callable
        Function to compute decoders (see `nengo.decoders`).
    eval_points : array_like, shape (n_eval_points, pre_size)
        Points at which to evaluate `function` when computing decoders.
    filter : float
        Post-synaptic time constant (PSTC) to use for filtering.
    function : callable
        Function to compute using the pre population (pre must be Ensemble).
    probes : dict
        description TODO
    transform : array_like, shape (post_size, pre_size)
        Linear transform mapping the pre output to the post input.
    """

    _decoders = None
    _eval_points = None
    _function = (None, 0)  # (handle, n_outputs)
    _transform = None

    def __init__(self, pre, post,
                 filter=0.005, transform=1.0, modulatory=False, **kwargs):
        self.pre = pre
        self.post = post

        self.filter = filter
        self.transform = transform
        self.modulatory = modulatory

        if isinstance(self.pre, Ensemble):
            self.decoder_solver = kwargs.pop(
                "decoder_solver", nengo.decoders.lstsq_L2)
            self.eval_points = kwargs.pop("eval_points", None)
            self.function = kwargs.pop("function", None)
        elif not isinstance(self.pre, (Neurons, Node)):
            raise ValueError("Objects of type '%s' cannot serve as 'pre'." %
                               self.pre.__class__.__name__)

        # Check that we've used all user-provided arguments
        if len(kwargs) > 0:
            raise TypeError("__init__() got an unexpected keyword argument "
                              "'%s'." % next(iter(kwargs)))

        # Check that shapes match up
        self._check_shapes(check_in_init=True)

        # Set up probes
        self.probes = {"signal": []}
        super(Connection, self).__init__()

    def _check_pre_ensemble(self, prop_name):
        if not isinstance(self.pre, Ensemble):
            raise ValueError("'%s' can only be set if 'pre' is an Ensemble." %
                             (prop_name))

    def _check_shapes(self, check_in_init=False):
        if not check_in_init and _in_stack(self.__init__):
            return  # skip automatic checks if we're in the init function

        in_dims, in_src = self._get_input_dimensions()
        out_dims, out_src = self._get_output_dimensions()

        if self.transform.ndim == 0:
            # check input dimensionality matches output dimensionality
            if (in_dims is not None and out_dims is not None
                    and in_dims != out_dims):
                raise ValueError("%s output size (%d) not equal to "
                                 "post %s (%d)." %
                                 (in_src, in_dims, out_src, out_dims))
        else:
            # check input dimensionality matches transform
            if in_dims is not None and in_dims != self.transform.shape[1]:
                raise ValueError("%s output size (%d) not equal to "
                                 "transform input size (%d)." %
                                 (in_src, in_dims, self.transform.shape[1]))

            # check output dimensionality matches transform
            if out_dims is not None and out_dims != self.transform.shape[0]:
                raise ValueError("Transform output size (%d) not equal to "
                                 "post %s (%d)." %
                                 (self.transform.shape[0], out_src, out_dims))

    def _get_input_dimensions(self):
        if isinstance(self.pre, Ensemble):
            if self.function is not None:
                dims, src = self._function[1], "Function"
            else:
                dims, src = self.pre.dimensions, "Pre population"
        elif isinstance(self.pre, Neurons):
            dims, src = self.pre.n_neurons, "Neurons"
        elif isinstance(self.pre, Node):
            dims, src = self.pre.size_out, "Node"
        return dims, src

    def _get_output_dimensions(self):
        if isinstance(self.post, Ensemble):
            dims, src = self.post.dimensions, "population dimensions"
        elif isinstance(self.post, Neurons):
            dims, src = self.post.n_neurons, "number of neurons"
        elif isinstance(self.post, Node):
            dims, src = self.post.size_in, "node input size"
        else:
            dims, src = None, str(self.post)
        return dims, src

    def __str__(self):
        return self.label + " (" + self.__class__.__name__ + ")"

    def __repr__(self):
        return str(self)

    @property
    def label(self):
        label = self.pre.label + ">" + self.post.label
        if self.function is not None:
            return label + ":" + self.function.__name__
        return label

    @property
    def dimensions(self):
        return self._get_input_dimensions()[0]

    @property
    def function(self):
        return self._function[0]

    @function.setter
    def function(self, _function):
        if _function is not None:
            self._check_pre_ensemble("function")
            x = (self._eval_points[0] if self._eval_points is not None else
                 np.zeros(self.pre.dimensions))
            size = np.asarray(_function(x)).size
        else:
            size = 0

        self._function = (_function, size)
        self._check_shapes()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = np.asarray(_transform)
        self._check_shapes()

    def add_to_model(self, model):
        model.connections.append(self)


class Probe(NengoObject):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    Parameters
    ----------
    name : str
        An arbitrary name for the object.
    dt : float
        Sampling period in seconds.
    """

    DEFAULTS = {
        Ensemble: "decoded_output",
        Node: "output",
    }

    def __init__(self, target, attr=None, dt=None, filter=None, **kwargs):
        self.target = target
        if attr is None:
            try:
                attr = self.DEFAULTS[target.__class__]
            except KeyError:
                for k in self.DEFAULTS:
                    if issubclass(target.__class__, k):
                        attr = self.DEFAULTS[k]
                        break
                else:
                    raise TypeError("Type %s has no default probe." %
                                      target.__class__.__name__)
        self.attr = attr
        self.label = "Probe(%s.%s)" % (target.label, attr)
        self.dt = dt
        self.filter = filter

        target.probe(self, **kwargs)
        super(Probe, self).__init__(add_to_model=False)


class Model(NengoObject):
    """A model contains ensembles, nodes, connections, probes, and other models.

    # TODO: Example usage.

    Parameters
    ----------
    label : basestring
        Name of the model.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.

    Attributes
    ----------
    label : basestring
        Name of the model
    seed : int
        Random seed used by the model.
    """

    def __init__(self, *args, **kwargs):
        label = kwargs.pop("label", "Model")
        seed = kwargs.pop("seed", None)

        if not isinstance(label, basestring):
            raise ValueError("Label '%s' must be str or unicode." % label)

        if not len(nengo.context):
            # Make this the default context
            nengo.context.append(self)
            is_root = True
        else:
            is_root = False

        self.label = label
        self.seed = seed
        self.ensembles = []
        self.nodes = []
        self.connections = []
        self.models = []

        with self:
            self.make(*args, **kwargs)

        super(Model, self).__init__(add_to_model=not is_root)

    def make(self, *args, **kwargs):
        return

    def add_to_model(self, model):
        model.models.append(self)

    def add(self, obj):
        """Adds a Nengo object to this model.

        This is generally only used for manually created nodes, not ones
        created by calling :func:`nef.Model.make_ensemble()` or
        :func:`nef.Model.make_node()`, as these are automatically added.
        A common usage is with user created subclasses, as in the following::

          node = net.add(MyNode('name'))

        Parameters
        ----------
        obj : Nengo object
            The Nengo object to add.

        Returns
        -------
        obj : Nengo object
            The Nengo object that was added.
        """
        if not isinstance(obj, NengoObject):
            raise ValueError("Object of type '%s' is not a NengoObject." %
                               obj.__class__.__name__)
        obj.add_to_model(self)
        return obj

    def __enter__(self):
        nengo.context.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()
