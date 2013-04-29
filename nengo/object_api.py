"""
object_api.py

This file provides objects for describing neural models. These objects
represent a high-level description language describing, for example, the
number and type of neurons, their arrangement into populations, and the
connections between them. The classes in this file do not actually implement a
simulator for the model they describe. In particular, this file does not
import numpy, and the objects described here do not have array attributes
describing e.g.  connection weights or membrane potentials.

Simulators for models described by the objects here can be found elsewhere,
such as:
    object_api_python.py: a pure-python implementation
    object_api_numpy.py: a numpy-based implementation

"""
import math
import random

try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        # -- Fall back on un-ordered dictionaries
        OrderedDict = dict

class SelfDependencyError(Exception):
    """Network cannot be simulated because some node input depends on the
    node's own output on the same time-step."""


class MultipleSourceError(Exception):
    """A variable at time t cannot be the output of multiple nodes."""


class Distribution(object):
    """
    Base class for random distributions (e.g. Uniform, Gaussian)
    """
    @property
    def dist_name(self):
        return self.__class__.__name__.lower()


class Uniform(Distribution):
    """Specification of a uniformly-distributed random number stream
    """
    def __init__(self, low, high, seed=None, generator=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.generator = generator


class Gaussian(Distribution):
    """Specification of a Gaussian-distributed random number stream
    """
    def __init__(self, mean, std, seed=None, generator=None):
        self.mean = mean
        self.std = std
        self.seed = seed
        self.generator = generator

#
# Top-level objects and containers
#

class Var(object):
    def __init__(self, name=None, size=None, dtype=float, shape=None):
        """
        Var represents a dynamic model element (e.g. membrane potential,
        adaptive weights), which is generally a multi-dimensional array of
        numbers at each time step of model simulation.

        The inputs and outputs of a model component (Node) are instances of either
        * Var (this class)
        * DelayedVar (created by this class's .delayed() function)

        Var refers to a dynamic model element at the *current* timestep.
        DelayedVar refers to a dynamic model element at the *previous* timestep.

        Parameters
        ----------

        string: identifier, not necessarily unique

        int: number of elements

        string: nature of numbers

        tuple of int: logical shape of array (optional)
        """
        self.name = name
        self.size = size
        self.dtype = dtype
        self.shape = shape
        if shape is None:
            if size is None:
                raise TypeError('size arg is required')
        else:
            _size = reduce(lambda a, b: a * b, shape, 1)
            if size is not None and _size != size:
                raise ValueError('shape and size are inconsistent')
            self.size = _size

    def __str__(self):
        clsname = self.__class__.__name__
        if self.name:
            return '%s{%s}' % (clsname, self.name)
        else:
            return '%s{%s}' % (clsname, id(self))

    def __repr__(self):
        return str(self)

    def delayed(self, delay=1):
        return DelayedVar(self, delay)


class DelayedVar(object):
    """
    DelayedVar stands for a dynamic model quantity (see Var) but from the
    previous simulator time step, rather than the current time step.
    """

    def __init__(self, var, delay):
        self.var = var
        self.delay = delay
        if self.delay != 1:
            raise NotImplementedError('do we need this case?')

    @property
    def name(self):
        return "%s(delay=%i)" % (self.var.name, self.delay)

    @property
    def size(self):
        return self.var.size

    @property
    def dtype(self):
        return self.var.dtype

    @property
    def shape(self):
        return self.var.shape


class Node(object):
    """
    A component of a brain model (e.g. Connection, Filter, Population)

    It is generally a function from self.inputs -> self.outputs, which are
    OrderedDict containers of Var instances.

    The inputs may contain DelayedVar instances, but the outputs may not.
    """
    def __init__(self):
        self.outputs = OrderedDict()
        self.inputs = OrderedDict()

    @property
    def output(self):
        return self.outputs['X']

    def add_to_network(self, network):
        network.nodes.append(self)


class Probe(Node):
    """
    Probe: a marker that some Var (target) should be recorded during simulation.
    """
    def __init__(self, target):
        Node.__init__(self)
        self.inputs['target'] = target

    @property
    def target(self):
        return self.inputs['target']

    def add_to_network(self, network):
        network.probes.append(self)


class Filter(Node):
    """
    TODO: rename this to ExponentialFilter, it is not a general base class.
    """
    tau=.02
    def __init__(self, var, tau=tau):
        """
        tau: float
        """
        Node.__init__(self)
        self.tau = tau
        self.outputs['X'] = Var(size=var.size)
        self.inputs['X_prev'] = self.outputs['X'].delayed()
        self.inputs['var'] = var

    def add_to_network(self, network):
        network.filters.append(self)


class Adder(Node):
    """
    This model component simply adds up its inputs to produce one output.
    """
    def __init__(self, size, args=()):
        Node.__init__(self)
        for ii, arg in enumerate(args):
            self.inputs[str(ii)] = arg
        self.outputs['X'] = Var(size=size)

    def add_to_network(self, network):
        network.filters.append(self)


class Connection(Node):
    """
    A Connection (Base instance) simply copies a signal from src -> dst
    """
    def __init__(self, src, dst):
        """
        Parameters
        ----------
        :param Var src:
        :param Var dst: 
        """
        Node.__init__(self)
        self.inputs['X'] = src
        self.outputs['X'] = dst

    def add_to_network(self, network):
        network.connections.append(self)

    @property
    def src(self):
        return self.inputs['X']

    @property
    def dst(self):
        return self.outputs['X']


class Network(object):
    """
    A Network is a container for the various Node instances that make up
    a neural model.  It can be used (a) as a container to to  keep track of
    Node instances, and (b) as the argument to configure a Simulator (see
    below).

    A Network can contain sub-networks (self.networks). The `all_*` properties
    recursively extract elements from those sub-networks, to provide a flattened
    view of the network.
    """
    def __init__(self):
        self.probes = []
        self.connections = []
        self.nodes = []
        self.networks = OrderedDict()
        self.filters = []

    def add(self, thing):
        thing.add_to_network(self)
        return thing

    @property
    def all_probes(self):
        if self.networks: raise NotImplementedError()
        return list(self.probes)

    @property
    def all_connections(self):
        if self.networks: raise NotImplementedError()
        return list(self.connections)

    @property
    def all_nodes(self):
        if self.networks: raise NotImplementedError()
        return list(self.nodes)

    @property
    def all_filters(self):
        if self.networks: raise NotImplementedError()
        return list(self.filters)

    @property
    def members(self):
        rval = []
        rval.extend(self.nodes)
        rval.extend(self.connections)
        rval.extend(self.probes)
        rval.extend(self.filters)
        return rval

    @property
    def all_members(self):
        rval = []
        rval.extend(self.all_nodes)
        rval.extend(self.all_connections)
        rval.extend(self.all_probes)
        rval.extend(self.all_filters)
        return rval


#
# Neuron Nodes
#


class Neurons(Node):
    """
    Neurons is a convenience base-class for subclasses that implement Neuron
    populations.
    """
    def __init__(self, size, input_current=None):
        """
        :param int size:
        :param Var input_current:
        """
        Node.__init__(self)
        self.size = size
        if input_current is None:
            input_current = Var(name='input_current', size=size)
        self._input_current = input_current
        self.inputs['input_current'] = input_current.delayed()
        self.outputs['X'] = Var(size=size)

    @property
    def input_current(self):
        # -- TODO setting this requires re-assigning the delayed view too
        return self._input_current


class LIFNeurons(Neurons):
    """
    LIFNeurons represents a population of LIF neurons.
    """
    # -- N.B. does not include input_current
    _input_names = ['alpha', 'j_bias', 'voltage', 'refractory_time']

    def __init__(self, size,
            input_current=None,
            tau_rc=0.02,
            tau_ref=0.002,
            max_rate=Uniform(200, 400),
            intercept=Uniform(-1, 1),
            seed=None):
        """
        Parameters
        ----------
        :param int size: number of neurons in this population
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        Neurons.__init__(self, size, input_current)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.max_rate = max_rate
        self.intercept = intercept
        self.seed = seed

        for name in self._input_names:
            self.outputs[name] = Var(name=name, size=size)
            self.inputs[name] = self.outputs[name].delayed()


    @property
    def alpha(self):
        return self.outputs['alpha']

    @property
    def j_bias(self):
        return self.outputs['j_bias']

    @property
    def voltage(self):
        return self.outputs['voltage']

    @property
    def refractory_time(self):
        return self.outputs['refractory_time']


#
# Function Nodes
#


class TimeNode(Node):
    """
    TimeNode represents a function of the simulator time.
    """
    def __init__(self, func, output=None, name=None):
        Node.__init__(self)
        self.func = func
        if output is None:
            # TODO: call func(0) to measure output?
            output = Var(shape=())
        self.outputs['X'] = output
        self.inputs['time'] = simulation_time
        if output.name is None:
            output.name = name
        self.name = name


class PiecewiseNode(Node):
    """
    PiecewiseNode represents a piecewise function of a scalar-valued input.
    """
    def __init__(self, table):
        """
        Parameters
        ----------
        table: Dictionary for piecewise lookup
        """
        self.table = table


#
# Connections
#

class RandomConnection(Connection):
    """
    RandomConnection represents a randomly-generated all-to-all linear signal
    transformation (aka random projection).
    """
    def __init__(self, src, dst, dist):
        Connection.__init__(self, src, dst)
        self.dist = dist
        self.outputs['weights'] = Var(size=src.size * dst.size,
                                      shape=(dst.size, src.size))
        self.inputs['weights'] = self.outputs['weights'].delayed()


class LearnedConnection(Connection):
    """
    LearnedConnection is a convenience base class for Connections that evolve
    in order to minimize a scalar-valued error signal.
    """
    def __init__(self, src, dst):
        Connection.__init__(self, src, dst)
        self.outputs['error_signal'] = Var(shape=())

    @property
    def error_signal(self):
        return self.outputs['error_signal']


class MSE_MinimizingConnection(LearnedConnection):
    """
    MSE_MinimizingConnection is a Connection that adapts to 
    minimize the mean-squared-error between its output and a target
    signal.
    """
    def __init__(self, src, dst, target, learning_rate=0.01):
        LearnedConnection.__init__(self, src, dst)
        self.inputs['target'] = target
        self.outputs['weights'] = Var(size=src.size * dst.size,
                                      shape=(dst.size, src.size))
        self.inputs['weights'] = self.outputs['weights'].delayed()
        self.learning_rate = learning_rate

    @property
    def target(self):
        return self.inputs['target']


class hPES_Connection(LearnedConnection):
    """
    hPES_Connection is a Connection that adapts to 
    minimize something that I don't yet understand (XXX).
    """
    theta_tau = 0.02
    unsupervised_rate_factor = 10.
    supervision_ratio = 1.0
    def __init__(self, src, dst, error_signal,
                 theta_tau=theta_tau,
                 unsupervised_rate_factor=unsupervised_rate_factor,
                 supervision_ratio=supervision_ratio,
                ):
        LearnedConnection.__init__(self, src, dst, error_signal)
        self.theta_tau = theta_tau
        self.unsupervised_rate_factor = unsupervised_rate_factor
        self.supervision_ratio = supervision_ratio

        self.seed = 123

        self.gains = Var()
        self.theta = Var()
        self.src_filtered = Var()
        self.dst_filtered = Var()
        self.weight_matrix = Var()
        self.supervised_learning_rate = Var()


#
# Simulator
#

simulation_time = Var('time', shape=())
"""simulation_time stands for the total elapsed time within a brain simulation
"""

simulation_stop_now = Var('stop_when', shape=())
"""simulation_stop_now can be the output of one model component (Node), and
if it is ever non-zero, the Simulator will finish the current time-step and
stop.
"""


class SimulatorBase(object):
    """ A Simulator evaluates a brain model, as expressed by a Network.
    It will simulate it by fixed time steps of size (dt) seconds.

    Calling run_steps will step the simulator forward from time 0, and return
    the values of all probed Vars in a dictionary of sequences.

    Calling reset() will move the simulator back to time 0, and prepare it to
    run forward again.  The result of run_steps after a reset should be
    identical to the first time that the simulator ran.
    """
    _backends = {}

    def __init__(self, network, dt):
        self.network = network
        self.dt = dt
        self.simulation_steps = 0

    def reset(self):
        self.simulation_steps = 0

    def run_steps(self, steps, dt):
        """
        Returns a dictionary mapping targets that have been probed
        to a list of either lists or arrays.
        """
        raise NotImplementedError('Use a simulator subclass')

    @property
    def simulation_time(self):
        return self.simulation_steps * self.dt

def Simulator(*args, **kwargs):
    """
    Factory method for creating a simulator.

    """
    backend = kwargs.pop('backend', 'reference')
    if backend not in SimulatorBase._backends:
        if backend == 'reference':
            import object_api_python
        elif backend == 'numpy':
            import object_api_numpy
        else:
            raise ValueError('backend "%s" not recognized, did you remember to'
                ' import the python module that implements that backend?' %
                backend, SimulatorBase._backends.keys())
    return SimulatorBase._backends[backend](*args, **kwargs)

