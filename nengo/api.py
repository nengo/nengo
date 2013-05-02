import math
import random
from collections import namedtuple

Uniform = namedtuple('Uniform', ['low', 'high'])
Gaussian = namedtuple('Gaussian', ['mean', 'std'])

def psc(psc=0.01):
    return {'type': 'ExponentialPSC', 'psc': psc}

def lif(tau_rc=0.02, tau_ref=0.002):
    return {'type': 'LIFNeuron', 'tau_rc': tau_rc, 'tau_ref': tau_ref}

def hpes(rate=5e-7, ratio=0.8):
    return {'type': 'HPESLearningRule', 'rate': rate, 'ratio': ratio}


class Var(object):
    def __init__(self, name=None, size=None, dtype=float, shape=None):
        """
        Var represents a dynamic model element (e.g. membrane potential,
        adaptive weights), which is generally a multi-dimensional array of
        numbers at each time step of model simulation.

        The inputs and outputs of a Nengo object are instances of either
        * Var (this class)
        * DelayedVar (created by this class's .delayed() function)

        Var refers to a dynamic model element at the *current* timestep.
        DelayedVar refers to a dynamic model element at the *previous* timestep.

        Parameters
        ----------
        name : str
            Identifier, not necessarily unique.
        size : int
            Number of elements.
        dtype : str or type
            Data type of the numbers.
        
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

        
class Model(object):
    """A model contains a single network and the ability to
    run simulations of that network.

    Model is the first part of the API that modelers
    become familiar with, and it is possible to create
    many of the models that one would want to create simply
    by making a model and calling functions on that model.

    For example, a model that implements a communication channel
    between two ensembles of neurons can be created with::

        import nengo
        model = nengo.Model("Communication channel")
        input = model.make_node("Input", values=[0])
        pre = model.make_ensemble("In", neurons=100, dimensions=1)
        post = model.make_ensemble("Out", neurons=100, dimensions=1)
        model.connect(input, pre)
        model.connect(pre, post)

    Parameters
    ----------
    name : str
        Name of the model.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.
    fixed_seed : int, optional
        Random number seed that will be fed to the random number generator
        before each random process. Unlike setting ``seed``,
        each new ensemble in the network will use ``fixed_seed``,
        meaning that ensembles with the same properties will have the same
        set of neurons generated.
    backend : str, optional
        The backend that this model should use.

        If you have installed a Nengo backend, such as the Theano backend,
        then pass in the appropriate string to use that backend for this model.

        **Default**: ``'numpy'``, the Python reference implementation.
        

    Attributes
    ----------
    name : str
        Name of the model
    seed : int
        Random seed used by the model.
    backend : str
        The backend that is implementing this model.
    time : float
        The amount of time that this model has been simulated.
    metadata : dict
        An editable dictionary that modelers can use to store
        extra information about a network.
    properties : read-only dict
        A collection of basic information about
        a network (e.g., number of neurons, number of synapses, etc.)

    """

    def __init__(self, name, seed=None, fixed_seed=None,
                 backend='numpy'):
        self.network = Network()
        self.name = name
        self.backend = backend
        self.byname = {}
        self.simulator = None
        if seed is None:
            self.seed = 123
        else:
            self.seed = seed
        if fixed_seed is not None:
            raise NotImplementedError()

    @property
    def time(self):
        return self.simulator.simulator_time

    ### Simulation methods

    def build(self, dt=0.001):
        """Builds an internal representation of the model.

        The API makes few claims about how the API calls
        are represented internally. These decisions are left
        up to the backends that implement the API.
        Generally, a backend only has to implement this
        function, and :func:`nengo.Model.run()`.

        Parameters
        ----------
        dt : float, optional
            The length, in seconds, of each timestep in the simulation.

            ``build()`` needs this because the build process often needs
            to simulate the model in order to collect information
            about the neurons used in the model in order to decide how
            to best connect them.

            **Default**: 0.001; i.e., 1 millisecond

        """
        self.simulator = Simulator(self.network, dt=dt)
    
    def reset(self):
        """Reset the state of the simulation.

        Runs through all nodes, then ensembles, then connections and then
        probes in the network and calls thier reset functions.

        """
        self.simulator.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        """Runs a simulation of the model.

        Parameters
        ----------
        time : float
            How long to run the simulation, in seconds.

            If called more than once, successive calls will continue
            the simulation for ``time`` more seconds.
            To reset the simulation, call :func:`nengo.Model.reset()`.
            Typical use cases are to either simply call it once::

              model.run(10)

            or to call it multiple times in a row::

              time = 0
              dt = 0.1
              while time < 10:
                  model.run(dt)
                  time += dt
        dt : float, optional
            The length of a timestep, in seconds.

            **Default**: 0.001
        output : str or None, optional
            Where probed data should be output.

            If ``output`` is None, then probed data will be returned
            by this function as a dictionary.

            If ``output`` is a string, it is interpreted as a path,
            and probed data will be written to that file.
            The file extension will be parsed to determine the type
            of file to write; any unrecognized extension
            will be ignored and a comma-separated value file will
            be created.

            **Default**: None, so this function returns a dictionary
            of probed data.

        Returns
        -------
        data : dictionary
            All of the probed data. This is only returned if
            ``output`` is None.

        """
        if self.simulator is None:
            self.simulator = API.Simulator(self.network, dt=dt,
                                           backend=self.backend_type)
        assert dt == self.simulator.dt
        if stop_when is not None:
            raise NotImplementedError()
        if output is not None:
            raise NotImplementedError()
        steps = int(time / dt)
        self.simulator.run_steps(steps )

    ### Model creation methods
        
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

        See Also
        --------
        Network.add : The same function for Networks

        """
        return self.network.add(obj)
        
    def get(self, target, default=None):
        """Return the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The ``target`` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a Nengo object is passed, ``get`` just confirms
            that ``target`` is a part of the model.

        default : optional
            If ``target`` is not in the model, then ``get`` will
            return ``default``.
            If ``default`` is not specified and ``target`` does not
            exist, a ``ValueError`` is raised.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        Raises
        ------
        ValueError
            If the ``target`` does not exist and no ``default`` is specified.

        See Also
        --------
        Network.get : The same function for Networks

        """
        return self.network.get(target, default)

    def make_alias(self, alias, target):
        """Adds a named shortcut to an existing Nengo object
        within this model.

        This is designed to simplify :func:`nengo.Model.connect()`,
        :func:`nengo.Model.get()`, and :func:`nengo.Model.remove()` calls.
        For example, you can do::

            model.make_alias('vision', 'A.B.C.D.E')
            model.make_alias('motor', 'W.X.Y.Z')
            model.connect('vision', 'motor')

        Parameters
        ----------
        alias : str
            The alias to assign to ``target``.
        target : str or Nengo object
            Identifies the Nengo object to be aliased.

        Raises
        ------
        ValueError
            If ``target`` can't be found in the model.
        
        See Also
        --------
        Network.make_alias : The same function for Networks

        """
        self.network.make_alias(alias, target)

    def make_ensemble(self, name, neurons, dimensions, max_rate=(50, 100),
                      intercept=(-1, 1), radius=1.0, encoders=None,
                      neuron_model=lif(), mode='spiking'):
        """Create and return an ensemble of neurons.

        The ensemble created by this function is automatically added to
        the model.

        Parameters
        ----------
        name : str
            Name of the ensemble. Must be unique within the model.
        neurons : int
            Number of neurons in the ensemble.
        dimensions : int
            Number of dimensions that this ensemble will represent.
        max_rate : iterable, optional
            A 2-element iterable containing the minimum and maximum
            values of a uniform distribution from which the maximum
            firing rates of neurons in the ensemble will be selected
            (in Hz).

            **Default**: (50, 100)
        intercept : iterable, optional
            A 2-element iterable containing the minimum and maximum
            values of a uniform distribution from which the x-intercepts
            of neuron tuning curves will be selected.

            **Default**: (-1, 1)
        radius : float, optional
            The representational range of the ensemble.
            I.e., the maximum value that can be represented
            in each dimension.

            **Default**: 1.0
        encoders : 2-D matrix of floats, optional
            A matrix containing encoding vectors for each neuron.

            **Default**: randomly generated vectors on the unit sphere.
        neuron_model : dict, optional
            Specifies the neuron model that this ensemble will
            be made up of.

            **Default**: A leaky integrate-and-fire (LIF) neuron
            with ``tau_rc=0.02``, ``tau_ref=0.002``.
        mode : {'spiking', 'direct', 'rate'}, optional
            Simulation mode.

            **Default**: 'spiking'

        See Also
        --------
        Network.make_ensemble : The same function for Networks

        """
        ens = self.byname[name] = Ensemble(name, neurons, dimensions,
                                           max_rate=max_rate,
                                           intercept=intercept,
                                           radius=radius,
                                           encoders=encoders,
                                           neuron_model=neuron_model,
                                           mode=mode,
                                           seed=self.seed,
                                          )
        self.network.add(ens._neurons)  # ???
        self.network.add(ens._encoder)  # ???
        self.network.add(ens._adder)    # ???
        self.seed += 101
        return ens

    def make_network(self, name, seed=None):
        """Create and return a network.

        Networks can contain other networks, and are useful
        for organizing ensembles and connections.
        This function creates a new network, which can then be used to
        create ensembles and other Nengo objects that are within that
        network.

        Parameters
        ----------
        name : str
            Name of the network. This must be unique within the model.

        Returns
        -------
        network : Network
            The created network.
        seed : int, optional
            Random number seed that will be fed to the
            random number generator. Setting this seed makes
            the creation of the network a deterministic process;
            however, each new ensemble in the network advances
            the random number generator, so if the network creation code
            changes, the entire network changes.

        See Also
        --------
        Network.make_network : The same function for Networks

        """
        return self.network.add_network(name, seed)

    def make_node(self, name, output):
        """Create and return a node of dimensionality ``len(output)``,
        which produces the defined output.

        Parameters
        ----------
        name : str
            Name of this node. Must be unique in the model.
        output : function, list of floats, dict
            The output that should be generated by this node.

            If ``output`` is a function, it will be called on each timestep;
            if it accepts a single parameter, it will be given
            the current time of the simulation.
            
            If ``output`` is a list of floats, that list will be
            used as constant output.

            If ``output`` is a dict, the output defines a piece-wise constant
            function in which the keys define when the value changes,
            and the values define what the value changes to.
        
        Returns
        -------
        node : Node
            The created Node

        See Also
        --------
        Network.make_node : The same function for Networks

        """
        return self.network.make_node(name, output)

    def remove(self, target):
        """Removes a Nengo object from the model.

        Parameters
        ----------
        target : str, Nengo object
            A string referencing the Nengo object to be removed
            (see `string reference <string_reference.html>`)
            or node or name of the node to be removed.

        Returns
        -------
        target : Nengo object
            The Nengo object removed.

        See Also
        --------
        Network.remove : The same function for Networks

        """
        self.network.remove(target)

    def probe(self, target, sample_every=None, static=False):
        """Probe a piece of data contained in the model.

        When a piece of data is probed, it will be recorded through
        the course of the simulation.

        Parameters
        ----------
        target : str, Nengo object
            The piece of data being probed.
            This can specified as a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object. Each Nengo object will emit
            what it considers to be the most useful piece of data
            by default; if that's not what you want,
            then specify the correct data using the string format.
        sample_every : float, optional
            How often to sample the target data, in seconds.

            Some types of data (e.g. connection weight matrices)
            are very large, and change relatively slowly.
            Use ``sample_every`` to limit the amount of data
            being recorded. For example::

              model.probe('A>B.weights', sample_every=0.5)

            records the value of the weight matrix between
            the ``A`` and ``B`` ensembles every 0.5 simulated seconds.

            **Default**: Every timestep (i.e., ``dt``).
        static : bool, optional
            Denotes if a piece of data does not change.

            Some data that you would want to know about the model
            does not change over the course of the simulation;
            this includes things like the properties of a model
            (e.g., number of neurons or connections) or the random seed
            associated with a model. In these cases, to record that data
            only once (for later being written to a file),
            set ``static`` to True.

            **Default**: False

        See Also
        --------
        Network.probe : The same function for Networks

        """
        obj = self.get(target)
        if isinstance(obj, Ensemble):
            return self.network.add(API.Probe(obj._adder.output))
        else:
            return self.network.add(API.Probe(obj))

    def connect(self, pre, post, function=None, transform=None,
                filter=psc(), learning_rule=None):
        """Connect ``pre`` to ``post``.

        Parameters
        ----------
        pre, post : str or Nengo object
            The items to connect.
            ``pre`` and ``post`` can be strings that identify a Nengo object
            (see `string reference <string_reference.html>`_), or they
            can be the Nengo objects themselves.

        function : Python function, optional
            The function that this connection will compute.
            This function takes as input the vector being represented by
            ``pre``, and returns another vector which will be
            projected to ``post``.
            If ``function`` is not specified, by default the
            identity function will be used (i.e., the function returns
            the same vector that it takes as input;
            :math:`f(\mathbf{x}) = \mathbf{x}`).
            The function takes a single parameter ``x``,
            which is the current value of the ``pre`` ensemble,
            and must return a float (for one-dimensional functions) or
            a list of floats.

            The following simple example connects two ensembles together
            such that the second ensemble represents
            the square of the first ensemble::

              def square(x):
                  return x * x
              pre = net.make_ensemble('pre', neurons=30, dimensions=1)
              post = net.make_ensemble('post', neurons=30, dimensions=1)
              net.connect(pre, post, function=square)

            or, slightly more succinctly::

              net.connect(pre, post, function=lambda x: x * x)

            **Default**: the ``identity`` function
            (:math:`f(x) = x`).
              
        transform : float matrix (``function`` dims by ``post`` dims), optional
            A matrix that maps the computed function onto ``post``.
            Its dimensionality is ``function`` output dimensions
            by ``post`` dimensions. If ``transform`` is not specified,
            the identity matrix will be used. This mainly makes sense
            when the dimensionality of the ``function`` output
            is exactly the dimensionality of ``post``; if this isn't true,
            then you should probably explicitly define ``transform``.

            The following simple example passes through the values
            represented by a 2-dimensional ensemble to
            the first and third dimension of a 4-dimensional ensemble::

              pre = model.make_ensemble('pre', neurons=40, dimensions=2)
              post = model.make_ensemble('post', neurons=80, dimensions=4)
              model.connect(pre, post, transform=[[1, 0], [0, 0], [0, 1], [0, 0]])

            The transform matrix is quite confusing to make manually;
            a helper function for making transforms is provided
            (see :func:`nengo.gen_transform()`).

            The following complex example computes the product of
            a 2-dimensional vector, and projects that to the second dimension
            of a 2-dimensional ensemble::

              def product(x):
                  return x[0] * x[1]
              pre = model.make_ensemble('pre', neurons=40, dimensions=2)
              post = model.make_ensemble('post', neurons=40, dimensions=2)
              model.connect(pre, post, func=product, transform=[[0], [1]])

            or, slightly more succinctly::

              model.connect(pre, post, func=lambda x: x[0] * x[1],
                            transform=[[0], [1]])

            **Default**: an identity matrix.

        filter : dict, optional
            ``filter`` contains information about the type of filter
            to use across this connection.

            **Default**: specifies an exponentially decaying filter
            with ``tau=0.01``.

        learning_rule : dict, optional
            ``learning_rule`` contains information about the type of
            learning rule that modifies this connection.

            **Default**: None

        Returns
        -------
        connection : Connection
            The Connection object created.

        See Also
        --------
        Connection : The Connection object
        Network.connect : The same function for Networks

        """
        return self.network.connect(
            pre, post, function, transform, filter, learning_rule)

    def connect_neurons(self, pre, post, weights,
                        filter=psc(), learning_rule=None):
        """Directly connect the neurons in the ``pre`` ensemble
        to the neurons in the ``post`` ensemble.

        ``connect_neurons`` connects ensembles in a way that bypasses
        many of the facilities that Nengo provides. However,
        even in Nengo models, it is often useful to work directly
        with neural activities rather than with the vectors that those
        activities represent. This is especially useful for
        gating and learning.

        Parameters
        ----------
        pre, post : str or Ensemble
            The items to connect.
            ``pre`` and ``post`` can be strings that identify an Ensemble
            (see `string reference <string_reference.html>`_), or they
            can be the Ensembles themselves.

        weights : float matrix (``pre`` neurons by ``post`` neurons)
            A matrix representing the connection weights
            between each ``pre`` neuron to each ``post`` neuron.
            The dimensionality of this matrix is ``pre`` neurons by
            ``post`` neurons.

            In the following simple example, an ensemble with a single neuron
            has strong negative connections to an ensemble with ten neurons::

              pre = model.make_ensemble("pre", neurons=1, dimensions=1)
              post = model.make_ensemble("post", neurons=10, dimensions=1)
              model.connect_neurons(pre, post, [[-1] * 10])

            The weight matrix is cumbersome to make manually;
            a helper function for making transforms is provided
            (see :func:`nengo.gen_weights()`).

        filter : dict, optional
            ``filter`` contains information about the type of filter
            to use across this connection.

            **Default**: specifies an exponentially decaying filter
            with ``tau=0.01``.

        learning_rule : dict, optional
            ``learning_rule`` contains information about the type of
            learning rule that modifies this connection.

            **Default**: None

        Returns
        -------
        connection : Connection
            The Connection object created.

        See Also
        --------
        Network.connect_neurons : The same function for Networks

        """
        return self.network.connect_neurons(
            pre, post, weights, filter, learning_rule)


class Network(object):
    """A Network is a collection of ensembles, other nodes, and connections.

    Networks help segregate a model or large network
    into more manageable pieces,
    and make it easier to share pieces of a model with other models.

    A Network can contain sub-networks (self.networks).
    In addition to properties that list all of the objects
    items in this network, analogous `all_*` properties
    recursively extract elements from this network's networks,
    to provide a flattened view of the network.

    Parameters
    ----------
    name : str
        Name of the network.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the network
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire network changes.

    Attributes
    ----------
    name : str
        A unique name that identifies the network.
    metadata : dict
        An editable dictionary that modelers can use to store
        extra information about a network.
    properties : readonly dict?
        A collection of basic information about
        a network (e.g., number of neurons, number of synapses, etc.)
    ensembles : list
        The ensembles contained in this network.
    nodes : list
        The nodes contained in this network.
    networks : list
        The networks contained in this network.
    connections : list
        The connections contained in this network.
    members : list
        The ensembles, nodes, networks, and connections
        contained in this network.
    all_ensembles : list
        The ensembles contained in this network,
        including those in sub-networks.
    all_nodes : list
        The nodes contained in this network.
    all_networks : list
        The networks contained in this network.
    all_connections : list
        The connections contained in this network,
        including those in sub-networks.
    all_members : list
        The ensembles, nodes, networks, and connections
        contained in this network, including those in sub-networks.
    probed : list
        A list of Vars that are being probed in this network.
    all_probed : list
        A list of Vars that are being probed in this network,
        including probed items in sub-networks.

    """
    def __init__(self, name, seed=None):
        self.connections = []
        self.ensembles = []
        self.nodes = []
        self.networks = []
        self.probed = []
        # self.filters = []

    @property
    def all_connections(self):
        rval = list(self.connections)
        for network in self.networks:
            rval.extend(network.all_connections)
        return rval

    @property
    def all_ensembles(self):
        rval = list(self.ensembles)
        for network in self.networks:
            rval.extend(network.all_ensembles)
        return rval

    @property
    def all_nodes(self):
        rval = list(self.nodes)
        for network in self.networks:
            rval.extend(network.all_nodes)
        return rval

    @property
    def all_networks(self):
        rval = list(self.networks)
        for network in self.networks:
            rval.extend(network.all_networks)
        return rval

    @property
    def all_probed(self):
        rval = list(self.probed)
        for network in self.networks:
            rval.extend(network.all_probed)
        return rval
    
    @property
    def members(self):
        rval = []
        rval.extend(self.ensembles)
        rval.extend(self.nodes)
        rval.extend(self.networks)
        rval.extend(self.connections)
        # rval.extend(self.probes)
        # rval.extend(self.filters)
        return rval

    @property
    def all_members(self):
        rval = []
        rval.extend(self.all_ensembles)
        rval.extend(self.all_nodes)
        rval.extend(self.all_networks)
        rval.extend(self.all_connections)
        return rval

    def add(self, obj):
        """Adds a Nengo object to this network.

        This is generally only used for manually created nodes, not ones
        created by calling :func:`nef.Network.make_ensemble()` or
        :func:`nef.Network.make_node()`, as these are automatically added.
        A common usage is with user created subclasses, as in the following::

          node = net.add(MyNode('name'))

        Parameters
        ----------
        nengo_object : Nengo object
            The Nengo object to add.

        Returns
        -------
        nengo_object : Nengo object
            The Nengo object that was added.

        See Also
        --------
        Model.add : The same function for Models

        """
        if isinstance(obj, Ensemble):
            self.ensembles.append(obj)
        elif isinstance(obj, Node):
            self.nodes.append(obj)
        elif isinstance(obj, Network):
            self.networks.append(obj)
        elif isinstance(obj, Connection):
            self.connections.append(obj)
        else:
            raise TypeError("Cannot add object of type "
                            + obj.__class__.__name__ + " to a network.")
        return obj

    def connect(self, pre, post, function=None, transform=None,
                filter=psc(), learning_rule=None):
        """Connect ``pre`` to ``post``.

        Parameters
        ----------
        pre, post : str or Nengo object
            The items to connect.
            ``pre`` and ``post`` can be strings that identify a Nengo object
            (see `string reference <string_reference.html>`_), or they
            can be the Nengo objects themselves.

        function : Python function, optional
            The function that this connection will compute.
            This function takes as input the vector being represented by
            ``pre``, and returns another vector which will be
            projected to ``post``.
            If ``function`` is not specified, by default the
            identity function will be used (i.e., the function returns
            the same vector that it takes as input;
            :math:`f(\mathbf{x}) = \mathbf{x}`).
            The function takes a single parameter ``x``,
            which is the current value of the ``pre`` ensemble,
            and must return a float (for one-dimensional functions) or
            a list of floats.

            The following simple example connects two ensembles together
            such that the second ensemble represents
            the square of the first ensemble::

              def square(x):
                  return x * x
              pre = net.make_ensemble('pre', neurons=30, dimensions=1)
              post = net.make_ensemble('post', neurons=30, dimensions=1)
              net.connect(pre, post, function=square)

            or, slightly more succinctly::

              net.connect(pre, post, function=lambda x: x * x)

            **Default**: the ``identity`` function
            (:math:`f(x) = x`).
              
        transform : float matrix (``function`` dims by ``post`` dims), optional
            A matrix that maps the computed function onto ``post``.
            Its dimensionality is ``function`` output dimensions
            by ``post`` dimensions. If ``transform`` is not specified,
            the identity matrix will be used. This mainly makes sense
            when the dimensionality of the ``function`` output
            is exactly the dimensionality of ``post``; if this isn't true,
            then you should probably explicitly define ``transform``.

            The following simple example passes through the values
            represented by a 2-dimensional ensemble to
            the first and third dimension of a 4-dimensional ensemble::

              pre = net.make_ensemble('pre', neurons=40, dimensions=2)
              post = net.make_ensemble('post', neurons=80, dimensions=4)
              net.connect(pre, post, transform=[[1, 0], [0, 0], [0, 1], [0, 0]])

            The transform matrix is quite confusing to make manually;
            a helper function for making transforms is provided
            (see :func:`nengo.gen_transform()`).

            The following complex example computes the product of
            a 2-dimensional vector, and projects that to the second dimension
            of a 2-dimensional ensemble::

              def product(x):
                  return x[0] * x[1]
              pre = net.make_ensemble('pre', neurons=40, dimensions=2)
              post = net.make_ensemble('post', neurons=40, dimensions=2)
              net.connect(pre, post, func=product, transform=[[0], [1]])

            or, slightly more succinctly::

              net.connect(pre, post, func=lambda x: x[0] * x[1],
                            transform=[[0], [1]])

            **Default**: an identity matrix.

        filter : dict, optional
            ``filter`` contains information about the type of filter
            to use across this connection.

            **Default**: specifies an exponentially decaying filter
            with ``tau=0.01``.

        learning_rule : dict, optional
            ``learning_rule`` contains information about the type of
            learning rule that modifies this connection.

            **Default**: None

        Returns
        -------
        connection : Connection
            The Connection object created.

        See Also
        --------
        Connection : The Connection object
        Model.connect : The same function for Models
        
        """
        connection = Connection(
            pre, post, function, transform, filter, learning_rule)
        self.connections.append(connection)
        return connection

    def connect_neurons(self, pre, post, weights,
                        filter=psc(), learning_rule=None):
        """Directly connect the neurons in the ``pre`` ensemble
        to the neurons in the ``post`` ensemble.

        ``connect_neurons`` connects ensembles in a way that bypasses
        many of the facilities that Nengo provides. However,
        even in Nengo models, it is often useful to work directly
        with neural activities rather than with the vectors that those
        activities represent. This is especially useful for
        gating and learning.

        Parameters
        ----------
        pre, post : str or Ensemble
            The items to connect.
            ``pre`` and ``post`` can be strings that identify an Ensemble
            (see `string reference <string_reference.html>`_), or they
            can be the Ensembles themselves.

        weights : float matrix (``pre`` neurons by ``post`` neurons)
            A matrix representing the connection weights
            between each ``pre`` neuron to each ``post`` neuron.
            The dimensionality of this matrix is ``pre`` neurons by
            ``post`` neurons.

            In the following simple example, an ensemble with a single neuron
            has strong negative connections to an ensemble with ten neurons::

              pre = net.make_ensemble("pre", neurons=1, dimensions=1)
              post = net.make_ensemble("post", neurons=10, dimensions=1)
              net.connect_neurons(pre, post, [[-1] * 10])

            The weight matrix is cumbersome to make manually;
            a helper function for making transforms is provided
            (see :func:`nengo.gen_weights()`).

        filter : dict, optional
            ``filter`` contains information about the type of filter
            to use across this connection.

            **Default**: specifies an exponentially decaying filter
            with ``tau=0.01``.

        learning_rule : dict, optional
            ``learning_rule`` contains information about the type of
            learning rule that modifies this connection.

            **Default**: None

        Returns
        -------
        connection : Connection
            The Connection object created.

        See Also
        --------
        Model.connect_neurons : The same function for Models

        """
        connection = Connection(pre, post, weights, filter, learning_rule)
        self.connections.append(connection)
        return connection

    def get(self, target, default=None):
        """Return the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The ``target`` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a Nengo object is passed, ``get`` just confirms
            that ``target`` is a part of the network.

        default : optional
            If ``target`` is not in the network, then ``get`` will
            return ``default``.
            If ``default`` is not specified and ``target`` does not
            exist, a ``ValueError`` is raised.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        Raises
        ------
        ValueError
            If the ``target`` does not exist and no ``default`` is specified.

        See Also
        --------
        Model.get : The same function for Models

        """
        pass

    def make_alias(self, alias, target):
        """Adds a named shortcut to an existing Nengo object
        within this network.

        This is designed to simplify :func:`nengo.Network.connect()`,
        :func:`nengo.Network.get()`, and :func:`nengo.Network.remove()` calls.
        For example, you can do::

            net.make_alias('vision', 'A.B.C.D.E')
            net.make_alias('motor', 'W.X.Y.Z')
            net.connect('vision', 'motor')

        Parameters
        ----------
        alias : str
            The alias to assign to ``target``.
        target : str or Nengo object
            Identifies the Nengo object to be aliased.

        Raises
        ------
        ValueError
            If ``target`` can't be found in the network.
        
        See Also
        --------
        Model.make_alias : The same function for Models

        """
        pass

    def make_ensemble(self, name, neurons, dimensions, max_rate=(50, 100),
                      intercept=(-1, 1), radius=1.0, encoders=None,
                      neuron_model=lif(), mode='spiking'):
        """Create and return an ensemble of neurons.

        The ensemble created by this function is automatically added to
        the network.

        Parameters
        ----------
        name : str
            Name of the ensemble. Must be unique within the network.
        neurons : int
            Number of neurons in the ensemble.
        dimensions : int
            Number of dimensions that this ensemble will represent.
        max_rate : iterable, optional
            A 2-element iterable containing the minimum and maximum
            values of a uniform distribution from which the maximum
            firing rates of neurons in the ensemble will be selected
            (in Hz).

            **Default**: (50, 100)
        intercept : iterable, optional
            A 2-element iterable containing the minimum and maximum
            values of a uniform distribution from which the x-intercepts
            of neuron tuning curves will be selected.

            **Default**: (-1, 1)
        radius : float, optional
            The representational range of the ensemble.
            I.e., the maximum value that can be represented
            in each dimension.

            **Default**: 1.0
        encoders : 2-D matrix of floats, optional
            A matrix containing encoding vectors for each neuron.

            **Default**: randomly generated vectors on the unit sphere.
        neuron_model : dict, optional
            Specifies the neuron model that this ensemble will
            be made up of.

            **Default**: A leaky integrate-and-fire (LIF) neuron
            with ``tau_rc=0.02``, ``tau_ref=0.002``.
        mode : {'spiking', 'direct', 'rate'}, optional
            Simulation mode.

            **Default**: 'spiking'

        See Also
        --------
        Model.make_ensemble : The same function for Models

        """
        pass

    def make_network(self, name, seed=None):
        """Create and return a network.

        Networks can contain other networks, and are useful
        for organizing ensembles and connections.
        This function creates a new network, which can then be used to
        create ensembles and other Nengo objects that are within that
        network.

        Parameters
        ----------
        name : str
            Name of the network. This must be unique within the network.

        Returns
        -------
        network : Network
            The created network.
        seed : int, optional
            Random number seed that will be fed to the
            random number generator. Setting this seed makes
            the creation of the network a deterministic process;
            however, each new ensemble in the network advances
            the random number generator, so if the network creation code
            changes, the entire network changes.

        See Also
        --------
        Model.make_network : The same function for Models

        """
        pass
    
    def make_node(self, name, output):
        """Create and return a node of dimensionality ``len(output)``,
        which produces the defined output.

        Parameters
        ----------
        name : str
            Name of this node. Must be unique in the network.
        output : function, list of floats, dict
            The output that should be generated by this node.

            If ``output`` is a function, it will be called on each timestep;
            if it accepts a single parameter, it will be given
            the current time of the simulation.
            
            If ``output`` is a list of floats, that list will be
            used as constant output.

            If ``output`` is a dict, the output defines a piece-wise constant
            function in which the keys define when the value changes,
            and the values define what the value changes to.
        
        Returns
        -------
        node : Node
            The created Node

        See Also
        --------
        Network.make_node : The same function for Networks

        """
        pass

    def remove(self, target):
        """Removes a Nengo object from the network.

        Parameters
        ----------
        target : str, Nengo object
            A string referencing the Nengo object to be removed
            (see `string reference <string_reference.html>`)
            or node or name of the node to be removed.

        Returns
        -------
        target : Nengo object
            The Nengo object removed.

        See Also
        --------
        Model.remove : The same function for Models

        """
        pass

    def probe(self, target, sample_every=None, static=False):
        """Probe a piece of data contained in the network.

        When a piece of data is probed, it will be recorded through
        the course of the simulation.

        Parameters
        ----------
        target : str, Nengo object
            The piece of data being probed.
            This can specified as a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object. Each Nengo object will emit
            what it considers to be the most useful piece of data
            by default; if that's not what you want,
            then specify the correct data using the string format.
        sample_every : float, optional
            How often to sample the target data, in seconds.

            Some types of data (e.g. connection weight matrices)
            are very large, and change relatively slowly.
            Use ``sample_every`` to limit the amount of data
            being recorded. For example::

              net.probe('A>B.weights', sample_every=0.5)

            records the value of the weight matrix between
            the ``A`` and ``B`` ensembles every 0.5 simulated seconds.

            **Default**: Every timestep (i.e., ``dt``).
        static : bool, optional
            Denotes if a piece of data does not change.

            Some data that you would want to know about the network
            does not change over the course of the simulation;
            this includes things like the properties of a network
            (e.g., number of neurons or connections) or the random seed
            associated with a network. In these cases, to record that data
            only once (for later being written to a file),
            set ``static`` to True.

            **Default**: False

        See Also
        --------
        Model.probe : The same function for Models

        """
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
    neurons : int
        The number of neurons in this ensemble.
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
    pass


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

    def __init__(self, name, output=None):
        pass

    def step(self):
        """A function that can be overwritten by a modeler
        to execute arbitrary code on each timestep.

        """
        pass

    def reset(self):
        """A function that will be called on network reset
        that allows a modeler to re-initialize the node's state.

        """
        pass


import inspect


def gen_transform(pre_dims, post_dims,
                  weight=1, index_pre=None, index_post=None):
    """Helper function used to create a ``pre_dims`` by ``post_dims``
    linear transformation matrix.

    Parameters
    ----------
    pre_dims, post_dims : int
        The numbers of presynaptic and postsynaptic dimensions.
    weight : float, optional
        The weight value to use in the transform.

        All values in the transform are either 0 or ``weight``.

        **Default**: 1.0
    index_pre, index_post : iterable of int
        Determines which values are non-zero, and indicates which
        dimensions of the pre-synaptic ensemble should be routed to which
        dimensions of the post-synaptic ensemble.

    Returns
    -------
    transform : 2D matrix of floats
        A two-dimensional transform matrix performing the requested routing.

    Examples
    --------

      # Sends the first two dims of pre to the first two dims of post
      >>> gen_transform(pre_dims=2, post_dims=3,
                        index_pre=[0, 1], index_post=[0, 1])
      [[1, 0], [0, 1], [0, 0]]

    """
    t = [[0] * pre_dims for i in range(post_dims)]
    if index_pre is None:
        index_pre = range(dim_pre)
    elif isinstance(index_pre, int):
        index_pre = [index_pre]
    if index_post is None:
        index_post = range(dim_post)
    elif isinstance(index_post, int):
        index_post = [index_post]

    for i in range(max(len(index_pre), len(index_post))):
        pre = index_pre[i % len(index_pre)]
        post = index_post[i % len(index_post)]
        t[post][pre] = weight
    return t

    
def gen_weights(pre_neurons, post_neurons, function):
    """Helper function used to create a ``pre_neurons`` by ``post_neurons``
    connection weight matrix.

    Parameters
    ----------
    pre_neurons, post_neurons : int
        The numbers of presynaptic and postsynaptic neurons.
    function : function
        A function that generates weights.

        If it accepts no arguments, it will be called to
        generate each individual weight (useful
        to great random weights, for example).
        If it accepts two arguments, it will be given the
        ``pre`` and ``post`` index in the weight matrix.

    Returns
    -------
    weights : 2D matrix of floats
        A two-dimensional connection weight matrix.

    Examples
    --------

      >>> gen_weights(2, 2, random.random)
      [[0.6281625119511959, 0.48560016153108376], [0.9639779858394248, 0.4768136917985597]]

      >>> def product(pre, post):
      ...     return pre * post
      >>> gen_weights(3, 3, product)
      [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

    """
    argspec = inspect.getargspec(func)
    if len(argspec[0]) == 0:
        return [[func() for _ in xrange(pre_neurons)
                 for _ in xrange(post_neurons)]]
    elif len(argspec[0]) == 2:
        return [[func(pre, post) for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]


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
    pass
    
