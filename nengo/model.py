import codecs
import inspect
import json
import logging
import math
import pickle
import os.path
import numpy as np

from . import core
from . import connections
from . import objects
from . import simulator


logger = logging.getLogger(__name__)

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


    Attributes
    ----------
    name : str
        Name of the model
    seed : int
        Random seed used by the model.
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
                 simulator=simulator.Simulator, dt=0.001):
        self.dt = dt

        self.signals = set()
        self.nonlinearities = set()
        self.encoders = set()
        self.decoders = set()
        self.transforms = set()
        self.filters = set()
        self.probes = set()

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.data = {}

        self.name = name
        self.simulator = simulator

        self.seed = np.random.randint(2**31-1) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)
        self.fixed_seed = fixed_seed

        self.simtime = self.add(core.Signal(name='simtime'))
        self.steps = self.add(core.Signal(name='steps'))
        self.one = self.add(core.Constant(1, value=[1.0], name='one'))

        # Automatically probe these
        self.probe(self.simtime)
        self.probe(self.steps)

        # -- steps counts by 1.0
        self.add(core.Filter(1.0, self.one, self.steps))
        self.add(core.Filter(1.0, self.steps, self.steps))

        # simtime <- dt * steps
        self.add(core.Filter(dt, self.one, self.simtime))
        self.add(core.Filter(dt, self.steps, self.simtime))

    def _get_new_seed(self):
        return self.rng.randint(2**31-1) if self.fixed_seed is None \
            else self.fixed_seed

    def __str__(self):
        return "Model: " + self.name

    @property
    def connections(self):
        return [o for o in self.objs.values() if isinstance(o, Connection)]

    @property
    def ensembles(self):
        return [o for o in self.objs.values() if isinstance(o, Ensemble)]

    @property
    def nodes(self):
        return [o for o in self.objs.values() if isinstance(o, Node)]

    @property
    def networks(self):
        return [o for o in self.objs.values() if isinstance(o, Network)]

    ### I/O

    def save(self, fname, format=None):
        """Save this model to a file.

        So far, JSON and Pickle are the possible formats.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        if format in ('json', '.json'):
            with codecs.open(fname, 'w', encoding='utf-8') as f:
                json.dump(self.to_json(), f, sort_keys=True, indent=2)
                logger.info("Saved %s successfully.", fname)
        else:
            # Default to pickle
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
                logger.info("Saved %s successfully.", fname)

    def to_json(self):
        d = {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'dt': self.dt,
            # 'simulator': ?? We probably don't want to serialize this
        }

        d['signals'] = [sig.to_json() for sig in self.signals]
        d['nonlinearities'] = [nl.to_json() for nl in self.nonlinearities]
        d['encoders'] = [enc.to_json() for enc in self.encoders]
        d['decoders'] = [dec.to_json() for dec in self.decoders]
        d['transforms'] = [trans.to_json() for trans in self.transforms]
        d['filters'] = [filt.to_json() for filt in self.filters]
        d['probes'] = [pr.to_json() for pr in self.probes]

        # d['aliases'] = self.aliases
        # d['objs'] = {k: v.to_json() for k, v in self.objs.items()}
        # d['probed'] = ?? Deal with later!
        # d['data'] = ?? Do we want to serialize this?
        return d

    @staticmethod
    def load(self, fname, format=None):
        """Load this model from a file.

        So far, JSON and Pickle are the possible formats.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        if format == 'json':
            with codecs.open(fname, 'r', encoding='utf-8') as f:
                return Model.from_json(json.load(f))
        else:
            # Default to pickle
            with open(fname, 'rb') as f:
                return pickle.load(f)

        raise IOError("Could not load {}".format(fname))

    ### Simulation methods

    def reset(self):
        """Reset the state of the simulation.

        Runs through all nodes, then ensembles, then connections and then
        probes in the network and calls thier reset functions.

        """
        logger.debug("Resetting simulator for %s", self.name)
        self.sim_obj.reset()

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
        if getattr(self, 'sim_obj', None) is None:
            logger.debug("Creating simulator for %s", self.name)
            self.sim_obj = self.simulator(self)

        steps = int(time // self.dt)
        logger.debug("Running for %f seconds; %d steps", time, steps)
        self.sim_obj.run_steps(steps)

        for k in self.probed:
            self.data[k] = self.sim_obj.probe_data(self.probed[k])

        return self.data

    ### Model manipulation

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
        if hasattr(obj, 'name') and self.objs.has_key(obj.name):
            raise ValueError("Something called " + obj.name + " already exists."
                             " Please choose a different name.")
        obj.add_to_model(self)
        if hasattr(obj, 'name') and not obj.__module__ == 'core':
            self.objs[obj.name] = obj
        return obj

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

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            logger.error("Cannot find %s in this model.", target)
            return default

        if not target in self.objs.values():
            logger.error("Cannot find %s in this model.", str(target))
            return default

        return target

    def get_string(self, target, default=None):
        """Return the canonical string of the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The ``target`` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a string is passed, ``get_string`` returns
            the canonical version of it; i.e., if it is
            an alias, the non-aliased version is returned.

        default : optional
            If ``target`` is not in the model, then ``get`` will
            return ``default``.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        Raises
        ------
        ValueError
            If the ``target`` does not exist and no ``default`` is specified.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                obj = self.aliases[target]
            elif self.objs.has_key(target):
                return target

        for k, v in self.objs.iteritems():
            if v == target:
                return k

        logger.warning("Cannot find %s in this model.", str(target))
        return default

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

        """
        obj = self.get(target)
        if obj is None:
            logger.warning("%s is not in this model.", str(target))
            return

        obj.remove_from_model(self)

        for k, v in self.objs.iteritems():
            if v == obj:
                del self.objs[k]
                logger.info("%s removed.", k)
        for k, v in self.aliases.iteritem():
            if v == obj:
                del self.aliases[k]
                logger.info("Alias '%s' removed.", k)

        return obj

    def alias(self, alias, target):
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

        """
        obj_s = self.get_string(target)
        if obj_s is None:
            raise ValueError(target + " cannot be found.")
        self.aliases[alias] = obj_s
        logger.info("%s aliased to %s", obj_s, alias)
        return self.get(obj_s)


    # Model creation methods

    def make_ensemble(self, name, neurons, dimensions,
                      max_rates=objects.Uniform(200, 300),
                      intercepts=objects.Uniform(-1, 1),
                      radius=1.0, encoders=None):
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
        max_rates : iterable, optional
            A 2-element iterable containing the minimum and maximum
            values of a uniform distribution from which the maximum
            firing rates of neurons in the ensemble will be selected
            (in Hz).

            **Default**: (50, 100)
        intercepts : iterable, optional
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
        Ensemble : The Ensemble object

        """
        ens = objects.Ensemble(
            name, neurons, dimensions,
            max_rates=max_rates, intercepts=intercepts, radius=radius,
            encoders=encoders, seed=self._get_new_seed())
        return self.add(ens)

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

        """
        net = objects.Network(name, seed, model=self)
        return self.add(net)

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
        Node : The Node object

        """
        node = objects.Node(name, output, input=self.simtime)
        return self.add(node)

    def probe(self, target, sample_every=None, filter=None):
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

        """
        if sample_every is None:
            sample_every = self.dt

        probe_type = ''
        key = target
        if isinstance(target, str):
            s = target.split('.')
            if len(s) > 1:
                target, probe_type = s[0], s[1]
        obj = self.get(target)

        if type(obj) == objects.Ensemble:
            obj_s = self.get_string(target)
            p = obj.probe(probe_type, sample_every, filter, self)
            self.probed[key] = p.probe
            return p

        if type(obj) != core.Signal:
            obj = obj.signal

        if filter is not None and filter > self.dt:
            fcoef, tcoef = _filter_coefs(pstc=filter, dt=self.dt)
            probe_sig = self.add(core.Signal(obj.n))
            self.add(core.Filter(fcoef, probe_sig, probe_sig))
            self.add(core.Transform(tcoef, obj, probe_sig))
            p = self.add(core.Probe(probe_sig, sample_every))
        else:
            p = self.add(core.Probe(obj, sample_every))

        self.probed[key] = p
        return p

    def connect(self, pre, post, **kwargs):
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

        """
        pre = self.get(pre)
        post = self.get(post)

        if type(pre) == objects.Ensemble:
            logger.info("Creating DecodedConnection")
            return self.add(connections.DecodedConnection(pre, post, **kwargs))
        else:
            logger.info("Creating SimpleConnection")
            return self.add(connections.SimpleConnection(pre, post, **kwargs))


def gen_transform(pre_dims, post_dims,
                  weight=1.0, index_pre=None, index_post=None):
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
    t = [[0 for pre in xrange(pre_dims)] for post in xrange(post_dims)]
    if index_pre is None:
        index_pre = range(dim_pre)
    elif isinstance(index_pre, int):
        index_pre = [index_pre]

    if index_post is None:
        index_post = range(dim_post)
    elif isinstance(index_post, int):
        index_post = [index_post]

    for i in xrange(min(len(index_pre), len(index_post))):  # was max
        pre = index_pre[i]  # [i % len(index_pre)]
        post = index_post[i]  # [i % len(index_post)]
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
        return [[func() for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]
    elif len(argspec[0]) == 2:
        return [[func(pre, post) for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]
