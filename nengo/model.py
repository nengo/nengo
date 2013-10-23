import codecs
from collections import OrderedDict
import copy
import json
import logging
import pickle
import os.path
import numpy as np

from . import connections
from . import core
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

    def __init__(self, name, seed=None):
        self.signals = []
        self.probes = []

        #
        # -- Build stuff --
        #
        self._operators = []

        self.objs = {}
        self.aliases = {}
        self.probed = OrderedDict()
        self.connections = []
        self.signal_probes = []

        self.name = name + ''  # -- make self.name a string, raise error otw
        self.seed = seed

        self.t = self.add(core.Signal(name='t'))
        self.steps = self.add(core.Signal(name='steps'))
        self.one = self.add(core.Constant([1.0], name='one'))

        # Automatically probe these
        self.probe(self.t)
        self.probe(self.steps)

        # -- steps counts by 1.0
        self._operators += [simulator.ProdUpdate(
                core.Constant(1), self.one, core.Constant(1), self.steps)]

        self._rng = None

    def __str__(self):
        return "Model: " + self.name

    def _get_new_seed(self):
        if self._rng is None:
            # never create rng without knowing the seed
            assert self.seed is not None
            self._rng = np.random.RandomState(self.seed)
        return self._rng.randint(2**32)

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

    @staticmethod
    def prep_for_simulation(model, dt):
        model.name = model.name + ", dt=%f" % dt
        model.dt = dt
        model._operators += [simulator.ProdUpdate(core.Constant(dt), model.one,
                                                  core.Constant(1), model.t)]

        # Sort all objects by name
        all_objs = sorted(model.objs.values(), key=lambda o: o.name)

        # 1. Build objects first
        logger.info("Building objects")
        for o in all_objs:
            o.build(model=model, dt=dt)

        # 2. Then probes
        logger.info("Building probes")
        for target in model.probed:
            if not isinstance(model.probed[target], core.Probe):
                model.probed[target].build(model=model, dt=dt)
                model.probed[target] = model.probed[target].probe

        # 3. Then connections
        logger.info("Building connections")
        for o in all_objs:
            for c in o.connections_out:
                c.build(model=model, dt=dt)
        for c in model.connections:
            c.build(model=model, dt=dt)

    def simulator(self, dt=0.001, sim_class=simulator.Simulator,
                  seed=None, **sim_args):
        """Get a new simulator object for the model.

        Parameters
        ----------
        dt : float, optional
            Fundamental unit of time for the simulator, in seconds.
        sim_class : child class of `Simulator`, optional
            The class of simulator to be used.
        seed : int, optional
            Random number seed for the simulator's random number generator.
            This random number generator is responsible for creating any random
            numbers used during simulation, such as random noise added to
            neurons' membrane voltages.
        **sim_args : optional
            Arguments to pass to the simulator constructor.

        Returns
        -------
        simulator : `sim_class`
            A new simulator object, containing a copy of the model in its
            current state.
        """
        logger.info("Copying model")
        memo = {}
        modelcopy = copy.deepcopy(self, memo)
        modelcopy.memo = memo

        if modelcopy.seed is None:
            modelcopy.seed = np.random.randint(2**32) # generate model seed

        if seed is None:
            seed = modelcopy._get_new_seed() # generate simulator seed

        self.prep_for_simulation(modelcopy, dt)
        return sim_class(model=modelcopy, **sim_args) # TODO: pass in seed

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
        try:
            obj.add_to_model(self)
            return obj
        except AttributeError:
            raise TypeError("Error in %s.add_to_model."%obj)

    def get(self, target, default=None):
        """Return the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The `target` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a Nengo object is passed, `get` just confirms
            that `target` is a part of the model.

        default : Nengo object, optional
            If `target` is not in the model, then `get` will
            return `default`.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by `target`.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            if default is None:
                logger.error("Cannot find %s in model %s.", target, self.name)
            return default

        return target

    def get_string(self, target, default=None):
        """Return the canonical string of the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The `target` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a string is passed, `get_string` returns
            the canonical version of it; i.e., if it is
            an alias, the non-aliased version is returned.

        default : Nengo object, optional
            If `target` is not in the model, then `get` will
            return `default`.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by `target`.

        Raises
        ------
        ValueError
            If the `target` does not exist and no `default` is specified.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                obj = self.aliases[target]
            elif self.objs.has_key(target):
                return target

        for k, v in self.objs.iteritems():
            if v == target:
                return k

        logger.warning("Cannot find %s in model %s.", str(target), self.name)
        return default

    # def data(self, target):
    #     target = self.get_string(target, target)
    #     if not isinstance(target, str):
    #         target = target.name
    #     return self._data[target]

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
            logger.warning("%s is not in model %s.", str(target), self.name)
            return

        if 'core' in obj.__module__:
            obj.remove_from_model(self)
        else:
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
            The alias to assign to `target`.
        target : str or Nengo object
            Identifies the Nengo object to be aliased.

        Raises
        ------
        ValueError
            If `target` can't be found in the model.

        """
        obj_s = self.get_string(target)
        if obj_s is None:
            raise ValueError(target + " cannot be found.")
        self.aliases[alias] = obj_s
        logger.info("%s aliased to %s", obj_s, alias)
        return self.get(obj_s)

    # Model creation methods

    def make_ensemble(self, name, neurons, dimensions, **ensemble_kwargs):
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
        **ensemble_kwargs : optional
            Additional arguments to pass to the `Ensemble` constructor

        See Also
        --------
        Ensemble : The Ensemble object
        """
        ens = objects.Ensemble(name, neurons, dimensions, **ensemble_kwargs)
        return self.add(ens)

    def make_node(self, name, output):
        """Create and return a node of dimensionality ``len(output)``,
        which produces the defined output.

        Parameters
        ----------
        name : str
            Name of this node. Must be unique in the model.
        output : function, list of floats, dict
            The output that should be generated by this node.

            If `output` is a function, it will be called on each timestep;
            if it accepts a single parameter, it will be given
            the current time of the simulation.

            If `output` is a list of floats, that list will be
            used as constant output.

            If `output` is a dict, the output defines a piece-wise constant
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
        if callable(output):
            node = objects.Node(name, output)
            self.connect(self.t, node, filter=None)
        else:
            node = objects.ConstantNode(name, output)
        return self.add(node)

    def connect(self, pre, post, **kwargs):
        """Connect `pre` to `post`.

        Parameters
        ----------
        pre, post : str or Nengo object
            The items to connect.
            `pre` and `post` can be strings that identify a Nengo object
            (see `string reference <string_reference.html>`_), or they
            can be the Nengo objects themselves.

        function : Python function, optional
            The function that this connection will compute.
            This function takes as input the vector being represented by
            `pre`, and returns another vector which will be
            projected to `post`.
            If `function` is not specified, by default the
            identity function will be used (i.e., the function returns
            the same vector that it takes as input;
            :math:`f(\mathbf{x}) = \mathbf{x}`).
            The function takes a single parameter `x`,
            which is the current value of the `pre` ensemble,
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

        transform : array_like (`function` dims by `post` dims), optional
            A matrix that maps the computed function onto `post`.
            Its dimensionality is `function` output dimensions
            by `post` dimensions. If `transform` is not specified,
            the identity matrix will be used. This mainly makes sense
            when the dimensionality of the `function` output
            is exactly the dimensionality of `post`; if this isn't true,
            then you should probably explicitly define `transform`.

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
            `filter` contains information about the type of filter
            to use across this connection.

            **Default**: specifies an exponentially decaying filter
            with ``tau=0.01``.

        learning_rule : dict, optional
            `learning_rule` contains information about the type of
            learning rule that modifies this connection.

            **Default**: None

        Returns
        -------
        connection : Connection
            The Connection object created.

        See Also
        --------
        Connection
        """
        pre = self.get(pre)
        post = self.get(post)

        if core.is_signal(pre):
            connection = connections.SignalConnection(pre, post, **kwargs)
            self.connections.append(connection)
            return connection
        else:
            return pre.connect_to(post, **kwargs)

    def probe(self, target, sample_every=0.001, filter=None):
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
            Use `sample_every` to limit the amount of data
            being recorded. For example::

              model.probe('A>B.weights', sample_every=0.5)

            records the value of the weight matrix between
            the `A` and `B` ensembles every 0.5 simulated seconds.

            **Default**: Every timestep (i.e., `dt`).
        static : bool, optional
            Denotes if a piece of data does not change.

            Some data that you would want to know about the model
            does not change over the course of the simulation;
            this includes things like the properties of a model
            (e.g., number of neurons or connections) or the random seed
            associated with a model. In these cases, to record that data
            only once (for later being written to a file),
            set `static` to True.

            **Default**: False

        See Also
        --------
        Probe
        """
        if core.is_signal(target):
            if filter is not None:
                p = objects.Probe(target.name, sample_every, target.n)
                self.signal_probes.append(p)
                self.connect(target, p, filter=filter)
            else:
                p = core.Probe(target, sample_every)
                self.add(p)
        elif isinstance(target, str):
            obj = self.get(target, "NotFound")
            if obj == "NotFound" and '.' in target:
                name, probe_name = target.rsplit('.', 1)
                obj = self.get(name)
                p = obj.probe(probe_name, sample_every, filter)
            elif obj == "NotFound":
                raise ValueError(str(target) + " cannot be found.")
            else:
                p = obj.probe(sample_every=sample_every, filter=filter)
        elif hasattr(target, 'probe'):
            p = target.probe(sample_every=sample_every, filter=filter)
        else:
            raise TypeError("Type " + target.__class__.__name__ + " "
                            "has no probe function.")

        self.probed[target] = p
