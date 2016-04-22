import collections
import warnings

import numpy as np

from nengo.builder.signal import Signal, SignalDict
from nengo.builder.operator import TimeUpdate
from nengo.cache import NoDecoderCache
from nengo.exceptions import BuildError


class Model(object):
    """Stores artifacts from the build process, which are used by `.Simulator`.

    Parameters
    ----------
    dt : float, optional (Default: 0.001)
        The length of a simulator timestep, in seconds.
    label : str, optional (Default: None)
        A name or description to differentiate models.
    decoder_cache : DecoderCache, optional (Default: ``NoDecoderCache()``)
        Interface to a cache for expensive parts of the build process.

    Attributes
    ----------
    config : Config or None
        Build functions can set a config object here to affect sub-builders.
    decoder_cache : DecoderCache
        Interface to a cache for expensive parts of the build process.
    dt : float
        The length of each timestep, in seconds.
    label : str or None
        A name or description to differentiate models.
    operators : list
        List of all operators created in the build process.
        All operators must be added to this list, as it is used by Simulator.
    params : dict
        Mapping from objects to namedtuples containing parameters generated
        in the build process.
    probes : list
        List of all probes. Probes must be added to this list in the build
        process, as this list is used by Simulator.
    seeded : dict
        All objects are assigned a seed, whether the user defined the seed
        or it was automatically generated. 'seeded' keeps track of whether
        the seed is user-defined. We consider the seed to be user-defined
        if it was set directly on the object, or if a seed was set on the
        network in which the object resides, or if a seed was set on any
        ancestor network of the network in which the object resides.
    seeds : dict
        Mapping from objects to the integer seed assigned to that object.
    sig : dict
        A dictionary of dictionaries that organizes all of the signals
        created in the build process, as build functions often need to
        access signals created by other build functions.
    step : Signal
        The current step (i.e., how many timesteps have occurred thus far).
    time : Signal
        The current point in time.
    toplevel : Network
        The top-level network being built.
        This is sometimes useful for accessing network elements after build,
        or for the network builder to determine if it is the top-level network.
    """

    def __init__(self, dt=0.001, label=None, decoder_cache=NoDecoderCache()):
        self.dt = dt
        self.label = label
        self.decoder_cache = decoder_cache

        # Will be filled in by the network builder
        self.toplevel = None
        self.config = None

        # Resources used by the build process
        self.operators = []
        self.params = {}
        self.probes = []
        self.seeds = {}
        self.seeded = {}

        self.sig = collections.defaultdict(dict)
        self.sig['common'][0] = Signal(0., readonly=True, name='ZERO')
        self.sig['common'][1] = Signal(1., readonly=True, name='ONE')

        self.step = Signal(np.array(0, dtype=np.int64), name='step')
        self.time = Signal(np.array(0, dtype=np.float64), name='time')
        self.add_op(TimeUpdate(self.step, self.time))

    def __str__(self):
        return "Model: %s" % self.label

    def add_op(self, op):
        """Add an operator to the model.

        In addition to adding the operator, this method performs additional
        error checking by calling the operator's ``make_step`` function.
        Calling ``make_step`` catches errors early, such as when signals are
        not properly initialized, which aids debugging. For that reason,
        we recommend calling this method over directly accessing
        the ``operators`` attribute.
        """
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict()
        op.init_signals(signals)
        op.make_step(signals, self.dt, np.random)

    def build(self, obj, *args, **kwargs):
        """Build an object into this model.

        See `.Builder.build` for more details.

        Parameters
        ----------
        obj : object
            The object to build into this model.
        """
        return Builder.build(self, obj, *args, **kwargs)

    def has_built(self, obj):
        """Returns true if the object has already been built in this model.

        .. note:: Some objects (e.g. synapses) can be built multiple times,
                  and therefore will always result in this method returning
                  ``False`` even though they have been built.

        This check is implemented by checking if the object is in the
        ``params`` dictionary. Build function should therefore add themselves
        to ``model.params`` if they cannot be built multiple times.

        Parameters
        ----------
        obj : object
            The object to query.
        """
        return obj in self.params


class Builder(object):
    """Manages the build functions known to the Nengo build process.

    Consists of two class methods to encapsulate the build function registry.
    All build functions should use the `.Builder.register` method as a
    decorator. For example::

        @nengo.builder.Builder.register(MyRule)
        def build_my_rule(model, my_rule, rule):
            ...

    registers a build function for ``MyRule`` objects.

    Build functions should not be called directly, but instead called through
    the `.Model.build` method. `.Model.build` uses the `.Builder.build` method
    to ensure that the correct build function is called based on the type of
    the object passed to it.
    For example, to build the learning rule type ``my_rule`` from above, do::

        model.build(my_rule, connection.learning_rule)

    This will call the ``build_my_rule`` function from above with the arguments
    ``model, my_rule, connection.learning_rule``.

    Attributes
    ----------
    builders : dict
        Mapping from types to the build function associated with that type.
    """

    builders = {}

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        """Build ``obj`` into ``model``.

        This method looks up the appropriate build function for ``obj`` and
        calls it with the model and other arguments provided.

        Note that if a build function is not specified for a particular type
        (e.g., `.EnsembleArray`), the type's method resolution order will be
        examined to look for superclasses
        with defined build functions (e.g., `.Network` in the case of
        `.EnsembleArray`).

        This indirection (calling `.Builder.build` instead of the build
        function directly) enables users to augment the build process in their
        own models, rather than having to modify Nengo itself.

        In addition to the parameters listed below, further positional and
        keyword arguments will be passed unchanged into the build function.

        Parameters
        ----------
        model : Model
            The `.Model` instance in which to store build artifacts.
        obj : object
            The object to build into the model.
        """
        if model.has_built(obj):
            # TODO: Prevent this at pre-build validation time.
            warnings.warn("Object %s has already been built." % obj)
            return None

        for obj_cls in obj.__class__.__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise BuildError(
                "Cannot build object of type %r" % obj.__class__.__name__)

        return cls.builders[obj_cls](model, obj, *args, **kwargs)

    @classmethod
    def register(cls, nengo_class):
        """A decorator for adding a class to the build function registry.

        Raises a warning if a build function already exists for the class.

        Parameters
        ----------
        nengo_class : Class
            The type associated with the build function being decorated.
        """
        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder
