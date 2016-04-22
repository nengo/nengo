import collections
import warnings

import numpy as np

from nengo.builder.signal import Signal, SignalDict
from nengo.builder.operator import TimeUpdate
from nengo.cache import NoDecoderCache
from nengo.exceptions import BuildError


class Model(object):
    """Output of the Builder, used by the Simulator.

    Attributes
    ----------
    config : Config instance or None
        Build functions can set a config object here to affect sub-builders.
    decoder_cache : DecoderCache instance
        An object that will be used to cache decoders for faster builds.
    dt : float
        The length of each timestep, in seconds.
    label : str or None
        A name or description to differentiate different models.
    operators : list of Operator instances
        List of all operators created in the build process.
        All operators must be added to this list, as it is used by Simulator.
    params : dict (NengoObject -> namedtuple)
        Mapping from objects to namedtuples containing parameters generated
        in the build process.
    probes : list of Probe
        List of all probes. Probes must be added to this list in the build
        process, as this list is used by Simulator.
    seeded : dict (NengoObject -> bool)
        All objects are assigned a seed, whether the user defined the seed
        or it was automatically generated. 'seeded' keeps track of whether
        the seed is user-defined. We consider the seed to be user-defined
        if it was set directly on the object, or if a seed was set on the
        network in which the object resides, or if a seed was set on any
        ancestor network of the network in which the object resides.
    seeds : dict (NengoObject -> int)
        Mapping from objects to the integer seed assigned to that object.
    sig : dict (str -> dict (object -> Signal))
        A dictionary of dictionaries that organizes all of the signals
        created in the build process, as build functions often need to
        access signals created by other build functions.
    step : Signal(dtype=int64)
        The current step (i.e., how many timesteps have occurred thus far).
    time : Signal(dtype=float64)
        The current point in time.
    toplevel : Network instance
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
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict()
        op.init_signals(signals)
        op.make_step(signals, self.dt, np.random)

    def build(self, obj, *args, **kwargs):
        return Builder.build(self, obj, *args, **kwargs)

    def has_built(self, obj):
        """Returns true if obj has built parameters.

        Some objects (e.g. synapses) do not add params and can be built more
        than once (i.e. this function returns false), allowing re-use.
        """
        return obj in self.params


class Builder(object):
    builders = {}

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
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
        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder
