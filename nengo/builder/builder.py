import collections
import warnings

import numpy as np

from nengo.builder.signal import Signal, SignalDict
from nengo.builder.operator import TimeUpdate
from nengo.cache import NoDecoderCache
from nengo.exceptions import BuildError


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt=0.001, label=None, decoder_cache=NoDecoderCache()):
        self.dt = dt
        self.label = label
        self.decoder_cache = decoder_cache

        # We want to keep track of the toplevel network
        self.toplevel = None
        # Builders can set a config object to affect sub-builders
        self.config = None

        # Resources used by the build process.
        self.operators = []
        self.params = {}
        self.seeds = {}
        self.probes = []

        self.sig = collections.defaultdict(dict)
        self.sig['common'][0] = Signal(0., readonly=True, name='ZERO')
        self.sig['common'][1] = Signal(1., readonly=True, name='ONE')

        self.step = Signal(np.array(0, dtype=np.int64), name='step')
        self.time = Signal(np.array(0, dtype=np.float64), name='time')
        self.add_op(TimeUpdate(self.step, self.time))

    def __str__(self):
        return "Model: %s" % self.label

    def build(self, obj, *args, **kwargs):
        return Builder.build(self, obj, *args, **kwargs)

    def add_op(self, op):
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict()
        op.init_signals(signals)
        op.make_step(signals, self.dt, np.random)

    def has_built(self, obj):
        """Returns true if obj has built parameters.

        Some objects (e.g. synapses) do not add params and can be built more
        than once (i.e. this function returns false), allowing re-use.
        """
        return obj in self.params


class Builder(object):
    builders = {}

    @classmethod
    def register(cls, nengo_class):
        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder

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
