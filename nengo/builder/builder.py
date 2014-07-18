import collections
import itertools
import warnings

import numpy as np

from nengo.builder.signal import SignalDict


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt=0.001, label=None):
        self.dt = dt
        self.label = label

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

    def __str__(self):
        return "Model: %s" % self.label

    def add_op(self, op):
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        op.init_signals(signals, self.dt)
        op.make_step(signals, self.dt)

    def build(self, *objs):
        return Builder.build(self, *objs)

    def has_built(self, *objs):
        """Returns true iff obj has been processed by build."""
        if len(objs) == 1:
            objs = objs[0]
        return objs in self.params


class Builder(object):
    builders = {}

    @classmethod
    def register(cls, *nengo_classes):
        def register_builder(build_fn):
            if nengo_classes in cls.builders:
                warnings.warn("Types '%s' already have a builder. Overwriting."
                              % str(nengo_classes))
            cls.builders[nengo_classes] = build_fn
            return build_fn
        return register_builder

    @classmethod
    def find_builder(cls, *objs):
        mros = tuple(obj.__class__.__mro__ for obj in objs)
        # itertools.product iterates in the order that we want,
        # with the most specific (and most likely) first
        for classes in itertools.product(*mros):
            if classes in cls.builders:
                return cls.builders[classes]
        else:
            raise TypeError("Cannot build types '%s'." %
                            str(tuple(obj.__class__.__name__ for obj in objs)))

    @classmethod
    def build(cls, model, *objs):
        if model.has_built(*objs):
            # TODO: Prevent this at pre-build validation time.
            warnings.warn("Object '%s' has already been built." % str(objs))
            return
        builder = cls.find_builder(*objs)
        builder(model, *objs)
        if not model.has_built(*objs):
            raise RuntimeError(
                "Build function '%s' did not add '%s' to model.params"
                % (builder.__name__, str(objs)))
