from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.params import Default, IntParam, NumberParam, StringParam
from nengo.synapses import SynapseParam


class TargetParam(NengoObjectParam):
    def validate(self, probe, target):
        obj = target.obj if isinstance(target, ObjView) else target
        if not hasattr(obj, 'probeable') or len(obj.probeable) == 0:
            raise TypeError(
                "Type '%s' is not probeable" % obj.__class__.__name__)

        # do this after; better to know that type is not Probable first
        super(TargetParam, self).validate(probe, target)


class AttributeParam(StringParam):
    def validate(self, probe, attr):
        super(AttributeParam, self).validate(probe, attr)
        if attr not in probe.obj.probeable:
            raise ValueError("Attribute '%s' is not probeable for '%s'."
                             % (attr, probe.obj))


class Probe(NengoObject):
    """A probe is an object that receives data from the simulation.

    This is to be used in any situation where you wish to gather simulation
    data (spike data, represented values, neuron voltages, etc.) for analysis.

    Probes cannot directly affect the simulation.

    TODO: Example usage for each object.

    Parameters
    ----------
    target : Ensemble, Node, Connection
        The Nengo object to connect to the probe.
    attr : str, optional
        The quantity to probe. Refer to the target's ``probeable`` list for
        details. Defaults to the first element in the list.
    sample_every : float, optional
        Sampling period in seconds.
    synapse : float, optional
        Post-synaptic time constant (PSTC) to use for filtering. Default is
        no filtering.
    seed : int
        The seed used for random number generation in the Connection.
    """

    target = TargetParam(nonzero_size_out=True)
    attr = AttributeParam(default=None)
    sample_every = NumberParam(default=None, optional=True, low=1e-10)
    synapse = SynapseParam(default=None)
    seed = IntParam(default=None, optional=True)

    def __init__(self, target, attr=None,
                 sample_every=Default, synapse=Default, seed=Default):
        self.target = target

        # We'll use the first in the list as default
        self.attr = attr if attr is not None else self.obj.probeable[0]
        self.sample_every = sample_every
        self.synapse = synapse
        self.seed = seed

    @property
    def obj(self):
        return (self.target.obj if isinstance(self.target, ObjView) else
                self.target)

    @property
    def slice(self):
        return (self.target.slice if isinstance(self.target, ObjView) else
                slice(None))

    @property
    def size_in(self):
        return self.target.size_out

    @property
    def size_out(self):
        return 0

    def __str__(self):
        return "<Probe of '%s' of %s>" % (self.attr, self.target)

    def __repr__(self):
        return "<Probe at 0x%x of '%s' of %s>" % (
            id(self), self.attr, self.target)
