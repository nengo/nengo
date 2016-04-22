from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.config import Config
from nengo.connection import Connection, LearningRule
from nengo.exceptions import ObsoleteError, ValidationError
from nengo.params import Default, ConnectionDefault, NumberParam, StringParam
from nengo.solvers import SolverParam
from nengo.synapses import SynapseParam


class TargetParam(NengoObjectParam):
    def validate(self, probe, target):
        obj = target.obj if isinstance(target, ObjView) else target
        if not hasattr(obj, 'probeable'):
            raise ValidationError("Type %r is not probeable"
                                  % obj.__class__.__name__,
                                  attr=self.name, obj=probe)

        # do this after; better to know that type is not Probable first
        if not isinstance(obj, LearningRule):
            super(TargetParam, self).validate(probe, target)


class AttributeParam(StringParam):
    def validate(self, probe, attr):
        super(AttributeParam, self).validate(probe, attr)
        if attr in ('decoders', 'transform'):
            raise ObsoleteError("'decoders' and 'transform' are now combined "
                                "into 'weights'. Probe 'weights' instead.",
                                since="v2.1.0")
        if attr not in probe.obj.probeable:
            raise ValidationError("Attribute %r is not probeable on %s."
                                  % (attr, probe.obj),
                                  attr=self.name, obj=probe)


class ProbeSolverParam(SolverParam):
    def __set__(self, instance, value):
        if value is ConnectionDefault:
            value = Config.default(Connection, 'solver')

        super(ProbeSolverParam, self).__set__(instance, value)

    def validate(self, conn, solver):
        super(ProbeSolverParam, self).validate(conn, solver)
        if solver is not None and solver.weights:
            raise ValidationError("weight solvers only work for ensemble to "
                                  "ensemble connections, not probes",
                                  attr=self.name, obj=conn)


class Probe(NengoObject):
    """A probe is an object that receives data from the simulation.

    This is to be used in any situation where you wish to gather simulation
    data (spike data, represented values, neuron voltages, etc.) for analysis.

    Probes cannot directly affect the simulation.

    All Nengo objects can be probed (except Probes themselves).
    Each object has different attributes that can be probed.
    To see what is probeable for each object, print its `probeable` attribute.

    >>> with nengo.Network():
    ...     ens = nengo.Ensemble(10, 1)
    >>> print(ens.probeable)
    ['decoded_output', 'input']

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
    solver : Solver, optional
        Instance of a Solver class to compute decoders for probes that require
        them (see `nengo.solvers`). Defaults to the same solver as Connection.
    seed : int
        The seed used for random number generation in the Connection.
    label : str, optional
        A name for the probe. Used for debugging and visualization.
    """

    target = TargetParam('target', nonzero_size_out=True)
    attr = AttributeParam('attr', default=None)
    sample_every = NumberParam(
        'sample_ever', default=None, optional=True, low=1e-10)
    synapse = SynapseParam('synapse', default=None)
    solver = ProbeSolverParam('solver', default=ConnectionDefault)

    def __init__(self, target, attr=None, sample_every=Default,
                 synapse=Default, solver=Default, label=Default, seed=Default):
        super(Probe, self).__init__(label=label, seed=seed)
        self.target = target
        self.attr = attr if attr is not None else self.obj.probeable[0]
        self.sample_every = sample_every
        self.synapse = synapse
        self.solver = solver

    def __repr__(self):
        return "<Probe%s at 0x%x of '%s' of %s>" % (
            "" if self.label is None else ' "%s"' % self.label,
            id(self), self.attr, self.target)

    def __str__(self):
        return "<Probe%s of '%s' of %s>" % (
            "" if self.label is None else ' "%s"' % self.label,
            self.attr, self.target)

    @property
    def obj(self):
        return (self.target.obj if isinstance(self.target, ObjView) else
                self.target)

    @property
    def size_in(self):
        return self.target.size_out

    @property
    def size_out(self):
        return 0

    @property
    def slice(self):
        return (self.target.slice if isinstance(self.target, ObjView) else
                None)
