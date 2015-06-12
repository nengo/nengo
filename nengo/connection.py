import logging
import weakref

import numpy as np

from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.dists import DistOrArrayParam
from nengo.ensemble import Ensemble, Neurons
from nengo.learning_rules import LearningRuleType, LearningRuleTypeParam
from nengo.node import Node
from nengo.params import (
    Default, Unconfigurable, ObsoleteParam, BoolParam, FunctionParam,
    IntParam, NdarrayParam)
from nengo.solvers import LstsqL2, SolverParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_iterable, iteritems

logger = logging.getLogger(__name__)


class ConnectionLearningRuleTypeParam(LearningRuleTypeParam):
    """Connection-specific validation for learning rules."""

    def __set__(self, conn, rule):
        conn._learning_rule = None
        super(ConnectionLearningRuleTypeParam, self).__set__(conn, rule)

    def validate_rule(self, conn, rule):
        super(ConnectionLearningRuleTypeParam, self).validate_rule(conn, rule)
        rule_type = ('Neurons' if conn.solver.weights
                     else type(conn.pre).__name__)
        if rule_type not in rule.modifies:
            raise ValueError("Learning rule '%s' cannot be applied to "
                             "connection with pre of type '%s'"
                             % (rule, type(conn.pre).__name__))


class ConnectionSolverParam(SolverParam):
    """Connection-specific validation for decoder solvers."""

    def validate(self, conn, solver):
        super(ConnectionSolverParam, self).validate(conn, solver)
        if solver is not None:
            if solver.weights and not isinstance(conn.pre, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections from ensembles "
                    "(got '%s')" % conn.pre.__class__.__name__)
            if solver.weights and not isinstance(conn.post, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections to ensembles "
                    "(got '%s')" % conn.post.__class__.__name__)


class EvalPointsParam(DistOrArrayParam):
    def validate(self, conn, ndarray):
        """Eval points are only valid when pre is an ensemble."""
        if not isinstance(conn.pre, Ensemble):
            msg = ("eval_points are only valid on connections from ensembles "
                   "(got type '%s')" % conn.pre.__class__.__name__)
            raise ValueError(msg)
        return super(EvalPointsParam, self).validate(conn, ndarray)


class ConnectionFunctionParam(FunctionParam):
    """Connection-specific validation for functions."""

    def function_args(self, conn, function):
        x = (conn.eval_points[0] if is_iterable(conn.eval_points)
             else np.zeros(conn.size_in))
        return (x,)

    def validate(self, conn, function_info):
        super(ConnectionFunctionParam, self).validate(conn, function_info)
        fn_ok = (Node, Ensemble)
        function, size = function_info

        if function is not None and not isinstance(conn.pre_obj, fn_ok):
            raise ValueError("function can only be set for connections from "
                             "an Ensemble or Node (got type '%s')"
                             % conn.pre_obj.__class__.__name__)

        type_pre = conn.pre_obj.__class__.__name__
        transform = conn.transform
        size_mid = conn.size_in if function is None else size

        if transform.ndim < 2 and size_mid != conn.size_out:
            raise ValueError("function output size is incorrect; should "
                             "return a vector of size %d" % conn.size_out)

        if transform.ndim == 2 and size_mid != transform.shape[1]:
            # check input dimensionality matches transform
            raise ValueError(
                "%s output size (%d) not equal to transform input size "
                "(%d)" % (type_pre, size_mid, transform.shape[1]))

        if (function is not None and isinstance(conn.pre_obj, Node) and
                conn.pre_obj.output is None):
            raise ValueError("Cannot apply functions to passthrough nodes")


class TransformParam(NdarrayParam):
    """The transform additionally validates size_out."""

    def __init__(self, default, optional=False, readonly=False):
        super(TransformParam, self).__init__(default, (), optional, readonly)

    def validate(self, conn, transform):
        transform = np.asarray(transform, dtype=np.float64)

        if transform.ndim == 0:
            self.shape = ()
        elif transform.ndim == 1:
            self.shape = ('size_out',)
        elif transform.ndim == 2:
            # Actually (size_out, size_mid) but Function handles size_mid
            self.shape = ('size_out', '*')
        else:
            raise ValueError("Cannot handle transforms with dimensions > 2")

        # Checks the shapes
        super(TransformParam, self).validate(conn, transform)

        if transform.ndim == 2:
            # check for repeated dimensions in lists, as these don't work
            # for two-dimensional transforms
            repeated_inds = lambda x: (
                not isinstance(x, slice) and np.unique(x).size != len(x))
            if repeated_inds(conn.pre_slice):
                raise ValueError("Input object selection has repeated indices")
            if repeated_inds(conn.post_slice):
                raise ValueError(
                    "Output object selection has repeated indices")

        return transform


class Connection(NengoObject):
    """Connects two objects together.

    Almost any Nengo object can act as the pre or post side of a connection.
    Additionally, you can use Python slice syntax to access only some of the
    dimensions of the pre or post object.

    For example, if ``node`` has ``size_out=2`` and ``ensemble`` has
    ``size_in=1``, we could not create the following connection::

        nengo.Connection(node, ensemble)

    But, we could create either of these two connections.

        nengo.Connection(node[0], ensemble)
        nengo.Connection(ndoe[1], ensemble)

    Parameters
    ----------
    pre : Ensemble or Neurons or Node
        The source Nengo object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    label : string
        A descriptive label for the connection.
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but not including `transform`.
    eval_points : (n_eval_points, pre_size) array_like or int
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    synapse : float, optional
        Post-synaptic time constant (PSTC) to use for filtering.
    transform : (post_size, pre_size) array_like, optional
        Linear transform mapping the pre output to the post input.
        This transform is in terms of the sliced size; if either pre
        or post is a slice, the transform must be of shape
        (len(pre_slice), len(post_slice)).
    solver : Solver
        Instance of a Solver class to compute decoders or weights
        (see `nengo.solvers`). If solver.weights is True, a full
        connection weight matrix is computed instead of decoders.
    function : callable, optional
        Function to compute using the pre population (pre must be Ensemble).
    eval_points : (n_eval_points, pre_size) array_like or int, optional
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    scale_eval_points : bool
        Indicates whether the eval_points should be scaled by the radius of
        the pre Ensemble. Defaults to True.
    learning_rule_type : instance or list or dict of LearningRuleType, optional
        Methods of modifying the connection weights during simulation.

    Attributes
    ----------
    dimensions : int
        The number of output dimensions of the pre object, including
        `function`, but before applying the `transform`.
    function : callable
        The given function.
    function_size : int
        The output dimensionality of the given function. Defaults to 0.
    label : str
        A human-readable connection label for debugging and visualization.
        Incorporates the labels of the pre and post objects.
    learning_rule : LearningRule or collection of LearningRule
        The LearningRule objects corresponding to `learning_rule_type`, and in
        the same format. Use these to probe the learning rules.
    learning_rule_type : instance or list or dict of LearningRuleType, optional
        The learning rule types.
    post : Ensemble or Neurons or Node or Probe
        The given pre object.
    pre : Ensemble or Neurons or Node
        The given pre object.
    transform : (post_size, pre_size) array_like
        Linear transform mapping the pre output to the post input.
    seed : int
        The seed used for random number generation.
    """

    pre = NengoObjectParam(nonzero_size_out=True)
    post = NengoObjectParam(nonzero_size_in=True)
    synapse = SynapseParam(default=Lowpass(0.005))
    transform = TransformParam(default=np.array(1.0))
    solver = ConnectionSolverParam(default=LstsqL2())
    function_info = ConnectionFunctionParam(default=None, optional=True)
    learning_rule_type = ConnectionLearningRuleTypeParam(
        default=None, optional=True)
    eval_points = EvalPointsParam(
        default=None, optional=True, sample_shape=('*', 'size_in'))
    scale_eval_points = BoolParam(default=True)
    seed = IntParam(default=None, optional=True)
    modulatory = ObsoleteParam("Modulatory connections have been removed. "
                               "Connect to a learning rule instead.",
                               "https://github.com/nengo/nengo/issues/632"
                               "#issuecomment-71663849")

    def __init__(self, pre, post, synapse=Default, transform=Default,
                 solver=Default, learning_rule_type=Default, function=Default,
                 eval_points=Default, scale_eval_points=Default, seed=Default,
                 modulatory=Unconfigurable):
        self.pre = pre
        self.post = post

        self.solver = solver  # Must be set before learning rule
        self.learning_rule_type = learning_rule_type
        self.synapse = synapse
        self.transform = transform
        self.scale_eval_points = scale_eval_points
        self.eval_points = eval_points  # Must be set before function
        self.function_info = function  # Must be set after transform
        self.modulatory = modulatory

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

    @property
    def probeable(self):
        probeables = ["output", "input", "transform"]
        if isinstance(self.pre, Ensemble):
            probeables += ["decoders"]

        return probeables

    @property
    def pre_obj(self):
        return self.pre.obj if isinstance(self.pre, ObjView) else self.pre

    @property
    def pre_slice(self):
        return self.pre.slice if isinstance(self.pre, ObjView) else slice(None)

    @property
    def post_obj(self):
        return self.post.obj if isinstance(self.post, ObjView) else self.post

    @property
    def post_slice(self):
        return (self.post.slice if isinstance(self.post, ObjView)
                else slice(None))

    @property
    def size_in(self):
        """Output size of sliced `pre`; input size of the function."""
        return self.pre.size_out

    @property
    def size_mid(self):
        """Output size of the function; input size of the transform.

        If the function is None, then `size_in == size_mid`.
        """
        size = self.function_info.size
        return self.size_in if size is None else size

    @property
    def size_out(self):
        """Output size of the transform; input size to the sliced post."""
        return self.post.size_in

    @property
    def _label(self):
        return "from %s to %s%s" % (
            self.pre, self.post,
            " computing '%s'" % self.function.__name__
            if self.function is not None else "")

    def __str__(self):
        return "<Connection %s>" % self._label

    def __repr__(self):
        return "<Connection at 0x%x %s>" % (id(self), self._label)

    @property
    def learning_rule(self):
        if self.learning_rule_type is not None and self._learning_rule is None:
            types = self.learning_rule_type
            if isinstance(types, dict):
                self._learning_rule = types.__class__()  # dict of same type
                for k, v in iteritems(types):
                    self._learning_rule[k] = LearningRule(self, v)
            elif is_iterable(types):
                self._learning_rule = [LearningRule(self, v) for v in types]
            elif isinstance(types, LearningRuleType):
                self._learning_rule = LearningRule(self, types)
            else:
                raise ValueError("Invalid type for `learning_rule_type`: %s"
                                 % (types.__class__.__name__))
        return self._learning_rule


class LearningRule(object):
    def __init__(self, connection, learning_rule_type):
        self._connection = weakref.ref(connection)
        self.learning_rule_type = learning_rule_type

    def __repr__(self):
        return "<LearningRule at 0x%x modifying %r with type %r>" % (
            id(self), self.connection, self.learning_rule_type)

    def __str__(self):
        return "<LearningRule modifying %s with type %s>" % (
            self.connection, self.learning_rule_type)

    @property
    def connection(self):
        return self._connection()

    @property
    def probeable(self):
        return self.learning_rule_type.probeable

    @property
    def size_in(self):  # size of error signal
        error_type = self.learning_rule_type.error_type.lower()
        if error_type == 'none':
            return 0
        elif error_type == 'decoder':
            if isinstance(self.connection.pre_obj, Neurons):
                return self.connection.pre_obj.ensemble.dimensions
            elif isinstance(self.connection.pre_obj, Ensemble):
                return self.connection.size_mid
            else:
                raise ValueError("Cannot learn on '%s' type" % (
                    self.connection.pre_obj.__class__.__name__))
        elif error_type == 'neuron':
            raise NotImplementedError()
        else:
            raise ValueError("Unrecognized error type '%s'" % (
                self.learning_rule_type.error_type))

    @property
    def size_out(self):
        return 0  # since a learning rule can't connect to anything
        # TODO: allow probing individual learning rules
