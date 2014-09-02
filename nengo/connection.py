import logging

import numpy as np

from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.ensemble import Ensemble
from nengo.learning_rules import LearningRuleParam
from nengo.node import Node
from nengo.params import (
    Default, BoolParam, FunctionParam, IntParam, ListParam, NdarrayParam)
from nengo.solvers import LstsqL2, SolverParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_iterable

logger = logging.getLogger(__name__)


class ConnectionLearningRuleParam(LearningRuleParam):
    """Connection-specific validation for learning rules."""

    def validate_rule(self, conn, rule):
        super(ConnectionLearningRuleParam, self).validate_rule(conn, rule)
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


class EvalPointsParam(NdarrayParam):
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

    TODO: Document slice syntax here and in the transform parameter.

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
    solver : Solver
        Instance of a Solver class to compute decoders or weights
        (see `nengo.decoders`). If solver.weights is True, a full
        connection weight matrix is computed instead of decoders.
    function : callable, optional
        Function to compute using the pre population (pre must be Ensemble).
    modulatory : bool, optional
        Specifies whether the connection is modulatory (does not physically
        connect to post, for use by learning rules), or not (default).
    eval_points : (n_eval_points, pre_size) array_like or int, optional
        Points at which to evaluate `function` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
    learning_rule : LearningRule or list of LearningRule, optional
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
    learning_rule : list of LearningRule
        The given learning rules. If given a single LearningRule, this will be
        a list with a single element.
    post : Ensemble or Neurons or Node or Probe
        The given pre object.
    pre : Ensemble or Neurons or Node
        The given pre object.
    transform : (post_size, pre_size) array_like
        Linear transform mapping the pre output to the post input.
    modulatory : bool
        Whether the output of this signal is to act as an error signal for a
        learning rule.
    seed : int
        The seed used for random number generation.
    """

    pre = NengoObjectParam(nonzero_size_out=True)
    post = NengoObjectParam(nonzero_size_in=True)
    synapse = SynapseParam(default=Lowpass(0.005))
    transform = TransformParam(default=np.array(1.0))
    solver = ConnectionSolverParam(default=LstsqL2())
    function_info = ConnectionFunctionParam(default=None, optional=True)
    modulatory = BoolParam(default=False)
    learning_rule = ConnectionLearningRuleParam(default=None, optional=True)
    eval_points = EvalPointsParam(
        default=None, optional=True, shape=('*', 'size_in'))
    seed = IntParam(default=None, optional=True)
    probeable = ListParam(default=['output', 'input'])

    def __init__(self, pre, post, synapse=Default, transform=Default,
                 solver=Default, learning_rule=Default, function=Default,
                 modulatory=Default, eval_points=Default, seed=Default):
        self.pre = pre
        self.post = post

        self.probeable = Default
        self.solver = solver  # Must be set before learning rule
        self.learning_rule = learning_rule
        self.modulatory = modulatory
        self.synapse = synapse
        self.transform = transform
        self.eval_points = eval_points  # Must be set before function
        self.function_info = function  # Must be set after transform

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

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
