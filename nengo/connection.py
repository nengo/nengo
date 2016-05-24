import logging
import weakref

import numpy as np

from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.dists import Distribution, DistOrArrayParam
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType, LearningRuleTypeParam
from nengo.node import Node
from nengo.params import (Default, Unconfigurable, ObsoleteParam,
                          BoolParam, FunctionParam)
from nengo.solvers import LstsqL2, SolverParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_iterable, iteritems

logger = logging.getLogger(__name__)


class PrePostParam(NengoObjectParam):
    def validate(self, conn, nengo_obj):
        super(PrePostParam, self).validate(conn, nengo_obj)
        if isinstance(nengo_obj, Connection):
            raise ValidationError(
                "Cannot connect to or from connections. "
                "Did you mean to connect to the connection's learning rule?",
                attr=self.name, obj=conn)


class ConnectionLearningRuleTypeParam(LearningRuleTypeParam):
    """Connection-specific validation for learning rules."""

    def __set__(self, conn, rule):
        conn._learning_rule = None
        super(ConnectionLearningRuleTypeParam, self).__set__(conn, rule)

    def validate_rule(self, conn, rule):
        super(ConnectionLearningRuleTypeParam, self).validate_rule(conn, rule)

        # --- Check pre object
        if rule.modifies in ('decoders', 'weights'):
            # pre object must be neural
            if not isinstance(conn.pre_obj, (Ensemble, Neurons)):
                raise ValidationError(
                    "pre' must be of type 'Ensemble' or 'Neurons' for "
                    "learning rule '%s' (got type %r)" % (
                        rule, type(conn.pre_obj).__name__),
                    attr=self.name, obj=conn)

        # --- Check post object
        if rule.modifies == 'encoders':
            if not isinstance(conn.post_obj, Ensemble):
                raise ValidationError(
                    "'post' must be of type 'Ensemble' (got %r) "
                    "for learning rule '%s'"
                    % (type(conn.pre_obj).__name__, rule),
                    attr=self.name, obj=conn)
        else:
            if not isinstance(conn.post_obj, (Ensemble, Neurons, Node)):
                raise ValidationError(
                    "'post' must be of type 'Ensemble', 'Neurons' or 'Node' "
                    "(got %r) for learning rule '%s'"
                    % (type(conn.post_obj).__name__, rule),
                    attr=self.name, obj=conn)

        if rule.modifies == 'weights':
            # If the rule modifies 'weights', then it must have full weights
            if conn.is_decoded:
                raise ValidationError(
                    "Learning rule '%s' can not be applied to decoded "
                    "connections. Try setting solver.weights to True or "
                    "connecting between two Neurons objects." % rule,
                    attr=self.name, obj=conn)

            # transform matrix must be 2D
            pre_size = (
                conn.pre_obj.n_neurons if isinstance(conn.pre_obj, Ensemble)
                else conn.pre.size_out)
            post_size = conn.post.size_in
            if (not conn.solver.weights and
                    conn.transform.shape != (post_size, pre_size)):
                raise ValidationError(
                    "Transform must be 2D array with shape post_neurons x "
                    "pre_neurons (%d, %d)" % (pre_size, post_size),
                    attr=self.name, obj=conn)


class ConnectionSolverParam(SolverParam):
    """Connection-specific validation for decoder solvers."""

    def validate(self, conn, solver):
        super(ConnectionSolverParam, self).validate(conn, solver)
        if solver is not None:
            if solver.weights and not isinstance(conn.pre_obj, Ensemble):
                raise ValidationError(
                    "weight solvers only work for connections from ensembles "
                    "(got %r)" % type(conn.pre_obj).__name__,
                    attr=self.name, obj=conn)
            if solver.weights and not isinstance(conn.post_obj, Ensemble):
                raise ValidationError(
                    "weight solvers only work for connections to ensembles "
                    "(got %r)" % type(conn.post_obj).__name__,
                    attr=self.name, obj=conn)


class EvalPointsParam(DistOrArrayParam):
    def validate(self, conn, distorarray):
        """Eval points are only valid when pre is an ensemble."""
        if not isinstance(conn.pre, Ensemble):
            msg = ("eval_points are only valid on connections from ensembles "
                   "(got type '%s')" % type(conn.pre).__name__)
            raise ValidationError(msg, attr=self.name, obj=conn)
        return super(EvalPointsParam, self).validate(conn, distorarray)


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
            raise ValidationError(
                "function can only be set for connections from an Ensemble or "
                "Node (got type %r)" % type(conn.pre_obj).__name__,
                attr=self.name, obj=conn)

        type_pre = type(conn.pre_obj).__name__
        transform = conn.transform
        size_mid = conn.size_in if function is None else size

        if isinstance(transform, np.ndarray):
            if transform.ndim < 2 and size_mid != conn.size_out:
                raise ValidationError(
                    "function output size is incorrect; should return a "
                    "vector of size %d" % conn.size_out, attr=self.name,
                    obj=conn)

            if transform.ndim == 2 and size_mid != transform.shape[1]:
                # check input dimensionality matches transform
                raise ValidationError(
                    "%s output size (%d) not equal to transform input size "
                    "(%d)" % (type_pre, size_mid, transform.shape[1]),
                    attr=self.name, obj=conn)

        if (function is not None and isinstance(conn.pre_obj, Node) and
                conn.pre_obj.output is None):
            raise ValidationError(
                "Cannot apply functions to passthrough nodes",
                attr=self.name, obj=conn)


class TransformParam(DistOrArrayParam):
    """The transform additionally validates size_out."""

    def __init__(self, name, default, optional=False, readonly=False):
        super(TransformParam, self).__init__(
            name, default, (), optional, readonly)

    def validate(self, conn, transform):
        if not isinstance(transform, Distribution):
            # if transform is an array, figure out what the correct shape
            # should be
            transform = np.asarray(transform, dtype=np.float64)

            if transform.ndim == 0:
                self.shape = ()
            elif transform.ndim == 1:
                self.shape = ('size_out',)
            elif transform.ndim == 2:
                # Actually (size_out, size_mid) but Function handles size_mid
                self.shape = ('size_out', '*')

                # check for repeated dimensions in lists, as these don't work
                # for two-dimensional transforms
                def repeated_inds(x):
                    return (not isinstance(x, slice) and
                            np.unique(x).size != len(x))
                if repeated_inds(conn.pre_slice):
                    raise ValidationError(
                        "Input object selection has repeated indices",
                        attr=self.name, obj=conn)
                if repeated_inds(conn.post_slice):
                    raise ValidationError(
                        "Output object selection has repeated indices",
                        attr=self.name, obj=conn)
            else:
                raise ValidationError(
                    "Cannot handle transforms with dimensions > 2",
                    attr=self.name, obj=conn)

        super(TransformParam, self).validate(conn, transform)

        return transform


class Connection(NengoObject):
    """Connects two objects together.

    The connection between the two object is unidirectional,
    transmitting information from the first argument, ``pre``,
    to the second argument, ``post``.

    Almost any Nengo object can act as the pre or post side of a connection.
    Additionally, you can use Python slice syntax to access only some of the
    dimensions of the pre or post object.

    For example, if ``node`` has ``size_out=2`` and ``ensemble`` has
    ``size_in=1``, we could not create the following connection::

        nengo.Connection(node, ensemble)

    But, we could create either of these two connections::

        nengo.Connection(node[0], ensemble)
        nengo.Connection(node[1], ensemble)

    Parameters
    ----------
    pre : Ensemble or Neurons or Node
        The source Nengo object for the connection.
    post : Ensemble or Neurons or Node or Probe
        The destination object for the connection.

    synapse : Synapse, optional \
              (Default: ``nengo.synapses.Lowpass(tau=0.005)``)
        Synapse model to use for filtering (see `~nengo.synapses.Synapse`).
    function : callable, optional (Default: None)
        Function to compute across the connection. Note that ``pre`` must be
        an ensemble to apply a function across the connection.
    transform : (size_out, size_mid) array_like, optional \
                (Default: ``np.array(1.0)``)
        Linear transform mapping the pre output to the post input.
        This transform is in terms of the sliced size; if either pre
        or post is a slice, the transform must be shaped according to
        the sliced dimensionality. Additionally, the function is applied
        before the transform, so if a function is computed across the
        connection, the transform must be of shape ``(size_out, size_mid)``.
    solver : Solver, optional (Default: ``nengo.solvers.LstsqL2()``)
        Solver instance to compute decoders or weights
        (see `~nengo.solvers.Solver`). If ``solver.weights`` is True, a full
        connection weight matrix is computed instead of decoders.
    learning_rule_type : LearningRuleType or iterable of LearningRuleType, \
                         optional (Default: None)
        Modifies the decoders or connection weights during simulation.
    eval_points : (n_eval_points, size_in) array_like or int, optional \
                  (Default: None)
        Points at which to evaluate ``function`` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
        If None, will use the eval_points associated with ``pre``.
    scale_eval_points : bool, optional (Default: True)
        Indicates whether the evaluation points should be scaled
        by the radius of the pre Ensemble.
    label : str, optional (Default: None)
        A descriptive label for the connection.
    seed : int, optional (Default: None)
        The seed used for random number generation.

    Attributes
    ----------
    is_decoded : bool
        True if and only if the connection is decoded. This will not occur
        when ``solver.weights`` is True or both pre and post are
        `~nengo.ensemble.Neurons`.
    function : callable
        The given function.
    function_size : int
        The output dimensionality of the given function. If no function is
        specified, function_size will be 0.
    label : str
        A human-readable connection label for debugging and visualization.
        If not overridden, incorporates the labels of the pre and post objects.
    learning_rule_type : instance or list or dict of LearningRuleType, optional
        The learning rule types.
    post : Ensemble or Neurons or Node or Probe or ObjView
        The given post object.
    post_obj : Ensemble or Neurons or Node or Probe
        The underlying post object, even if ``post`` is an ``ObjView``.
    post_slice : slice or list or None
        The slice associated with ``post`` if it is an ObjView, or None.
    pre : Ensemble or Neurons or Node or ObjView
        The given pre object.
    pre_obj : Ensemble or Neurons or Node
        The underlying pre object, even if ``post`` is an ``ObjView``.
    pre_slice : slice or list or None
        The slice associated with ``pre`` if it is an ObjView, or None.
    seed : int
        The seed used for random number generation.
    size_in : int
        Input size of the computed function, equal to ``pre.size_out``.
    size_mid : int
        Output size of the function ``len(function(np.zeros(size_in)))``,
        equal to ``size_in`` if ``function is None``.
    size_out : int
        Output size of the transform, equal to ``post.size_in``.
    solver : Solver
        The Solver instance that will be used to compute decoders or weights
        (see ``nengo.solvers``).
    synapse : Synapse
        The Synapse model used for filtering across the connection
        (see ``nengo.synapses``).
    transform : (size_out, size_mid) array_like
        Linear transform mapping the pre function output to the post input.
    """

    probeable = ('output', 'input', 'weights')

    pre = PrePostParam('pre', nonzero_size_out=True)
    post = PrePostParam('post', nonzero_size_in=True)
    synapse = SynapseParam('synapse', default=Lowpass(tau=0.005))
    function_info = ConnectionFunctionParam(
        'function', default=None, optional=True)
    transform = TransformParam('transform', default=np.array(1.0))
    solver = ConnectionSolverParam('solver', default=LstsqL2())
    learning_rule_type = ConnectionLearningRuleTypeParam(
        'learning_rule_type', default=None, optional=True)
    eval_points = EvalPointsParam('eval_points',
                                  default=None,
                                  optional=True,
                                  sample_shape=('*', 'size_in'))
    scale_eval_points = BoolParam('scale_eval_points', default=True)
    modulatory = ObsoleteParam(
        'modulatory',
        "Modulatory connections have been removed. "
        "Connect to a learning rule instead.",
        since="v2.1.0",
        url="https://github.com/nengo/nengo/issues/632#issuecomment-71663849")

    def __init__(self, pre, post, synapse=Default, function=Default,
                 transform=Default, solver=Default, learning_rule_type=Default,
                 eval_points=Default, scale_eval_points=Default,
                 label=Default, seed=Default, modulatory=Unconfigurable):
        super(Connection, self).__init__(label=label, seed=seed)

        self.pre = pre
        self.post = post

        self.synapse = synapse
        self.transform = transform
        self.scale_eval_points = scale_eval_points
        self.eval_points = eval_points  # Must be set before function
        self.function_info = function  # Must be set after transform
        self.solver = solver  # Must be set before learning rule
        self.learning_rule_type = learning_rule_type  # set after transform
        self.modulatory = modulatory

    def __str__(self):
        return "<Connection %s>" % self._str

    def __repr__(self):
        return "<Connection at 0x%x %s>" % (id(self), self._str)

    @property
    def _str(self):
        if self.label is not None:
            return self.label

        desc = "" if self.function is None else " computing '%s'" % (
            getattr(self.function, '__name__', str(self.function)))
        return "from %s to %s%s" % (self.pre, self.post, desc)

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

    @property
    def is_decoded(self):
        return not (self.solver.weights or (
            isinstance(self.pre_obj, Neurons) and
            isinstance(self.post_obj, Neurons)))

    @property
    def _label(self):
        if self.label is not None:
            return self.label

        return "from %s to %s%s" % (
            self.pre, self.post,
            " computing '%s'" % self.function.__name__
            if self.function is not None else "")

    @property
    def learning_rule(self):
        """(LearningRule or iterable) Connectable learning rule object(s)."""
        if self.learning_rule_type is not None and self._learning_rule is None:
            types = self.learning_rule_type
            if isinstance(types, dict):
                self._learning_rule = type(types)()  # dict of same type
                for k, v in iteritems(types):
                    self._learning_rule[k] = LearningRule(self, v)
            elif is_iterable(types):
                self._learning_rule = [LearningRule(self, v) for v in types]
            elif isinstance(types, LearningRuleType):
                self._learning_rule = LearningRule(self, types)
            else:
                raise ValidationError(
                    "Invalid type %r" % type(types).__name__,
                    attr='learning_rule_type', obj=self)

        return self._learning_rule

    @property
    def post_obj(self):
        return self.post.obj if isinstance(self.post, ObjView) else self.post

    @property
    def post_slice(self):
        return (self.post.slice if isinstance(self.post, ObjView)
                else slice(None))

    @property
    def pre_obj(self):
        return self.pre.obj if isinstance(self.pre, ObjView) else self.pre

    @property
    def pre_slice(self):
        return self.pre.slice if isinstance(self.pre, ObjView) else slice(None)

    @property
    def size_in(self):
        """(int) The number of output dimensions of the pre object.

        Also the input size of the function, if one is specified.
        """
        return self.pre.size_out

    @property
    def size_mid(self):
        """(int) The number of output dimensions of the function, if specified.

        If the function is not specified, then ``size_in == size_mid``.
        """
        size = self.function_info.size
        return self.size_in if size is None else size

    @property
    def size_out(self):
        """(int) The number of input dimensions of the post object.

        Also the number of output dimensions of the transform.
        """
        return self.post.size_in


class LearningRule(object):
    """An interface for making connections to a learning rule.

    Connections to a learning rule are to allow elements of the network to
    affect the learning rule. For example, learning rules that use error
    information can obtain that information through a connection.

    Learning rule objects should only ever be accessed through the
    ``learning_rule`` attribute of a connection.
    """

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
        """(Connection) The connection modified by the learning rule."""
        return self._connection()

    @property
    def error_type(self):
        """(str) The type of information expected by the learning rule."""
        return self.learning_rule_type.error_type

    @property
    def modifies(self):
        """(str) The variable modified by the learning rule."""
        return self.learning_rule_type.modifies

    @property
    def probeable(self):
        """(tuple) Signals that can be probed in the learning rule."""
        return self.learning_rule_type.probeable

    @property
    def size_in(self):
        """(int) Dimensionality of the signal expected by the learning rule."""
        if self.error_type == 'none':
            return 0
        elif self.error_type == 'scalar':
            return 1
        elif self.error_type == 'decoded':
            return (self.connection.post_obj.ensemble.size_in
                    if isinstance(self.connection.post_obj, Neurons) else
                    self.connection.size_out)
        elif self.error_type == 'neuron':
            raise NotImplementedError()
        else:
            raise ValidationError(
                "Unrecognized error type %r" % self.error_type,
                attr='error_type', obj=self)

    @property
    def size_out(self):
        """(int) Cannot connect from learning rules, so always 0."""
        return 0  # since a learning rule can't connect to anything
        # TODO: allow probing individual learning rules
