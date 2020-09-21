import warnings

import numpy as np

from nengo.base import NengoObject, NengoObjectParam, ObjView
from nengo.dists import DistOrArrayParam, Distribution
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType, LearningRuleTypeParam
from nengo.neurons import Direct
from nengo.node import Node
from nengo.params import BoolParam, Default, FunctionInfo, Parameter
from nengo.rc import rc
from nengo.solvers import LstsqL2, SolverParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.transforms import Dense, NoTransform
from nengo.utils.functions import function_name
from nengo.utils.numpy import is_array_like, is_iterable
from nengo.utils.stdlib import checked_call


class PrePostParam(NengoObjectParam):
    def coerce(self, conn, nengo_obj):
        if isinstance(nengo_obj, Connection):
            raise ValidationError(
                "Cannot connect to or from connections. "
                "Did you mean to connect to the connection's learning rule?",
                attr=self.name,
                obj=conn,
            )
        return super().coerce(conn, nengo_obj)


class ConnectionLearningRuleTypeParam(LearningRuleTypeParam):
    """Connection-specific validation for learning rules."""

    coerce_defaults = False

    def check_pre(self, conn, rule):
        pre = conn.pre_obj
        if rule.modifies in ("decoders", "weights"):
            # pre object must be neural
            if not isinstance(pre, (Ensemble, Neurons)):
                raise ValidationError(
                    "'pre' must be of type 'Ensemble' or 'Neurons' for "
                    "learning rule '%s' (got type %r)" % (rule, type(pre).__name__),
                    attr=self.name,
                    obj=conn,
                )
            if isinstance(pre, Ensemble) and isinstance(pre.neuron_type, Direct):
                raise ValidationError(
                    "'pre' cannot have neuron type 'Direct'. Connections from "
                    "'Direct' ensembles do not have decoders or weights.",
                    attr=self.name,
                    obj=conn,
                )

    def check_post(self, conn, rule):
        if rule.modifies == "encoders":
            if not isinstance(conn.post_obj, Ensemble):
                raise ValidationError(
                    "'post' must be of type 'Ensemble' (got %r) "
                    "for learning rule '%s'" % (type(conn.post_obj).__name__, rule),
                    attr=self.name,
                    obj=conn,
                )
            if conn._to_neurons:
                raise ValidationError(
                    "Cannot use an encoder learning rule with a direct neural "
                    "connection (since encoders are not used in that case).",
                    attr=self.name,
                    obj=conn,
                )
        else:
            if not isinstance(conn.post_obj, (Ensemble, Neurons, Node)):
                raise ValidationError(
                    "'post' must be of type 'Ensemble', 'Neurons' or 'Node' "
                    "(got %r) for learning rule '%s'"
                    % (type(conn.post_obj).__name__, rule),
                    attr=self.name,
                    obj=conn,
                )

    def check_rule(self, conn, rule):
        super().check_rule(conn, rule)
        self.check_pre(conn, rule)
        self.check_post(conn, rule)
        self.check_transform(conn, rule)

    def check_transform(self, conn, rule):
        if not conn.has_weights and rule.modifies in ("weights", "decoders"):
            raise ValidationError(
                "Learning rule '%s' cannot be applied to a connection that does not "
                "have weights (transform=None)" % rule,
                attr=self.name,
                obj=conn,
            )

        if rule.modifies == "weights":
            # If the rule modifies 'weights', then it must have full weights
            if not conn._to_neurons:
                raise ValidationError(
                    "Learning rule '%s' can only be applied on connections to neurons. "
                    "Try setting `solver.weights` to True or "
                    "connecting between two Neurons objects." % rule,
                    attr=self.name,
                    obj=conn,
                )

            # transform matrix must be 2D
            pre = conn.pre_obj
            pre_size = pre.n_neurons if isinstance(pre, Ensemble) else conn.pre.size_out
            post_size = conn.post.size_in
            if not conn.solver.weights and conn.transform.shape != (
                post_size,
                pre_size,
            ):
                raise ValidationError(
                    "Transform must be 2D array with shape post_neurons x "
                    "pre_neurons (%d, %d)" % (pre_size, post_size),
                    attr=self.name,
                    obj=conn,
                )


class ConnectionSolverParam(SolverParam):
    """Connection-specific validation for decoder solvers."""

    coerce_defaults = False

    def coerce(self, conn, solver):
        solver = super().coerce(conn, solver)
        if solver is not None:
            # it's true that setting the solver on any connection without a pre Ensemble
            # has no effect, but we only warn when ``weights=True`` because in this case
            # we can be sure that it's not the default solver
            if solver.weights and not isinstance(conn.pre_obj, Ensemble):
                warnings.warn(
                    "For connections from objects other than Ensembles, setting the "
                    "solver has no effect"
                )

            if solver.weights and not isinstance(conn.post_obj, Ensemble):
                warnings.warn(
                    "For connections to objects other than Ensembles, setting "
                    "`weights=True` on a solver has no effect"
                )

        return solver


class EvalPointsParam(DistOrArrayParam):
    coerce_defaults = False

    def coerce(self, conn, distorarray):
        """Eval points are only valid when pre is an ensemble."""
        if distorarray is not None and not isinstance(conn.pre, Ensemble):
            msg = (
                "eval_points are only valid on connections from ensembles "
                "(got type '%s')" % type(conn.pre).__name__
            )
            raise ValidationError(msg, attr=self.name, obj=conn)
        return super().coerce(conn, distorarray)


class ConnectionFunctionParam(Parameter):
    """Connection-specific validation for functions."""

    coerce_defaults = False

    def check_array(self, conn, ndarray):
        if not isinstance(conn.eval_points, np.ndarray):
            raise ValidationError(
                "In order to set 'function' to specific points, 'eval_points' "
                "must also be set to specific points.",
                attr=self.name,
                obj=conn,
            )

        if ndarray.ndim != 2:
            raise ValidationError(
                "array must be 2D (got %dD)" % ndarray.ndim, attr=self.name, obj=conn
            )

        if ndarray.shape[0] != conn.eval_points.shape[0]:
            raise ValidationError(
                "Number of evaluation points must match number "
                "of function points (%d != %d)"
                % (ndarray.shape[0], conn.eval_points.shape[0]),
                attr=self.name,
                obj=conn,
            )

    def check_function_can_be_applied(self, conn, function_info):
        function, _ = function_info
        type_pre = type(conn.pre_obj).__name__

        if function is not None:
            if not isinstance(conn.pre_obj, (Node, Ensemble)):
                raise ValidationError(
                    "function can only be set for connections from an Ensemble"
                    " or Node (got type %r)" % type_pre,
                    attr=self.name,
                    obj=conn,
                )

            if isinstance(conn.pre_obj, Node) and conn.pre_obj.output is None:
                raise ValidationError(
                    "Cannot apply functions to passthrough nodes",
                    attr=self.name,
                    obj=conn,
                )

    def coerce(self, conn, function):
        function = super().coerce(conn, function)

        if function is None:
            function_info = FunctionInfo(function=None, size=None)
        elif isinstance(function, FunctionInfo):
            function_info = function
        elif is_array_like(function):
            array = np.array(function, copy=False, dtype=rc.float_dtype)
            self.check_array(conn, array)
            function_info = FunctionInfo(function=array, size=array.shape[1])
        elif callable(function):
            function_info = FunctionInfo(
                function=function, size=self.determine_size(conn, function)
            )
            # TODO: necessary?
            super().coerce(conn, function_info)
        else:
            raise ValidationError(
                "Invalid connection function type %r "
                "(must be callable or array-like)" % type(function).__name__,
                attr=self.name,
                obj=conn,
            )

        self.check_function_can_be_applied(conn, function_info)

        return function_info

    def determine_size(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise ValidationError(
                "function '%s' must accept a single np.array argument" % function,
                attr=self.name,
                obj=instance,
            )
        return np.asarray(value).size

    def function_args(self, conn, function):
        x = (
            conn.eval_points[0]
            if is_iterable(conn.eval_points)
            else np.zeros(conn.size_in)
        )
        return (x,)


class ConnectionTransformParam(Parameter):
    """Connection-specific validation for transforms."""

    coerce_defaults = False

    def coerce(self, conn, transform):
        if transform is None:
            transform = NoTransform(conn.size_mid)
        elif is_array_like(transform) or isinstance(transform, Distribution):
            transform = Dense((conn.size_out, conn.size_mid), transform)

        if transform.size_in != conn.size_mid:
            if isinstance(transform, Dense) and (
                transform.shape[0] == transform.shape[1]
            ):
                # we provide a different error message in this case;
                # the transform is not changing the dimensionality of the
                # signal, so the blame most likely lies with the function
                raise ValidationError(
                    "Function output size is incorrect; should return a "
                    "vector of size %d" % conn.size_mid,
                    attr=self.name,
                    obj=conn,
                )
            else:
                raise ValidationError(
                    "Transform input size (%d) not equal to %s output size "
                    "(%d)"
                    % (transform.size_in, type(conn.pre_obj).__name__, conn.size_mid),
                    attr=self.name,
                    obj=conn,
                )

        if transform.size_out != conn.size_out:
            raise ValidationError(
                "Transform output size (%d) not equal to connection "
                "output size (%d)" % (transform.size_out, conn.size_out),
                attr=self.name,
                obj=conn,
            )

        # we don't support repeated indices on 2D transforms because it makes
        # the matrix multiplication more complicated (we'd need to expand
        # the weight matrix for the duplicated rows/columns). it could be done
        # if there were a demand at some point.
        if isinstance(transform, Dense) and len(transform.init_shape) == 2:

            def repeated_inds(x):
                return not isinstance(x, slice) and np.unique(x).size != len(x)

            if repeated_inds(conn.pre_slice):
                raise ValidationError(
                    "Input object selection has repeated indices",
                    attr=self.name,
                    obj=conn,
                )
            if repeated_inds(conn.post_slice):
                raise ValidationError(
                    "Output object selection has repeated indices",
                    attr=self.name,
                    obj=conn,
                )

        return super().coerce(conn, transform)


class Connection(NengoObject):
    """Connects two objects together.

    The connection between the two object is unidirectional,
    transmitting information from the first argument, ``pre``,
    to the second argument, ``post``.

    Almost any Nengo object can act as the pre or post side of a connection.
    Additionally, you can use Python slice syntax to access only some of the
    dimensions of the pre or post object.

    For example, if ``node`` has ``size_out=2`` and ``ensemble`` has
    ``size_in=1``:

    .. testcode::

       with nengo.Network() as net:
           node = nengo.Node(np.zeros(2))
           ensemble = nengo.Ensemble(10, 1)

    We could not create the following connection:

    .. testcode::

       with net:
           nengo.Connection(node, ensemble)

    .. testoutput::
       :hide:

       Traceback (most recent call last):
       ...
       nengo.exceptions.ValidationError: init: Shape of initial value () does not \
       match expected shape (1, 2)

    But, we could create either of these two connections:

    .. testcode::

       with net:
           nengo.Connection(node[0], ensemble)
           nengo.Connection(node[1], ensemble)

    Parameters
    ----------
    pre : Ensemble or Neurons or Node
        The source Nengo object for the connection.
    post : Ensemble or Neurons or Node or LearningRule
        The destination object for the connection.
    synapse : Synapse or None, optional
        Synapse model to use for filtering (see `~nengo.synapses.Synapse`).
        If *None*, no synapse will be used and information will be transmitted
        without any delay (if supported by the backend---some backends may
        introduce a single time step delay).

        Note that at least one connection must have a synapse that is not
        *None* if components are connected in a cycle. Furthermore, a synaptic
        filter with a zero time constant is different from a *None* synapse
        as a synaptic filter will always add a delay of at least one time step.
    function : callable or (n_eval_points, size_mid) array_like, optional
        Function to compute across the connection. Note that ``pre`` must be
        an ensemble to apply a function across the connection.
        If an array is passed, the function is implicitly defined by the
        points in the array and the provided ``eval_points``, which have a
        one-to-one correspondence.
    transform : (size_out, size_mid) array_like, optional
        Linear transform mapping the pre output to the post input.
        This transform is in terms of the sliced size; if either pre
        or post is a slice, the transform must be shaped according to
        the sliced dimensionality. Additionally, the function is applied
        before the transform, so if a function is computed across the
        connection, the transform must be of shape ``(size_out, size_mid)``.
    solver : Solver, optional
        Solver instance to compute decoders or weights
        (see `~nengo.solvers.Solver`). If ``solver.weights`` is True, a full
        connection weight matrix is computed instead of decoders.
    learning_rule_type : LearningRuleType or iterable of LearningRuleType, optional
        Modifies the decoders or connection weights during simulation.
    eval_points : (n_eval_points, size_in) array_like or int, optional
        Points at which to evaluate ``function`` when computing decoders,
        spanning the interval (-pre.radius, pre.radius) in each dimension.
        If None, will use the eval_points associated with ``pre``.
    scale_eval_points : bool, optional
        Indicates whether the evaluation points should be scaled
        by the radius of the pre Ensemble.
    label : str, optional
        A descriptive label for the connection.
    seed : int, optional
        The seed used for random number generation.

    Attributes
    ----------
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
    solver : Solver
        The Solver instance that will be used to compute decoders or weights
        (see ``nengo.solvers``).
    synapse : Synapse
        The Synapse model used for filtering across the connection
        (see ``nengo.synapses``).
    transform : (size_out, size_mid) array_like
        Linear transform mapping the pre function output to the post input.

    Properties
    ----------
    learning_rule : LearningRule or iterable of LearningRule
        Connectable learning rule object(s) associated with this connection.
    size_in : int
        The number of output dimensions of the pre object.
        Also the input size of the function, if one is specified.
    size_mid : int
        The number of output dimensions of the function, if specified.
        If the function is not specified, then ``size_in == size_mid``.
    size_out : int
        The number of input dimensions of the post object.
        Also the number of output dimensions of the transform.
    """

    probeable = ("output", "input", "weights")

    pre = PrePostParam("pre", nonzero_size_out=True)
    post = PrePostParam("post", nonzero_size_in=True)
    synapse = SynapseParam("synapse", default=Lowpass(tau=0.005))
    function_info = ConnectionFunctionParam("function", default=None, optional=True)
    transform = ConnectionTransformParam("transform", default=None, optional=True)
    solver = ConnectionSolverParam("solver", default=LstsqL2())
    learning_rule_type = ConnectionLearningRuleTypeParam(
        "learning_rule_type", default=None, optional=True
    )
    eval_points = EvalPointsParam(
        "eval_points", default=None, optional=True, sample_shape=("*", "size_in")
    )
    scale_eval_points = BoolParam("scale_eval_points", default=True)

    _param_init_order = [
        "pre",
        "post",
        "synapse",
        "eval_points",
        "function_info",
        "transform",
        "solver",
        "learning_rule_type",
    ]

    def __init__(
        self,
        pre,
        post,
        synapse=Default,
        function=Default,
        transform=Default,
        solver=Default,
        learning_rule_type=Default,
        eval_points=Default,
        scale_eval_points=Default,
        label=Default,
        seed=Default,
    ):
        super().__init__(label=label, seed=seed)

        self.pre = pre
        self.post = post

        self.synapse = synapse
        self.eval_points = eval_points  # Must be set before function
        self.scale_eval_points = scale_eval_points
        self.function_info = function
        self.transform = transform  # Must be set after function
        self.solver = solver  # Must be set before learning rule
        self.learning_rule_type = learning_rule_type  # set after transform

    def __str__(self):
        return self._str(include_id=False)

    def __repr__(self):
        return self._str(include_id=True)

    def _str(self, include_id):
        desc = "<Connection "
        if include_id:
            desc += "at 0x%x " % id(self)

        if self.label is None:
            desc += "from %s to %s%s" % (
                self.pre,
                self.post,
                (
                    ""
                    if self.function is None
                    else " computing '%s'" % (function_name(self.function))
                ),
            )
        else:
            desc += self.label

        desc += ">"

        return desc

    @property
    def function(self):
        return self.function_info.function

    @function.setter
    def function(self, function):
        self.function_info = function

    @property
    def has_weights(self):
        return not isinstance(self.transform, NoTransform) or (
            isinstance(self.pre_obj, Ensemble)
            and not isinstance(self.pre_obj.neuron_type, Direct)
        )

    @property
    def is_decoded(self):
        warnings.warn(
            "is_decoded is deprecated; directly check the pre/post objects for the "
            "properties of interest instead",
            DeprecationWarning,
        )
        return not (
            self.solver.weights
            or (
                isinstance(self.pre_obj, Neurons) and isinstance(self.post_obj, Neurons)
            )
        )

    @property
    def _to_neurons(self):
        return isinstance(self.post_obj, Neurons) or (
            isinstance(self.pre_obj, Ensemble)
            and isinstance(self.post_obj, Ensemble)
            and self.solver.weights
        )

    @property
    def _label(self):
        if self.label is not None:
            return self.label

        return "from %s to %s%s" % (
            self.pre,
            self.post,
            " computing '%s'" % function_name(self.function)
            if self.function is not None
            else "",
        )

    @property
    def learning_rule(self):
        """(LearningRule or iterable) Connectable learning rule object(s)."""
        if self.learning_rule_type is None:
            return None

        types = self.learning_rule_type
        if isinstance(types, dict):
            learning_rule = type(types)()  # dict of same type
            for k, v in types.items():
                learning_rule[k] = LearningRule(self, v)
        elif is_iterable(types):
            learning_rule = [LearningRule(self, v) for v in types]
        elif isinstance(types, LearningRuleType):
            learning_rule = LearningRule(self, types)
        else:
            raise ValidationError(
                "Invalid type %r" % type(types).__name__,
                attr="learning_rule_type",
                obj=self,
            )

        return learning_rule

    @property
    def post_obj(self):
        return self.post.obj if isinstance(self.post, ObjView) else self.post

    @property
    def post_slice(self):
        return self.post.slice if isinstance(self.post, ObjView) else slice(None)

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


class LearningRule:
    """An interface for making connections to a learning rule.

    Connections to a learning rule are to allow elements of the network to
    affect the learning rule. For example, learning rules that use error
    information can obtain that information through a connection.

    Learning rule objects should only ever be accessed through the
    ``learning_rule`` attribute of a connection.
    """

    def __init__(self, connection, learning_rule_type):
        self._connection = connection
        self.learning_rule_type = learning_rule_type

    def __repr__(self):
        return "<LearningRule at 0x%x modifying %r with type %r>" % (
            id(self),
            self.connection,
            self.learning_rule_type,
        )

    def __str__(self):
        return "<LearningRule modifying %s with type %s>" % (
            self.connection,
            self.learning_rule_type,
        )

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self._connection is other._connection
            and self.learning_rule_type == other.learning_rule_type
        )

    def __hash__(self):
        # +1 to avoid collision with ensemble
        return hash(self._connection) + hash(self.learning_rule_type) + 1

    def __getitem__(self, key):
        return ObjView(self, key)

    @property
    def connection(self):
        """(Connection) The connection modified by the learning rule."""
        return self._connection

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
        conn = self.connection
        size_in = self.learning_rule_type.size_in
        if size_in == "pre":
            return conn.size_in
        elif size_in == "mid":
            return conn.size_mid
        elif size_in == "post":
            return conn.size_out
        elif size_in == "pre_state":
            return (
                conn.pre_obj.ensemble.size_out
                if isinstance(conn.pre_obj, Neurons)
                else conn.size_in
            )
        elif size_in == "post_state":
            return (
                conn.post_obj.ensemble.size_in
                if isinstance(conn.post_obj, Neurons)
                else conn.size_out
            )
        else:
            return size_in  # should be an integer

    @property
    def size_out(self):
        """(int) Cannot connect from learning rules, so always 0."""
        return 0  # since a learning rule can't connect to anything
        # TODO: allow probing individual learning rules
