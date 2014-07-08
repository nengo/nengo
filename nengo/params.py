import collections
import warnings

import numpy as np

from nengo.config import Parameter
from nengo.decoders import Solver
from nengo.learning_rules import LearningRule
from nengo.neurons import NeuronType
from nengo.synapses import Lowpass, Synapse
from nengo.utils.compat import is_integer, is_iterable, is_number, is_string
from nengo.utils.distributions import Distribution
from nengo.utils.inspect import checked_call
import nengo.utils.numpy as npext


class BoolParam(Parameter):
    def validate(self, instance, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("Must be a boolean; got '%s'" % boolean)


class NumberParam(Parameter):
    def __init__(self, default, low=None, high=None, optional=False):
        self.low = low
        self.high = high
        if default is not None and low is not None:
            assert default >= low
        if default is not None and high is not None:
            assert default <= high
        if low is not None and high is not None:
            assert low < high
        super(NumberParam, self).__init__(default, optional)

    def validate(self, instance, num):
        if not is_number(num):
            raise ValueError("Must be a number; got '%s'" % num)
        if self.low is not None and num < self.low:
            raise ValueError("Number must be greater than %s" % self.low)
        if self.high is not None and num > self.high:
            raise ValueError("Number must be less than %s" % self.high)


class IntParam(NumberParam):
    def validate(self, instance, num):
        if not is_integer(num):
            raise ValueError("Must be an integer; got '%s'" % num)
        super(IntParam, self).validate(instance, num)


class StringParam(Parameter):
    def validate(self, instance, string):
        if not is_string(string):
            raise ValueError("Must be a string; got '%s'" % string)


class ListParam(Parameter):
    def validate(self, instance, lst):
        if not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % str(lst))


class DictParam(Parameter):
    def validate(self, instance, dct):
        if not isinstance(dct, dict):
            raise ValueError("Must be a dictionary; got '%s'" % str(dct))


class NodeOutputParam(Parameter):
    def __init__(self, default, optional=True):
        assert optional  # None has meaning (passthrough node)
        super(NodeOutputParam, self).__init__(default, optional)

    def __set__(self, node, output):
        # --- Validate and set the new size_out
        if output is None:
            if node.size_out is not None:
                warnings.warn("'Node.size_out' will be overwritten with "
                              "'Node.size_in' when 'Node.output=None'")
            node.size_out = node.size_in
        elif callable(output) and node.size_out is not None:
            # We trust user's size_out if set, because calling output
            # may have unintended consequences (e.g., network communication)
            pass
        elif callable(output):
            result = self.validate_callable(node, output)
            node.size_out = 0 if result is None else result.size
        else:
            # Make into correctly shaped numpy array before validation
            output = npext.array(
                output, min_dims=1, copy=False, dtype=np.float64)
            self.validate_ndarray(node, output)
            node.size_out = output.size

        # --- Set output
        self.data[node] = output

    def validate_callable(self, node, output):
        t, x = np.asarray(0.0), np.zeros(node.size_in)
        args = (t, x) if node.size_in > 0 else (t,)
        result, invoked = checked_call(output, *args)
        if not invoked:
            raise TypeError("output function '%s' must accept %d argument%s" %
                            (output, len(args), 's' if len(args) > 1 else ''))

        if result is not None:
            result = np.asarray(result)
            if len(result.shape) > 1:
                raise ValueError("Node output must be a vector (got shape %s)"
                                 % (result.shape,))
        return result

    def validate_ndarray(self, node, output):
        if len(output.shape) > 1:
            raise ValueError("Node output must be a vector (got shape %s)"
                             % (output.shape,))

        if node.size_in != 0:
            raise TypeError("output must be callable if size_in != 0")

        if node.size_out is not None and node.size_out != output.size:
            raise ValueError("Size of Node output (%d) does not match size_out"
                             "(%d)" % (output.size, node.size_out))


class DistributionParam(Parameter):
    """Can be a Distribution or samples from a distribution."""

    def __init__(self, default, sample_shape, optional=False, readonly=False):
        self.sample_shape = sample_shape
        super(DistributionParam, self).__init__(
            default, optional, readonly)

    def __set__(self, instance, dist):
        self.validate_none(instance, dist)
        self.validate_readonly(instance, dist)

        if isinstance(dist, Distribution):
            self.validate_distribution(instance, dist)
        elif dist is not None:
            dist = self.validate_ndarray(instance, dist)

        self.data[instance] = dist

    def validate_ndarray(self, instance, dist):
        ndim = len(self.sample_shape)
        try:
            dist = np.asarray(dist, dtype=np.float64)
        except ValueError:
            raise ValueError("Must be a Distribution or %dD array" % ndim)

        if self.scalar_ok and dist.size == 1:
            dist.shape = ()
        elif dist.ndim != ndim:
            raise ValueError("Array must be %dD (got %dD)" % (ndim, dist.ndim))
        else:
            for i, attr in enumerate(self.sample_shape):
                if attr == '*':
                    continue
                desired = getattr(instance, attr)
                if dist.shape[i] != desired:
                    raise ValueError("Shape[%d] should be %d (got %d)"
                                     % (i, desired, dist.shape[i]))
        return dist

    def validate_distribution(self, instance, dist):
        assert 0 < len(self.sample_shape) <= 2


class ConnEvalPointsParam(DistributionParam):
    def __set__(self, conn, dist):
        self.validate_pre(conn, dist)
        super(ConnEvalPointsParam, self).__set__(conn, dist)

    def validate_pre(self, conn, dist):
        """Eval points are only valid when pre is an ensemble."""
        from nengo.objects import Ensemble

        if dist is None:
            return

        pre = conn.pre
        if not isinstance(pre, Ensemble):
            raise ValueError("eval_points only work on connections from "
                             "ensembles (got '%s')" % pre.__class__.__name__)


class NeuronTypeParam(Parameter):
    def __set__(self, instance, neurons):
        self.validate_none(instance, neurons)
        if neurons is not None:
            self.validate(instance, neurons)
        self.validate_readonly(instance, neurons)
        if hasattr(instance, 'probeable'):
            self.update_probeable(instance, neurons)
        self.data[instance] = neurons

    def update_probeable(self, instance, neurons):
        """Update the probeable list."""
        # We could use a set instead and this would be easier, but we use
        # the first member of the list as the default probeable, so that
        # doesn't work.
        if instance in self.data and self.data[instance] is not None:
            for attr in self.data[instance].probeable:
                if attr in instance.probeable:
                    instance.probeable.remove(attr)

        if neurons is not None:
            for attr in neurons.probeable:
                if attr not in instance.probeable:
                    instance.probeable.append(attr)

    def validate(self, instance, neurons):
        if not isinstance(neurons, NeuronType):
            raise ValueError("'%s' is not a neuron type" % neurons)


class SynapseParam(Parameter):
    def __init__(self, default, optional=True, readonly=False):
        assert optional  # None has meaning (no filtering)
        super(SynapseParam, self).__init__(
            default, optional, readonly)

    def __set__(self, conn, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        self.validate(conn, synapse)
        self.validate_readonly(conn, synapse)
        self.data[conn] = synapse

    def validate(self, conn, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValueError("'%s' is not a synapse type" % synapse)


class SolverParam(Parameter):
    def validate(self, instance, solver):
        from nengo.objects import Connection, Ensemble
        if not isinstance(solver, Solver):
            raise ValueError("'%s' is not a solver" % solver)
        if isinstance(instance, Connection):
            if solver.weights and not isinstance(instance.pre, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections from ensembles "
                    "(got '%s')" % instance.pre.__class__.__name__)
            if solver.weights and not isinstance(instance.post, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections to ensembles "
                    "(got '%s')" % instance.post.__class__.__name__)


class LearningRuleParam(Parameter):
    def validate(self, instance, rule):
        if is_iterable(rule):
            for lr in rule:
                self.validate_rule(instance, lr)
        elif rule is not None:
            self.validate_rule(instance, rule)

    def validate_rule(self, instance, rule):
        from nengo.objects import Connection
        if not isinstance(rule, LearningRule):
            raise ValueError("'%s' is not a learning rule" % rule)

        if isinstance(instance, Connection):
            rule_type = ('Neurons' if instance.solver.weights
                         else type(instance.pre).__name__)
            if rule_type not in rule.modifies:
                raise ValueError("Learning rule '%s' cannot be applied to "
                                 "connection with pre of type '%s'"
                                 % (rule, type(instance.pre).__name__))


FunctionInfo = collections.namedtuple('FunctionInfo', ['function', 'size'])


class FunctionParam(Parameter):
    def __get__(self, instance, type_):
        value = Parameter.__get__(self, instance, type_)
        return value.function if isinstance(value, FunctionInfo) else value

    def size(self, instance):
        value = Parameter.__get__(self, instance, None)
        return value.size if isinstance(value, FunctionInfo) else value

    def __set__(self, instance, function):
        from nengo.objects import Connection
        self.validate_none(instance, function)
        self.validate_readonly(instance, function)
        self.validate_function(instance, function)

        if function is not None:
            size = self.validate_call(instance, function)
            function = FunctionInfo(function=function, size=size)

        if isinstance(instance, Connection):
            # This validation is Connection specific
            self.validate_connection(instance, function)

        # Set this at the end in case validate_connection fails
        self.data[instance] = function

    def validate_function(self, instance, function):
        if function is not None and not callable(function):
            raise ValueError("function '%s' must be callable" % function)

    def function_args(self, instance, function):
        from nengo.objects import Connection
        if isinstance(instance, Connection):
            x = (instance.eval_points[0] if is_iterable(instance.eval_points)
                 else np.zeros(instance.size_in))
        else:
            x = np.zeros(1)
        return (x,)

    def validate_call(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise TypeError("function '%s' must accept a single "
                            "np.array argument" % function)
        return np.asarray(value).size

    def validate_connection(self, conn, function):
        from nengo.objects import Node, Ensemble
        fn_ok = (Node, Ensemble)

        if function is not None and not isinstance(conn.pre, fn_ok):
            raise ValueError("function can only be set for connections from "
                             "an Ensemble or Node (got type '%s')"
                             % conn.pre.__class__.__name__)

        type_pre = conn.pre.__class__.__name__
        transform = conn.transform
        size_mid = conn.size_in if function is None else function.size

        if transform.ndim < 2 and size_mid != conn.size_out:
            raise ValueError("function output size is incorrect; should "
                             "return a vector of size %d" % conn.size_out)

        if transform.ndim == 2 and size_mid != transform.shape[1]:
            # check input dimensionality matches transform
            raise ValueError(
                "%s output size (%d) not equal to transform input size "
                "(%d)" % (type_pre, size_mid, transform.shape[1]))


class TransformParam(Parameter):
    """The transform additionally validates size_out."""
    def __set__(self, conn, transform):
        self.validate_none(conn, transform)
        transform = np.asarray(transform)
        self.validate(conn, transform)
        self.data[conn] = transform

    def validate(self, conn, transform):
        type_post = conn.post.__class__.__name__
        size_out = conn.size_out

        if transform.ndim == 1 and transform.size != size_out:
            raise ValueError("Transform length (%d) not equal to "
                             "%s output size (%d)" %
                             (transform.size, type_post, size_out))

        if transform.ndim == 2:
            # check output dimensionality matches transform
            if size_out != transform.shape[0]:
                raise ValueError("Transform output size (%d) not equal to "
                                 "%s input size (%d)" %
                                 (transform.shape[0], type_post, size_out))

            # check for repeated dimensions in lists, as these don't work
            # for two-dimensional transforms
            repeated_inds = lambda x: (
                not isinstance(x, slice) and np.unique(x).size != len(x))
            if repeated_inds(conn.pre_slice):
                raise ValueError("Input object selection has repeated indices")
            if repeated_inds(conn.post_slice):
                raise ValueError(
                    "Output object selection has repeated indices")

        if transform.ndim > 2:
            raise ValueError("Cannot handle transform tensors "
                             "with dimensions > 2")

NengoObjectInfo = collections.namedtuple('NengoObjectInfo',
                                         ['obj', 'slice', 'size'])


class NengoObjectParam(Parameter):
    def __init__(self, default=None, optional=False, readonly=True,
                 disallow=None, role='pre'):
        assert default is None  # These can't have defaults
        self.disallow = [] if disallow is None else disallow
        self.role = role
        super(NengoObjectParam, self).__init__(
            default, optional, readonly)

    def __get__(self, instance, type_):
        value = Parameter.__get__(self, instance, type_)
        return value.obj if isinstance(value, NengoObjectInfo) else value

    def size(self, instance):
        value = Parameter.__get__(self, instance, None)
        return value.size if isinstance(value, NengoObjectInfo) else value

    def slice(self, instance):
        value = Parameter.__get__(self, instance, None)
        return value.slice if isinstance(value, NengoObjectInfo) else value

    def __set__(self, instance, nengo_obj):
        from nengo.objects import ObjView
        self.validate_none(instance, nengo_obj)

        if not isinstance(nengo_obj, ObjView):
            nengo_obj = ObjView(nengo_obj)
        nengo_obj.role = self.role
        self.validate(instance, nengo_obj.obj)
        self.validate_readonly(instance, nengo_obj.obj)
        self.data[instance] = NengoObjectInfo(obj=nengo_obj.obj,
                                              slice=nengo_obj.slice,
                                              size=len(nengo_obj))

    def validate(self, instance, nengo_obj):
        from nengo.objects import NengoObject, Neurons
        if not isinstance(nengo_obj, (NengoObject, Neurons)):
            raise ValueError("'%s' is not a Nengo object" % nengo_obj)

        for n_type in self.disallow:
            if isinstance(nengo_obj, n_type):
                raise ValueError("Objects of type '%s' disallowed." % n_type)
