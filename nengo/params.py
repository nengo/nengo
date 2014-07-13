import warnings
import weakref

import numpy as np

from nengo.decoders import Solver
from nengo.learning_rules import LearningRule
from nengo.neurons import NeuronType
from nengo.synapses import Lowpass, Synapse
from nengo.utils.compat import is_integer, is_iterable, is_number, is_string
from nengo.utils.distributions import Distribution
from nengo.utils.inspect import checked_call
import nengo.utils.numpy as npext


class DefaultType:
    def __repr__(self):
        return "Default"
Default = DefaultType()


def is_param(obj):
    return isinstance(obj, Parameter)


class Parameter(object):
    """Simple descriptor for storing configuration parameters.

    Parameters
    ----------
    default : object
        The value returned if the parameter hasn't been explicitly set.
    optional : bool, optional
        Whether this parameter accepts the value None. By default,
        parameters are not optional (i.e., cannot be set to ``None``).
    readonly : bool, optional
        Whether the parameter can be set multiple times.
        By default, parameters can be set multiple times.
    """
    def __init__(self, default, optional=False, readonly=False):
        self.default = default
        self.optional = optional
        self.readonly = readonly
        # readonly Parameters must have default=None
        assert not readonly or default is None
        # use WeakKey dictionaries so items can still be garbage collected
        self.defaults = weakref.WeakKeyDictionary()
        self.data = weakref.WeakKeyDictionary()

    def __contains__(self, key):
        return key in self.data or key in self.defaults

    def __delete__(self, instance):
        del self.data[instance]

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.validate(instance, value)
        self.data[instance] = value

    def __repr__(self):
        return "%s(default=%s, optional=%s, readonly=%s)" % (
            self.__class__.__name__,
            self.default,
            self.optional,
            self.readonly)

    def validate(self, instance, value):
        if value is Default:
            raise ValueError("Default is not a valid value.")
        if self.readonly and instance in self.data:
            raise ValueError("Parameter is read-only; cannot be changed.")
        if not self.optional and value is None:
            raise ValueError("Parameter is not optional; cannot set to None")


class BoolParam(Parameter):
    def validate(self, instance, boolean):
        if boolean is not None and not isinstance(boolean, bool):
            raise ValueError("Must be a boolean; got '%s'" % boolean)
        super(BoolParam, self).validate(instance, boolean)


class NumberParam(Parameter):
    def __init__(self, default, low=None, high=None,
                 optional=False, readonly=False):
        self.low = low
        self.high = high
        super(NumberParam, self).__init__(default, optional, readonly)

    def validate(self, instance, num):
        if num is not None:
            if not is_number(num):
                raise ValueError("Must be a number; got '%s'" % num)
            if self.low is not None and num < self.low:
                raise ValueError("Number must be greater than %s" % self.low)
            if self.high is not None and num > self.high:
                raise ValueError("Number must be less than %s" % self.high)
        super(NumberParam, self).validate(instance, num)


class IntParam(NumberParam):
    def validate(self, instance, num):
        if num is not None and not is_integer(num):
            raise ValueError("Must be an integer; got '%s'" % num)
        super(IntParam, self).validate(instance, num)


class StringParam(Parameter):
    def validate(self, instance, string):
        if string is not None and not is_string(string):
            raise ValueError("Must be a string; got '%s'" % string)
        super(StringParam, self).validate(instance, string)


class ListParam(Parameter):
    def validate(self, instance, lst):
        if lst is not None and not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % str(lst))
        super(ListParam, self).validate(instance, lst)


class DictParam(Parameter):
    def validate(self, instance, dct):
        if dct is not None and not isinstance(dct, dict):
            raise ValueError("Must be a dictionary; got '%s'" % str(dct))
        super(DictParam, self).validate(instance, dct)


class NodeOutputParam(Parameter):
    def __init__(self, default, optional=True, readonly=False):
        assert optional  # None has meaning (passthrough node)
        super(NodeOutputParam, self).__init__(default, optional, readonly)

    def __set__(self, node, output):
        super(NodeOutputParam, self).validate(node, output)

        # --- Validate and set the new size_out
        if output is None:
            if node.size_out is not None:
                warnings.warn("'Node.size_out' is being overwritten with "
                              "'Node.size_in' since 'Node.output=None'")
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


class NdarrayParam(Parameter):
    """Can be a NumPy ndarray, or something that can be coerced into one."""

    def __init__(self, default, shape, optional=False, readonly=False):
        assert shape is not None
        self.shape = shape
        super(NdarrayParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, ndarray):
        super(NdarrayParam, self).validate(instance, ndarray)
        if ndarray is not None:
            ndarray = self.validate(instance, ndarray)
        self.data[instance] = ndarray

    def validate(self, instance, ndarray):
        ndim = len(self.shape)
        try:
            ndarray = np.asarray(ndarray, dtype=np.float64)
        except TypeError:
            raise ValueError("Must be a float NumPy array (got type '%s')"
                             % ndarray.__class__.__name__)

        if ndarray.ndim != ndim:
            raise ValueError("ndarray must be %dD (got %dD)"
                             % (ndim, ndarray.ndim))
        for i, attr in enumerate(self.shape):
            assert is_integer(attr) or is_string(attr), (
                "shape can only be an int or str representing an attribute")
            if attr == '*':
                continue

            if is_integer(attr):
                desired = attr
            elif is_string(attr):
                desired = getattr(instance, attr)

            if ndarray.shape[i] != desired:
                raise ValueError("shape[%d] should be %d (got %d)"
                                 % (i, desired, ndarray.shape[i]))
        return ndarray


class DistributionParam(NdarrayParam):
    """Can be a Distribution or samples from a distribution."""

    def __init__(self, default, sample_shape, optional=False, readonly=False):
        super(DistributionParam, self).__init__(
            default, sample_shape, optional, readonly)

    def validate(self, instance, dist):
        if dist is not None and not isinstance(dist, Distribution):
            try:
                dist = super(DistributionParam, self).validate(instance, dist)
            except ValueError:
                raise ValueError("Must be a distribution or NumPy array")
        return dist


class ConnEvalPointsParam(NdarrayParam):
    def __set__(self, conn, ndarray):
        if ndarray is not None:
            self.validate_pre(conn, ndarray)
        super(ConnEvalPointsParam, self).__set__(conn, ndarray)

    def validate_pre(self, conn, ndarray):
        """Eval points are only valid when pre is an ensemble."""
        from nengo.objects import Ensemble
        if not isinstance(conn.pre, Ensemble):
            msg = ("eval_points are only valid on connections from ensembles "
                   "(got type '%s')" % conn.pre.__class__.__name__)
            raise ValueError(msg)


class NeuronTypeParam(Parameter):
    def __set__(self, instance, neurons):
        self.validate(instance, neurons)
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
        if neurons is not None and not isinstance(neurons, NeuronType):
            raise ValueError("'%s' is not a neuron type" % neurons)
        super(NeuronTypeParam, self).validate(instance, neurons)


class SynapseParam(Parameter):
    def __init__(self, default, optional=True, readonly=False):
        assert optional  # None has meaning (no filtering)
        super(SynapseParam, self).__init__(
            default, optional, readonly)

    def __set__(self, conn, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        self.validate(conn, synapse)
        self.data[conn] = synapse

    def validate(self, conn, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValueError("'%s' is not a synapse type" % synapse)
        super(SynapseParam, self).validate(conn, synapse)


class SolverParam(Parameter):
    def validate(self, instance, solver):
        from nengo.objects import Connection, Ensemble
        if solver is not None and not isinstance(solver, Solver):
            raise ValueError("'%s' is not a solver" % solver)
        if solver is not None and isinstance(instance, Connection):
            if solver.weights and not isinstance(instance.pre, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections from ensembles "
                    "(got '%s')" % instance.pre.__class__.__name__)
            if solver.weights and not isinstance(instance.post, Ensemble):
                raise ValueError(
                    "weight solvers only work for connections to ensembles "
                    "(got '%s')" % instance.post.__class__.__name__)
        super(SolverParam, self).validate(instance, solver)


class LearningRuleParam(Parameter):
    def validate(self, instance, rule):
        if is_iterable(rule):
            for lr in rule:
                self.validate_rule(instance, lr)
        elif rule is not None:
            self.validate_rule(instance, rule)
        super(LearningRuleParam, self).validate(instance, rule)

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


class NengoObjectParam(Parameter):
    def __init__(self, default=None, disallow=None, optional=False,
                 readonly=True):
        assert default is None  # These can't have defaults
        self.disallow = [] if disallow is None else disallow
        super(NengoObjectParam, self).__init__(default, optional, readonly)

    def validate(self, instance, nengo_obj):
        from nengo.objects import NengoObject, Neurons, ObjView
        if not isinstance(nengo_obj, (NengoObject, Neurons, ObjView)):
            raise ValueError("'%s' is not a Nengo object" % nengo_obj)
        for n_type in self.disallow:
            if isinstance(nengo_obj, n_type):
                raise ValueError("Objects of type '%s' disallowed." % n_type)
        super(NengoObjectParam, self).validate(instance, nengo_obj)
