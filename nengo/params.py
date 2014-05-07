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
        super(NumberParam, self).validate(instance, num)


class StringParam(Parameter):
    def validate(self, instance, string):
        if not is_string(string):
            raise ValueError("Must be a string; got '%s'" % string)


class ListParam(Parameter):
    def validate(self, instance, lst):
        if not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % lst)


class NodeOutput(Parameter):
    def __init__(self, default, optional=True, modifies=None):
        assert optional  # None has meaning (passthrough node)
        super(NodeOutput, self).__init__(default, optional, modifies)

    def __set__(self, node, output):
        # --- Validate and set the new size_out
        if output is None:
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

    def __init__(self, default, optional=False, modifies=None,
                 sample_shape=None, scalar_ok=True):
        self.sample_shape = sample_shape
        self.scalar_ok = scalar_ok
        super(DistributionParam, self).__init__(default, optional, modifies)

    def __set__(self, instance, dist):
        self.validate_none(instance, dist)

        if isinstance(dist, Distribution):
            self.validate_distribution(instance, dist)
        elif dist is not None:
            dist = self.validate_ndarray(instance, dist)

        self.data[instance] = dist

    def validate_ndarray(self, instance, dist):
        try:
            dist = np.asarray(dist, dtype=np.float64)
        except ValueError:
            raise ValueError("Must be a Distribution or %dD array"
                             % len(self.sample_shape))

        if self.scalar_ok and dist.size == 1:
            dist.shape = ()
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


class NeuronTypeParam(Parameter):
    def __set__(self, ens, neurons):
        self.validate_none(ens, neurons)
        self.validate(ens, neurons)

        # --- Update the probeable list
        # We could use a set instead and this would be easier, but we use
        # the first member of the list as the default probeable
        if hasattr(ens, 'neuron_type') and ens.neuron_type is not None:
            for attr in ens.neuron_type.probeable:
                if attr in ens.probeable:
                    ens.probeable.remove(attr)

        for attr in neurons.probeable:
            if attr not in ens.probeable:
                ens.probeable.append(attr)

        self.data[ens] = neurons

    def validate(self, ens, neurons):
        if not isinstance(neurons, NeuronType):
            raise ValueError("'%s' is not a neuron type" % neurons)


class SynapseParam(Parameter):
    def __init__(self, default, optional=True, modifies=None):
        assert optional  # None has meaning (no filtering)
        super(SynapseParam, self).__init__(default, optional, modifies)

    def __set__(self, conn, synapse):
        if is_number(synapse):
            synapse = Lowpass(synapse)
        self.validate(conn, synapse)
        self.data[conn] = synapse

    def validate(self, conn, synapse):
        if synapse is not None and not isinstance(synapse, Synapse):
            raise ValueError("'%s' is not a synapse type" % synapse)


class SolverParam(Parameter):
    def validate(self, conn, solver):
        if not isinstance(solver, Solver):
            raise ValueError("'%s' is not a solver" % solver)


class LearningRuleParam(Parameter):
    def validate(self, conn, rule):
        if is_iterable(rule):
            for lr in rule:
                self.validate_rule(conn, lr)
        elif not isinstance(rule, LearningRule):
            raise ValueError("'%s' is not a learning rule" % rule)

    def validate_rule(self, conn, rule):
        if not isinstance(rule, LearningRule):
            raise ValueError("'%s' is not a learning rule" % rule)

        rule_type = ('Neurons' if conn.solver.weights
                     else type(conn.pre).__name__)
        if rule_type not in rule.modifies:
            raise ValueError("Learning rule '%s' cannot be applied to "
                             "connection with pre of type '%s'"
                             % (rule, type(conn.pre).__name__))
