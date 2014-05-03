import numpy as np

from nengo.config import Default, Parameter
from nengo.utils.compat import is_integer, is_number, is_string
from nengo.utils.inspect import checked_call
import nengo.utils.numpy as npext


class BoolParam(Parameter):
    def validate(self, instance, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("Must be a boolean; got '%s'" % boolean)
        return True


class NumberParam(Parameter):
    def __init__(self, default, low=None, high=None, mandatory=False):
        self.low = low
        self.high = high
        super(NumberParam, self).__init__(default, mandatory)

    def validate(self, instance, num):
        if not is_number(num):
            raise ValueError("Must be a number; got '%s'" % num)
        if self.low is not None and num < self.low:
            raise ValueError("Number must be greater than %s" % self.low)
        if self.high is not None and num > self.high:
            raise ValueError("Number must be less than %s" % self.high)
        return True


class IntParam(NumberParam):
    def validate(self, instance, num):
        if not is_integer(num):
            raise ValueError("Must be an integer; got '%s'" % num)
        return super(NumberParam, self).validate(instance, num)


class StringParam(Parameter):
    def validate(self, instance, string):
        if not is_string(string):
            raise ValueError("Must be a string; got '%s'" % string)
        return True


class ListParam(Parameter):
    def validate(self, instance, lst):
        if not isinstance(lst, list):
            raise ValueError("Must be a list; got '%s'" % lst)
        return True


class NodeOutput(Parameter):
    def __set__(self, node, output):
        if output is Default:
            output = self.default

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
