import collections
import weakref

import numpy as np

from nengo.dists import Distribution
from nengo.processes import StochasticProcess
from nengo.utils.compat import is_integer, is_number, is_string
from nengo.utils.numpy import compare
from nengo.utils.stdlib import checked_call


class DefaultType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Default = DefaultType("Default")
ConnectionDefault = DefaultType("ConnectionDefault")
Unconfigurable = DefaultType("Unconfigurable")


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
        assert not readonly or default in [None, Unconfigurable]
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

    @property
    def is_configurable(self):
        return self.default is not Unconfigurable

    def validate(self, instance, value):
        if value is Default:
            raise ValueError("Default is not a valid value. To reset a "
                             "parameter, use `del`.")
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
    def __init__(self, default,
                 low=None, high=None, low_open=False, high_open=False,
                 optional=False, readonly=False):
        self.low = low
        self.high = high
        self.low_open = low_open
        self.high_open = high_open
        super(NumberParam, self).__init__(default, optional, readonly)

    def __set__(self, instance, value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()  # convert scalar array to Python object
        super(NumberParam, self).__set__(instance, value)

    def validate(self, instance, num):
        if num is not None:
            if not is_number(num):
                raise ValueError("Must be a number; got '%s'" % num)
            low_comp = 0 if self.low_open else -1
            if self.low is not None and compare(num, self.low) <= low_comp:
                raise ValueError("Value must be greater than %s%s (got %s)" % (
                    "" if self.low_open else "or equal to ", self.low, num))
            high_comp = 0 if self.high_open else 1
            if self.high is not None and compare(num, self.high) >= high_comp:
                raise ValueError("Value must be less than %s%s (got %s)" % (
                    "" if self.high_open else "or equal to ", self.high, num))
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

            desired = attr if is_integer(attr) else getattr(instance, attr)

            if not is_integer(desired):
                raise ValueError("%s not yet initialized; cannot determine "
                                 "if shape is correct. Consider using a "
                                 "distribution instead." % attr)

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
            dist = super(DistributionParam, self).validate(instance, dist)
        return dist


class StochasticProcessParam(Parameter):
    """Can be a StochasticProcess."""

    def validate(self, instance, process):
        super(StochasticProcessParam, self).validate(instance, process)

        if process is not None and not isinstance(process, StochasticProcess):
            raise ValueError(
                "Must be StochasticProcess (got type {0}).".format(
                    process.__class__.__name__))

        return process


FunctionInfo = collections.namedtuple('FunctionInfo', ['function', 'size'])


class FunctionParam(Parameter):
    def __set__(self, instance, function):
        size = (self.determine_size(instance, function)
                if callable(function) else None)
        function_info = FunctionInfo(function=function, size=size)
        super(FunctionParam, self).__set__(instance, function_info)

    def function_args(self, instance, function):
        return (np.zeros(1),)

    def determine_size(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise TypeError("function '%s' must accept a single "
                            "np.array argument" % function)
        return np.asarray(value).size

    def validate(self, instance, function_info):
        function = function_info.function
        if function is not None and not callable(function):
            raise ValueError("function '%s' must be callable" % function)
        super(FunctionParam, self).validate(instance, function)
