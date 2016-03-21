import collections
import inspect

import numpy as np

from nengo.exceptions import (
    ConfigError, ObsoleteError, ReadonlyError, ValidationError)
from nengo.utils.compat import (
    is_array, is_integer, is_number, is_string, itervalues)
from nengo.utils.numpy import array_hash, compare
from nengo.utils.stdlib import WeakKeyIDDictionary, checked_call


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
        If true, the parameter can only be set once.
        By default, parameters can be set multiple times.
    """

    equatable = False

    def __init__(self, name,
                 default=Unconfigurable, optional=False, readonly=None):
        # freeze Unconfigurables by default
        readonly = default is Unconfigurable if readonly is None else readonly

        if not is_string(name):
            raise ValueError("'name' must be a string (got %r)" % name)
        if not isinstance(optional, bool):
            raise ValueError("'optional' must be boolean (got %r)" % optional)
        if not isinstance(readonly, bool):
            raise ValueError("'readonly' must be boolean (got %r)" % readonly)

        self.name = name
        self.default = default
        self.optional = optional
        self.readonly = readonly

        # default values set by config system
        self._defaults = WeakKeyIDDictionary()

        # param values set on objects
        self.data = WeakKeyIDDictionary()

    def __contains__(self, key):
        return key in self.data or key in self._defaults

    def __delete__(self, instance):
        del self.data[instance]

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        if not self.configurable and instance not in self.data:
            raise ValidationError(
                "Unconfigurable parameters have no defaults. Please ensure the"
                " value of the parameter is set before trying to access it.",
                attr=self.name, obj=instance)
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        new_value = self.validate(instance, value)
        if new_value is not None:
            value = new_value
        self.data[instance] = value

    def __repr__(self):
        return "%s(%s, default=%s, optional=%s, readonly=%s)" % (
            self.__class__.__name__,
            repr(self.name),
            self.default,
            self.optional,
            self.readonly)

    @property
    def configurable(self):
        return self.default is not Unconfigurable

    def del_default(self, obj):
        del self._defaults[obj]

    def get_default(self, obj):
        return self._defaults.get(obj, self.default)

    def set_default(self, obj, value):
        if not self.configurable:
            raise ConfigError("Parameter '%s' is not configurable" % self)
        new_value = self.validate(obj, value)
        if new_value is not None:
            value = new_value
        self._defaults[obj] = value

    def equal(self, instance_a, instance_b):
        a = self.__get__(instance_a, None)
        b = self.__get__(instance_b, None)
        if self.equatable:
            # always use array_equal, in case one argument is an array
            return np.array_equal(a, b)
        else:
            return a is b

    def hashvalue(self, instance):
        """Returns a hashable value (`hash` can be called on the output)."""
        value = self.__get__(instance, None)
        if self.equatable:
            return value
        else:
            return id(value)

    def validate(self, instance, value):
        if isinstance(value, DefaultType):
            raise ValidationError("Default is not a valid value. To reset a "
                                  "parameter, use 'del'.",
                                  attr=self.name, obj=instance)
        if self.readonly and instance in self.data:
            raise ReadonlyError(attr=self.name, obj=instance)
        if not self.optional and value is None:
            raise ValidationError("Parameter is not optional; cannot set to "
                                  "None", attr=self.name, obj=instance)


class ObsoleteParam(Parameter):
    """A parameter that is no longer supported."""

    def __init__(self, name, short_msg, since=None, url=None):
        self.short_msg = short_msg
        self.since = since
        self.url = url
        super(ObsoleteParam, self).__init__(name, optional=True)

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        self.raise_error()

    def validate(self, instance, value):
        if value is not Unconfigurable:
            # don't allow setting to anything other than unconfigurable default
            self.raise_error()

    def raise_error(self):
        raise ObsoleteError(self.short_msg, since=self.since, url=self.url)


class BoolParam(Parameter):
    """A parameter where the value is a boolean."""

    equatable = True

    def validate(self, instance, boolean):
        if boolean is not None and not isinstance(boolean, bool):
            raise ValidationError("Must be a boolean; got '%s'" % boolean,
                                  attr=self.name, obj=instance)
        return super(BoolParam, self).validate(instance, boolean)


class NumberParam(Parameter):
    """A parameter where the value is a number."""

    equatable = True

    def __init__(self, name, default=Unconfigurable,
                 low=None, high=None, low_open=False, high_open=False,
                 optional=False, readonly=None):
        self.low = low
        self.high = high
        self.low_open = low_open
        self.high_open = high_open
        super(NumberParam, self).__init__(name, default, optional, readonly)

    def __set__(self, instance, value):
        if is_array(value) and value.shape == ():
            value = value.item()  # convert scalar array to Python object
        super(NumberParam, self).__set__(instance, value)

    def validate(self, instance, num):
        if num is not None:
            if not is_number(num):
                raise ValidationError("Must be a number; got '%s'" % num,
                                      attr=self.name, obj=instance)
            low_comp = 0 if self.low_open else -1
            if self.low is not None and compare(num, self.low) <= low_comp:
                raise ValidationError(
                    "Value must be greater than %s%s (got %s)" % (
                        "" if self.low_open else "or equal to ",
                        self.low,
                        num), attr=self.name, obj=instance)
            high_comp = 0 if self.high_open else 1
            if self.high is not None and compare(num, self.high) >= high_comp:
                raise ValidationError(
                    "Value must be less than %s%s (got %s)" % (
                        "" if self.high_open else "or equal to ",
                        self.high,
                        num), attr=self.name, obj=instance)
        return super(NumberParam, self).validate(instance, num)


class IntParam(NumberParam):
    """A parameter where the value is an integer."""

    def validate(self, instance, num):
        if num is not None and not is_integer(num):
            raise ValidationError("Must be an integer; got '%s'" % num,
                                  attr=self.name, obj=instance)
        return super(IntParam, self).validate(instance, num)


class StringParam(Parameter):
    """A parameter where the value is a string."""

    equatable = True

    def validate(self, instance, string):
        if string is not None and not is_string(string):
            raise ValidationError("Must be a string; got '%s'" % string,
                                  attr=self.name, obj=instance)
        return super(StringParam, self).validate(instance, string)


class EnumParam(StringParam):
    """A parameter where the value must be one of a finite set of strings."""

    def __init__(self, name, default=Unconfigurable, values=(), lower=True,
                 optional=False, readonly=None):
        assert all(is_string(s) for s in values)
        if lower:
            values = tuple(s.lower() for s in values)
        value_set = set(values)
        assert len(values) == len(value_set)
        self.values = values
        self.value_set = value_set
        self.lower = lower
        super(EnumParam, self).__init__(name, default, optional, readonly)

    def validate(self, instance, string):
        new_string = super(EnumParam, self).validate(instance, string)
        if new_string is not None:
            string = new_string
        string = string.lower() if self.lower else string
        if string not in self.value_set:
            raise ValidationError("String %r must be one of %s"
                                  % (string, list(self.values)),
                                  attr=self.name, obj=instance)
        return string


class TupleParam(Parameter):
    """A parameter where the value is a tuple."""

    def __init__(self, name, default=Unconfigurable, length=None,
                 optional=False, readonly=None):
        self.length = length
        super(TupleParam, self).__init__(name, default, optional, readonly)

    def __set__(self, instance, value):
        try:
            value = tuple(value)
        except TypeError:
            raise ValidationError("Value must be castable to a tuple",
                                  attr=self.name, obj=instance)
        super(TupleParam, self).__set__(instance, value)

    def validate(self, instance, value):
        if value is not None:
            if self.length is not None and len(value) != self.length:
                raise ValidationError("Must be %d items (got %d)"
                                      % (self.length, len(value)),
                                      attr=self.name, obj=instance)
        return super(TupleParam, self).validate(instance, value)


class DictParam(Parameter):
    """A parameter where the value is a dictionary."""

    def validate(self, instance, dct):
        if dct is not None and not isinstance(dct, dict):
            raise ValidationError("Must be a dictionary; got '%s'" % str(dct),
                                  attr=self.name, obj=instance)
        return super(DictParam, self).validate(instance, dct)


class NdarrayParam(Parameter):
    """A parameter where the value is a NumPy ndarray.

    If the passed value is an ndarray, a view onto that array is stored.
    If the passed value is not an ndarray, it will be cast to an ndarray
    of float64s and stored.
    """

    equatable = True

    def __init__(self, name, default=Unconfigurable, shape=None,
                 optional=False, readonly=None):
        assert shape is not None
        assert shape.count('...') <= 1, "Cannot have more than one ellipsis"
        self.shape = shape
        super(NdarrayParam, self).__init__(name, default, optional, readonly)

    def hashvalue(self, instance):
        return array_hash(self.__get__(instance, None))

    def validate(self, instance, value):
        if value is not None:
            value = self.validate_ndarray(instance, value)
        super(NdarrayParam, self).validate(instance, value)
        return value

    def validate_ndarray(self, instance, ndarray):  # noqa: C901
        if isinstance(ndarray, np.ndarray):
            ndarray = ndarray.view()
        else:
            try:
                ndarray = np.array(ndarray, dtype=np.float64)
            except (ValueError, TypeError):
                raise ValidationError(
                    "Must be a float NumPy array (got type %r)"
                    % ndarray.__class__.__name__, attr=self.name, obj=instance)

        if self.readonly:
            ndarray.setflags(write=False)

        if '...' in self.shape:
            # Convert '...' to the appropriate number of '*'s
            nfixed = len(self.shape) - 1
            n = ndarray.ndim - nfixed
            if n < 0:
                raise ValidationError("ndarray must be at least %dD (got %dD)"
                                      % (nfixed, ndarray.ndim),
                                      attr=self.name, obj=instance)

            i = self.shape.index('...')
            shape = list(self.shape[:i]) + (['*'] * n)
            if i < len(self.shape) - 1:
                shape.extend(self.shape[i+1:])
        else:
            shape = self.shape

        if ndarray.ndim != len(shape):
                raise ValidationError("ndarray must be %dD (got %dD)"
                                      % (len(shape), ndarray.ndim),
                                      attr=self.name, obj=instance)

        for i, attr in enumerate(shape):
            assert is_integer(attr) or is_string(attr), (
                "shape can only be an int or str representing an attribute")
            if attr == '*':
                continue

            desired = attr if is_integer(attr) else getattr(instance, attr)

            if not is_integer(desired):
                raise ValidationError(
                    "%s not yet initialized; cannot determine if shape is "
                    "correct. Consider using a distribution instead." % attr,
                    attr=self.name, obj=instance)

            if ndarray.shape[i] != desired:
                raise ValidationError("shape[%d] should be %d (got %d)"
                                      % (i, desired, ndarray.shape[i]),
                                      attr=self.name, obj=instance)
        return ndarray


FunctionInfo = collections.namedtuple('FunctionInfo', ['function', 'size'])


class FunctionParam(Parameter):
    """A parameter where the value is a function."""

    def __set__(self, instance, function):
        size = (self.determine_size(instance, function)
                if callable(function) else None)
        function_info = FunctionInfo(function=function, size=size)
        super(FunctionParam, self).__set__(instance, function_info)

    def determine_size(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise ValidationError("function '%s' must accept a single "
                                  "np.array argument" % function,
                                  attr=self.name, obj=instance)
        return np.asarray(value).size

    def function_args(self, instance, function):
        return (np.zeros(1),)

    def validate(self, instance, function_info):
        function = function_info.function
        if function is not None and not callable(function):
            raise ValidationError("function '%s' must be callable" % function,
                                  attr=self.name, obj=instance)
        return super(FunctionParam, self).validate(instance, function)


class FrozenObject(object):
    """An object with parameters that cannot change value after instantiation.

    Since such objects are read-only ("frozen"), they can be safely used in
    multiple locations, compared, etc.
    """

    def __init__(self):
        self._paramdict = {
            k: v for k, v in inspect.getmembers(self.__class__)
            if isinstance(v, Parameter)}
        for p in self._params:
            if not p.readonly:
                msg = "All parameters of a FrozenObject must be readonly"
                raise ReadonlyError(attr=p, obj=self, msg=msg)

    @property
    def _params(self):
        return itervalues(self._paramdict)

    def __eq__(self, other):
        if self is other:  # quick check for speed
            return True
        return self.__class__ == other.__class__ and all(
            p.equal(self, other) for p in self._params)

    def __hash__(self):
        return hash((self.__class__, tuple(
            p.hashvalue(self) for p in self._params)))

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop('_paramdict')  # do not pickle the param dict itself
        for k in self._paramdict:
            d[k] = getattr(self, k)

        return d

    def __setstate__(self, state):
        FrozenObject.__init__(self)  # set up the param dict
        for k in self._paramdict:
            setattr(self, k, state.pop(k))
        self.__dict__.update(state)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ', '.join(
            "%s=%r" % (k, getattr(self, k)) for k in sorted(self._paramdict)))


class Deferral(object):
    def __init__(self, fn=None):
        super(Deferral, self).__init__()
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Deferrable(Parameter):
    def __init__(
            self, static, default=None):
        name = static.name
        if default is None:
            default = static.default
        optional = static.optional
        readonly = static.readonly
        super(Deferrable, self).__init__(
            name, default=default, optional=optional, readonly=readonly)
        self.static = static

    def __set__(self, instance, value):
        if not isinstance(value, Deferral):
            self.static.__set__(instance, value)
            value = self.static.__get__(instance, value)
        super(Deferrable, self).__set__(instance, value)

    def validate(self, instance, value):
        if isinstance(value, Deferral):
            return super(Deferrable, self).validate(instance, value)
        else:
            return self.static.validate(instance, value)

    def __repr__(self):
        return "%s(static=%s, default=%s)" % (
            self.__class__.__name__,
            repr(self.static),
            self.default)


class Undeferred(object):
    def __init__(self, inst, *args, **kwargs):
        super(Undeferred, self).__init__()
        self.inst = inst
        self.args = args
        self.kwargs = kwargs
        self.cache = {}

    def __hash__(self):
        return hash(super(Undeferred, self).__getattribute__('inst'))

    def __eq__(self, other):
        return super(Undeferred, self).__getattribute__('inst') == other

    def __getattribute__(self, name):
        inst = super(Undeferred, self).__getattribute__('inst')
        args = super(Undeferred, self).__getattribute__('args')
        kwargs = super(Undeferred, self).__getattribute__('kwargs')
        cache = super(Undeferred, self).__getattribute__('cache')

        if name in cache:
            return cache[name]

        attr = getattr(inst, name)
        if isinstance(attr, Deferral):
            value = attr(*args, **kwargs)
            getattr(inst.__class__, name).validate(inst, value)
            cache[name] = value
            return value
        else:
            return attr
