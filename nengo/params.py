import collections
import inspect

import numpy as np

from nengo.exceptions import ConfigError, ObsoleteError, ReadonlyError, ValidationError
from nengo.rc import rc
from nengo.utils.numpy import (
    array_hash,
    compare,
    is_array,
    is_array_like,
    is_integer,
    is_number,
)
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


def iter_params(obj):
    """Iterate over the names of all parameters of an object."""
    obj = obj if inspect.isclass(obj) else type(obj)
    return (
        name
        for name in dir(obj)
        if is_param(getattr(obj, name))
        and not isinstance(getattr(obj, name), ObsoleteParam)
    )


def equal(a, b):
    if is_array_like(a) or is_array_like(b):
        return np.array_equal(a, b)
    else:
        return a == b


class Parameter:
    """Simple descriptor for storing configuration parameters.

    Parameters
    ----------
    name : str
        Name of the parameter.
    default : object
        The value returned if the parameter hasn't been explicitly set.
    optional : bool, optional
        Whether this parameter accepts the value None. By default,
        parameters are not optional (i.e., cannot be set to ``None``).
    readonly : bool, optional
        If true, the parameter can only be set once.
        By default, parameters can be set multiple times.

    Attributes
    ----------
    coerce_defaults : bool
        If True, validate values for this parameter when they are set in a
        `.Config` object. Setting a parameter directly on an object will
        always be validated.
    equatable : bool
        If True, parameter values can be compared for equality
        (``a==b``); otherwise equality checks will just compare object
        identity (``a is b``).
    """

    coerce_defaults = True
    equatable = False

    def __init__(self, name, default=Unconfigurable, optional=False, readonly=None):
        # freeze Unconfigurables by default
        readonly = default is Unconfigurable if readonly is None else readonly

        if not isinstance(name, str):
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

    def __getstate__(self):
        state = {}
        state.update(self.__dict__)
        state["_defaults"] = dict(state["_defaults"].items())
        state["data"] = dict(state["data"].items())
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            if k in ["_defaults", "data"]:
                v = WeakKeyIDDictionary(v)
            setattr(self, k, v)

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
                attr=self.name,
                obj=instance,
            )
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = self.coerce(instance, value)

    def __repr__(self):
        return "%s(%r, default=%s, optional=%s, readonly=%s)" % (
            type(self).__name__,
            self.name,
            self.default,
            self.optional,
            self.readonly,
        )

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
        self._defaults[obj] = self.coerce(obj, value) if self.coerce_defaults else value

    def check_type(self, instance, value, type_):
        if value is not None and not isinstance(value, type_):
            if isinstance(type_, tuple):
                type_str = " or ".join((t.__name__ for t in type_))
            else:
                type_str = type_.__name__
            raise ValidationError(
                "Must be of type %r (got type %r)." % (type_str, type(value).__name__),
                attr=self.name,
                obj=instance,
            )

    def coerce(self, instance, value):
        if isinstance(value, DefaultType):
            raise ValidationError(
                "Default is not a valid value. To reset a " "parameter, use 'del'.",
                attr=self.name,
                obj=instance,
            )
        if self.readonly and instance in self.data:
            raise ReadonlyError(attr=self.name, obj=instance)
        if not self.optional and value is None:
            raise ValidationError(
                "Parameter is not optional; cannot set to " "None",
                attr=self.name,
                obj=instance,
            )
        return value

    def equal(self, instance_a, instance_b):
        a = self.__get__(instance_a, None)
        b = self.__get__(instance_b, None)
        if self.equatable:
            return equal(a, b)
        else:
            return a is b

    def hashvalue(self, instance):
        """Returns a hashable value (`hash` can be called on the output)."""
        value = self.__get__(instance, None)
        if self.equatable:
            return value
        else:
            return id(value)


class ObsoleteParam(Parameter):
    """A parameter that is no longer supported."""

    def __init__(self, name, short_msg, since=None, url=None):
        self.short_msg = short_msg
        self.since = since
        self.url = url
        super().__init__(name, optional=True)

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        self.raise_error()

    def coerce(self, instance, value):
        if value is not Unconfigurable:
            # don't allow setting to anything other than unconfigurable default
            self.raise_error()
        return value

    def raise_error(self):
        raise ObsoleteError(self.short_msg, since=self.since, url=self.url)


class BoolParam(Parameter):
    """A parameter where the value is a boolean."""

    equatable = True

    def coerce(self, instance, value):
        self.check_type(instance, value, bool)
        return super().coerce(instance, value)


class NumberParam(Parameter):
    """A parameter where the value is a number."""

    equatable = True

    def __init__(
        self,
        name,
        default=Unconfigurable,
        low=None,
        high=None,
        low_open=False,
        high_open=False,
        optional=False,
        readonly=None,
    ):
        self.low = low
        self.high = high
        self.low_open = low_open
        self.high_open = high_open
        super().__init__(name, default, optional, readonly)

    def coerce(self, instance, num):
        if num is not None:
            if is_array(num) and num.shape == ():
                num = num.item()  # convert scalar array to Python object

            if not is_number(num):
                raise ValidationError(
                    "Must be a number; got '%s'" % num, attr=self.name, obj=instance
                )
            low_comp = 0 if self.low_open else -1
            if self.low is not None and compare(num, self.low) <= low_comp:
                raise ValidationError(
                    "Value must be greater than %s%s (got %s)"
                    % ("" if self.low_open else "or equal to ", self.low, num),
                    attr=self.name,
                    obj=instance,
                )
            high_comp = 0 if self.high_open else 1
            if self.high is not None and compare(num, self.high) >= high_comp:
                raise ValidationError(
                    "Value must be less than %s%s (got %s)"
                    % ("" if self.high_open else "or equal to ", self.high, num),
                    attr=self.name,
                    obj=instance,
                )
        return super().coerce(instance, num)


class IntParam(NumberParam):
    """A parameter where the value is an integer."""

    def coerce(self, instance, num):
        self.check_type(instance, num, (int, np.integer))
        return super().coerce(instance, num)


class StringParam(Parameter):
    """A parameter where the value is a string."""

    equatable = True

    def coerce(self, instance, string):
        self.check_type(instance, string, (str,))
        return super().coerce(instance, string)


class EnumParam(StringParam):
    """A parameter where the value must be one of a finite set of strings."""

    def __init__(
        self,
        name,
        default=Unconfigurable,
        values=(),
        lower=True,
        optional=False,
        readonly=None,
    ):
        assert all(isinstance(s, str) for s in values)
        if lower:
            values = tuple(s.lower() for s in values)
        value_set = set(values)
        assert len(values) == len(value_set)
        self.values = values
        self.value_set = value_set
        self.lower = lower
        super().__init__(name, default, optional, readonly)

    def coerce(self, instance, string):
        string = super().coerce(instance, string)
        string = string.lower() if self.lower else string
        if string not in self.value_set:
            raise ValidationError(
                "String %r must be one of %s" % (string, list(self.values)),
                attr=self.name,
                obj=instance,
            )
        return string


class TupleParam(Parameter):
    """A parameter where the value is a tuple."""

    def __init__(
        self, name, default=Unconfigurable, length=None, optional=False, readonly=None
    ):
        self.length = length
        super().__init__(name, default, optional, readonly)

    def coerce(self, instance, value):
        value = super().coerce(instance, value)
        try:
            value = tuple(value)
        except TypeError:
            raise ValidationError(
                "Value must be castable to a tuple", attr=self.name, obj=instance
            )
        if value is not None:
            if self.length is not None and len(value) != self.length:
                raise ValidationError(
                    "Must be %d items (got %d)" % (self.length, len(value)),
                    attr=self.name,
                    obj=instance,
                )
        return value


class ShapeParam(TupleParam):
    """A parameter where the value is a tuple of integers."""

    equatable = True

    def __init__(
        self,
        name,
        default=Unconfigurable,
        length=None,
        low=0,
        optional=False,
        readonly=None,
    ):
        super().__init__(
            name, default=default, length=length, optional=optional, readonly=readonly
        )
        self.low = low

    def coerce(self, instance, value):
        value = super().coerce(instance, value)
        for i, v in enumerate(value):
            if not is_integer(v):
                raise ValidationError(
                    "Element %d must be an int (got type %r)" % (i, type(v).__name__),
                    attr=self.name,
                    obj=instance,
                )
            if self.low is not None and v < self.low:
                raise ValidationError(
                    "Element %d must be >= %d (got %d)" % (i, self.low, v),
                    attr=self.name,
                    obj=instance,
                )

        return value


class DictParam(Parameter):
    """A parameter where the value is a dictionary."""

    def coerce(self, instance, value):
        self.check_type(instance, value, dict)
        return super().coerce(instance, value)


class NdarrayParam(Parameter):
    """A parameter where the value is a NumPy ndarray.

    If the passed value is an ndarray, a view onto that array is stored.
    If the passed value is not an ndarray, it will be cast to an ndarray
    of ``dtype`` and stored.
    """

    equatable = True

    def __init__(
        self,
        name,
        default=Unconfigurable,
        shape=None,
        dtype=None,
        optional=False,
        readonly=None,
    ):
        if shape is not None:
            assert shape.count("...") <= 1, "Cannot have more than one ellipsis"
        self.shape = shape
        self._dtype = dtype
        super().__init__(name, default, optional, readonly)

    @property
    def coerce_defaults(self):
        if self.shape is None:
            return True
        return all(is_integer(dim) or dim in ("...", "*") for dim in self.shape)

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        return rc.float_dtype

    def hashvalue(self, instance):
        return array_hash(self.__get__(instance, None))

    def coerce(self, instance, value):
        if value is not None:
            value = self.coerce_ndarray(instance, value)
        return super().coerce(instance, value)

    def coerce_ndarray(self, instance, ndarray):  # noqa: C901
        if isinstance(ndarray, np.ndarray):
            ndarray = ndarray.view()
        else:
            try:
                ndarray = np.array(ndarray, dtype=self.dtype)
            except (ValueError, TypeError):
                raise ValidationError(
                    "Must be a %s NumPy array (got type %r)"
                    % (self.dtype, type(ndarray).__name__),
                    attr=self.name,
                    obj=instance,
                )

        if self.readonly:
            ndarray.setflags(write=False)

        if self.shape is None:
            return ndarray

        if "..." in self.shape:
            # Convert '...' to the appropriate number of '*'s
            nfixed = len(self.shape) - 1
            n = ndarray.ndim - nfixed
            if n < 0:
                raise ValidationError(
                    "ndarray must be at least %dD (got %dD)" % (nfixed, ndarray.ndim),
                    attr=self.name,
                    obj=instance,
                )

            i = self.shape.index("...")
            shape = list(self.shape[:i]) + (["*"] * n)
            if i < len(self.shape) - 1:
                shape.extend(self.shape[i + 1 :])
        else:
            shape = self.shape

        if ndarray.ndim != len(shape):
            raise ValidationError(
                "ndarray must be %dD (got %dD)" % (len(shape), ndarray.ndim),
                attr=self.name,
                obj=instance,
            )

        for i, attr in enumerate(shape):
            assert is_integer(attr) or isinstance(
                attr, str
            ), "shape can only be an int or str representing an attribute"
            if attr == "*":
                continue

            desired = attr if is_integer(attr) else getattr(instance, attr)

            if not is_integer(desired):
                raise ValidationError(
                    "%s not yet initialized; cannot determine if shape is "
                    "correct. Consider using a distribution instead." % attr,
                    attr=self.name,
                    obj=instance,
                )

            if ndarray.shape[i] != desired:
                raise ValidationError(
                    "shape[%d] should be %d (got %d)" % (i, desired, ndarray.shape[i]),
                    attr=self.name,
                    obj=instance,
                )
        return ndarray


FunctionInfo = collections.namedtuple("FunctionInfo", ["function", "size"])


class FunctionParam(Parameter):
    """A parameter where the value is a function."""

    def determine_size(self, instance, function):
        args = self.function_args(instance, function)
        value, invoked = checked_call(function, *args)
        if not invoked:
            raise ValidationError(
                "function '%s' must accept a single " "np.array argument" % function,
                attr=self.name,
                obj=instance,
            )
        return np.asarray(value).size

    def function_args(self, instance, function):
        return (np.zeros(1),)

    def coerce(self, instance, function):
        function = super().coerce(instance, function)
        if isinstance(function, FunctionInfo):
            function_info = function
            function = function_info.function
        else:
            size = (
                self.determine_size(instance, function) if callable(function) else None
            )
            function_info = FunctionInfo(function=function, size=size)

        if function is not None and not callable(function):
            raise ValidationError(
                "function '%s' must be callable" % function,
                attr=self.name,
                obj=instance,
            )

        return function_info


class FrozenObject:
    """An object with parameters that cannot change value after instantiation.

    Since such objects are read-only ("frozen"), they can be safely used in
    multiple locations, compared, etc.
    """

    # Order in which parameters have to be initialized.
    # Missing parameters will be initialized last in an undefined order.
    # This is needed for pickling and copying of Nengo objects when the
    # parameter initialization order matters.
    _param_init_order = []

    def __init__(self):
        self._paramdict = collections.OrderedDict(
            (k, v)
            for k, v in inspect.getmembers(type(self))
            if isinstance(v, Parameter) and not isinstance(v, ObsoleteParam)
        )
        for p in self._params:
            if not p.readonly:
                msg = "All parameters of a FrozenObject must be readonly"
                raise ReadonlyError(attr=p, obj=self, msg=msg)
        self.__argreprs = None

    @property
    def _params(self):
        return list(self._paramdict.values())

    def __eq__(self, other):
        if self is other:  # quick check for speed
            return True
        return type(self) == type(other) and all(
            p.equal(self, other) for p in self._params
        )

    def __hash__(self):
        return hash((type(self), tuple(p.hashvalue(self) for p in self._params)))

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_paramdict")  # do not pickle the param dict itself
        for k in self._paramdict:
            d[k] = getattr(self, k)

        return d

    def __setstate__(self, state):
        FrozenObject.__init__(self)  # set up the param dict

        for attr in self._param_init_order:
            setattr(self, attr, state.pop(attr))
        for attr in set(self._paramdict).difference(self._param_init_order):
            setattr(self, attr, state.pop(attr))

        self.__dict__.update(state)

    def __repr__(self):
        if isinstance(self._argreprs, str):
            return "<%s at 0x%x>" % (type(self).__name__, id(self))
        return "%s(%s)" % (type(self).__name__, ", ".join(self._argreprs))

    @property
    def _argreprs(self):
        if self.__argreprs is not None:
            return self.__argreprs

        # get arguments to display from __init__ functions
        spec = inspect.getfullargspec(type(self).__init__)
        defaults = {}
        if spec.defaults is not None:
            defaults.update(zip(spec.args[-len(spec.defaults) :], spec.defaults))

        self.__argreprs = []
        for arg in spec.args[1:]:  # start at 1 to drop `self`
            if not hasattr(self, arg):
                # We rely on storing the initial arguments. If we don't have
                # them, we don't auto-generate a repr.
                self.__argreprs = "Cannot find %r" % arg
                break
            value = getattr(self, arg)

            param = self._paramdict.get(arg, None)
            if arg in defaults:
                not_default = not equal(value, defaults[arg])
            elif param is not None and param.default is not Unconfigurable:
                not_default = not equal(value, param.default)
            else:
                not_default = True

            if not_default:
                self.__argreprs.append("%s=%r" % (arg, value))

        return self.__argreprs
