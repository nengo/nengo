"""A customizable configuration system for setting default parameters and
backend-specific info.

The idea here is that a backend can create a Config and ConfigItems to
define the set of parameters that their backend supports.
Parameters are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at
http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""

import collections
import inspect
import weakref


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
    modifies : list of str, optional
        A list of attributes that this parameter modifies when being set.
    readonly : bool, optional
        Whether the parameter can be set multiple times.
        By default, parameters can be set multiple times.
    """
    def __init__(self, default, optional=False, modifies=None, readonly=False):
        self.default = default
        self.optional = optional
        self.modifies = modifies
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
        assert value is not Default
        self.validate_none(instance, value)
        if value is not None:
            self.validate(instance, value)
        self.validate_readonly(instance, value)
        self.data[instance] = value

    def __repr__(self):
        return "%s(default=%s)" % (self.__class__.__name__, self.default)

    def validate(self, instance, value):
        pass

    def validate_none(self, instance, value):
        if not self.optional and value is None:
            raise ValueError("Parameter is not optional; cannot set to None")

    def validate_readonly(self, instance, value):
        if self.readonly and instance in self.data:
            raise ValueError("Parameter is read-only; cannot be changed.")


class ClassParams(object):
    """A class to store extra parameters and defaults on Nengo classes.

    This is used by the ``Config`` object to add new ``Parameter``s
    to existing Nengo objects. It should not be instantiated manually.
    """

    def __init__(self, configures):
        self._extraparams = {}
        self._configures = configures
        assert inspect.isclass(configures)

    def __getattribute__(self, key):
        """Overridden to handle instance descriptors manually."""
        try:
            # Parameters are never stored in __dict__
            return super(ClassParams, self).__getattribute__(key)
        except AttributeError:
            # get_param gives a good error message, so this is sufficient
            param = self.get_param(key)
            return param.defaults[self] if self in param else param.default

    def __setattr__(self, key, value):
        """Overridden to handle instance descriptors manually.

        Everything not starting with _ is assumed to be a parameter.
        """
        if key.startswith("_"):
            super(ClassParams, self).__setattr__(key, value)
        else:
            self.get_param(key).defaults[self] = value

    def __str__(self):
        lines = ["All parameters for %s:" % self._configures.__name__]

        # Only print defaults if we've configured them
        for attr in self.default_params:
            if self in self.get_param(attr):
                lines.append("  %s: %s" % (attr, getattr(self, attr)))

        # Print all extra params
        for attr in self.extra_params:
            lines.append("  %s: %s" % (attr, getattr(self, attr)))

        return "\n".join(lines)

    def get_param(self, key):
        if key in self._extraparams:
            return self._extraparams[key]
        elif key in dir(self._configures):
            return getattr(self._configures, key)
        raise AttributeError("Unknown config parameter '%s'" % key)

    def set_param(self, key, value):
        if not is_param(value):
            raise TypeError("%s is not a parameter" % key)
        elif key in dir(self._configures):
            raise ValueError("%s is already a parameter in %s. "
                             "Please choose a different name."
                             % (key, self._configures.__name__))
        self._extraparams[key] = value

    def update(self, d):
        """Sets a number of parameters at once given a dictionary."""
        for key in d:
            setattr(self, key, d[key])

    @property
    def default_params(self):
        return (attr for attr in dir(self._configures)
                if is_param(getattr(self._configures, attr)))

    @property
    def extra_params(self):
        return self._extraparams.keys()

    @property
    def params(self):
        return list(self.default_params) + list(self.extra_params)


class InstanceParams(object):
    """A class to store extra parameters on Nengo objects.

    This restricts the amount of configurability for instances.
    All you can do is get and set parameter values; getting and setting
    the parameters themselves can only be done on the class.
    """

    def __init__(self, configures, clsparams):
        self._configures = configures
        self._clsparams = clsparams
        assert not inspect.isclass(configures)

    def __getattribute__(self, key):
        try:
            return super(InstanceParams, self).__getattribute__(key)
        except AttributeError:
            if key in self._clsparams.default_params:
                raise
            param = self._clsparams.get_param(key)
            return param.__get__(self, self.__class__)

    def __setattr__(self, key, value):
        """Everything not starting with _ is assumed to be a parameter."""
        if key.startswith("_"):
            super(InstanceParams, self).__setattr__(key, value)
        elif key in dir(self._configures):
            # Disallow configuring attributes the instance already has
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (self.__class__.__name__, key))
        else:
            self._clsparams.get_param(key).__set__(self, value)

    def __str__(self):
        lines = ["Parameters set for %s:" % str(self._configures)]
        for attr in self._clsparams.params:
            if self in self._clsparams.get_param(attr):
                lines.append("  %s: %s" % (attr, getattr(self, attr)))
        return "\n".join(lines)

    def get_param(self, key):
        raise ValueError("Cannot get parameters on an instance; use "
                         "'config[%s].get_param' instead."
                         % self._configures.__class__.__name__)

    def set_param(self, key, value):
        raise ValueError("Cannot set parameters on an instance; use "
                         "'config[%s].set_param' instead."
                         % self._configures.__class__.__name__)


class Config(object):
    """Configures network-level behavior and backend specific parameters.

    Every ``Network`` contains an associated ``Config`` object which can
    be manipulated to change overall network behavior, and to store
    backend specific parameters. Subnetworks inherit the ``Config`` of
    their parent, but can be manipulated independently.
    The top-level network inherits ``nengo.toplevel_config``.

    Attributes
    ----------
    params : dict
        Maps configured classes and instances to their ``ClassParams``
        or ``InstanceParams`` object.

    Example
    -------
    >>> class A(object): pass
    >>> inst = A()
    >>> config = Config()
    >>> config.configures(A)
    >>> config[A].set_param('amount', Parameter(default=1))
    >>> print(config[inst].amount)
    1
    >>> config[inst].amount = 3
    >>> print(config[inst].amount)
    3
    >>> print(config[A].amount)
    1
    """

    context = collections.deque(maxlen=100)  # static stack of Config objects

    def __init__(self):
        self.params = {}

    @classmethod
    def default(cls, nengo_cls, param):
        """Look up the current default value for a parameter.

        The default is found by going through the config stack, top to bottom.
        If no default is found there, then the parameter's default value
        is returned.
        """

        # Get the descriptor
        desc = getattr(nengo_cls, param)
        for config in reversed(cls.context):

            # If a default has been set for this config, return it
            if config[nengo_cls] in desc:
                return getattr(config[nengo_cls], param)

        # Otherwise, return the param default
        return desc.default

    def __enter__(self):
        Config.context.append(self)

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Config.context) == 0:
            raise RuntimeError("Config.context in bad state; was empty when "
                               "exiting from a 'with' block.")

        config = Config.context.pop()

        if config is not self:
            raise RuntimeError("Config.context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self, config))

    def __getitem__(self, key):
        # If we have the exact thing, we'll just return it
        if key in self.params:
            return self.params[key]

        # If key is a class return a superclass's ClassParams
        if inspect.isclass(key):
            for cls in key.__mro__:
                if cls in self.params:
                    return self.params[cls]

            # If no superclass ClassParams, KeyError
            raise KeyError(
                "Type '%(name)s' is not set up for configuration. "
                "Call 'configure(%(name)s)' first." % {'name': key.__name__})

        # For new instances, if we configure a class in the mro we're good
        for cls in key.__class__.__mro__:
            if cls in self.params:
                clsparams = self.params[cls]
                instparams = InstanceParams(key, clsparams)
                self.params[key] = instparams
                return instparams

        # If we don't configure the class, KeyError
        raise KeyError(
            "Type '%(name)s' is not set up for configuration. Call "
            "configure('%(name)s') first." % {'name': key.__class__.__name__})

    def configures(self, cls):
        """Start configuring a particular class and its instances."""
        self.params[cls] = ClassParams(cls)
