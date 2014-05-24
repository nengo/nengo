"""A customizable configuration system for setting backend-specific info.

The idea here is that a backend can create a Config and ConfigItems to
define the set of parameters that their backend supports.
Parameters are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at
http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""

import copy
import inspect
import weakref


def is_param(obj):
    return hasattr(obj, '__get__') and hasattr(obj, '__set__')


class Parameter(object):
    """Simple descriptor for storing configuration parameters"""
    def __init__(self, default):
        self.default = default
        # use a WeakKey dictionary so items can still be garbage collected
        self.data = weakref.WeakKeyDictionary()

    def __delete__(self, instance):
        del self.data[instance]

    def __get__(self, instance, type_):
        if instance is None:
            # Return self so default can be inspected
            return self
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        if value == 'default':
            # NB: If default is overridden, value will not be updated
            value = self.default
        self.data[instance] = value

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "%s(default=%s)" % (self.__class__.__name__, self.default)

    def configures(self, instance):
        return instance in self.data


class ExtraParams(object):
    """A class to store extra sets of configurable parameters.

    This is used by the ``Config`` object to add new ``Parameter``s
    to existing Nengo objects. It should not be instantiated manually.
    """

    def __init__(self, configures, parent=None):
        """Initialize with the parameters in the object we're configuring.

        Also optionally copy over parameters from another ExtraParams object
        passed as ``parent``.
        """
        self._params = []
        self._configures = configures
        self._parent = parent

        # We have to find params on the configured class manually
        # because it might be a class or an instance
        for attr in dir(configures):
            param = getattr(configures, attr)
            if isinstance(param, Parameter):
                self.set_param(attr, copy.copy(param))

        # For the parent, we can use its params list
        if parent is not None:
            for attr in parent.params:
                param = parent.get_param(attr)
                self.set_param(attr, copy.copy(param))

    def __getattribute__(self, key):
        """Overridden to handle instance descriptors manually."""
        attr = super(ExtraParams, self).__getattribute__(key)
        if is_param(attr):
            # Defer to parent if a param is not explicitly configured
            if not attr.configures(self) and self._parent is not None:
                return getattr(self._parent, key)
            return attr.__get__(self, self.__class__)
        return attr

    def __setattr__(self, key, value):
        """Overridden to handle instance descriptors manually.

        Everything not starting with _ is assumed to be a parameter.
        """
        if key.startswith("_"):
            super(ExtraParams, self).__setattr__(key, value)
        else:
            self.get_param(key).__set__(self, value)

    def __str__(self):
        rep = "All parameters for %s:\n" % str(self._configures)
        for attr in self.params:
            rep += "  %s: %s\n" % (attr, getattr(self, attr))
        return rep

    def get_param(self, key):
        value = super(ExtraParams, self).__getattribute__(key)
        if not is_param(value):
            raise AttributeError("Unknown config parameter '%s'" % key)
        return value

    def set_param(self, key, value):
        if not is_param(value):
            raise TypeError("%s is not a parameter" % key)
        super(ExtraParams, self).__setattr__(key, value)
        self._params.append(key)

    @property
    def params(self):
        return iter(self._params)


class Config(object):
    """Configures network-level behavior and backend specific parameters.

    Every ``Network`` contains an associated ``Config`` object which can
    be manipulated to change overall network behavior, and to store
    backend specific parameters. Subnetworks inherit the ``Config`` of
    their parent, but can be manipulated independently.
    The top-level network inherits ``nengo.toplevel_config``.

    Attributes
    ----------
    extra_params : dict
        Maps configured classes and instances to their ``ExtraParams`` object.

    Example
    -------
    >>> class A(object): pass
    >>> inst = A()
    >>> config = Config()
    >>> config.configures(A)
    >>> config[A].amount = Parameter(default=1)
    >>> print(config[inst].amount)
    1
    >>> config[inst].amount = 3
    >>> print(config[inst].amount)
    3
    >>> print(config[A].amount)
    1
    """

    def __init__(self, parent=None):
        self.extra_params = {}
        if parent is None:
            return

        # If a parent is passed, we'll copy its config
        for key in parent.extra_params:
            # But only the classes...
            # instances should only be configured in their owning network
            if inspect.isclass(key):
                self.extra_params[key] = ExtraParams(
                    key, parent=parent.extra_params[key])

    def __getitem__(self, key):
        # If key is a class and we have an ExtraParams for it, we'll return it
        eps = self.extra_params.get(key, None)
        if eps is not None:
            return eps

        # If item is a class we don't have an ExtraParams for, KeyError
        if inspect.isclass(key):
            raise KeyError(
                "Type '%(name)s' is not set up for configuration. "
                "Call configure('%(name)s') first." % {'name': key.__name__})

        # If item isn't a class, it's an instance
        # If we configure the class, we'll configure the instance
        for cls in key.__class__.__mro__:
            if cls in self.extra_params:
                eps = ExtraParams(cls, parent=self.extra_params[cls])
                self.extra_params[key] = eps
                return eps

        # If we don't configure the class, KeyError
        raise KeyError(
            "Type '%(name)s' is not set up for configuration. Call "
            "configure('%(name)s') first." % {'name': key.__class__.__name__})

    def configure(self, klass):
        """Start configuring a particular class and its instances."""
        self.extra_params[klass] = ExtraParams(klass)
