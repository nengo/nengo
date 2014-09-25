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

from nengo.params import is_param


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

    def __delattr__(self, key):
        if key.startswith("_"):
            super(ClassParams, self).__delattr__(key)
        else:
            del self.get_param(key).defaults[self]

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
        if key in self.extra_params:
            return self._extraparams[key]
        elif key in self.default_params:
            return getattr(self._configures, key)
        else:
            raise AttributeError("Unknown config parameter '%s'" % key)

    def set_param(self, key, value):
        if key in dir(self):
            raise ValueError("'%s' is a reserved key. "
                             "Please choose a different name." % key)
        elif key in self.default_params:
            raise ValueError("'%s' is already a parameter in %s. "
                             "Please choose a different name."
                             % (key, self._configures.__name__))
        elif not is_param(value):
            raise TypeError("%s is not a parameter" % key)
        else:
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
        return list(self._extraparams)

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

    def __delattr__(self, key):
        if key.startswith("_"):
            super(InstanceParams, self).__delattr__(key)
        elif key in dir(self._configures):
            # Disallow configuring attributes the instance already has
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (self.__class__.__name__, key))
        else:
            self._clsparams.get_param(key).__delete__(self)

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
