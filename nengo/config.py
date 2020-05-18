"""Configuration system to set defaults and backend-specific info.

The idea here is that a backend can create a Config and ConfigItems to
define the set of parameters that their backend supports.
Parameters are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at
https://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb

"""

import inspect

from nengo.exceptions import ConfigError
from nengo.params import is_param, iter_params
from nengo.utils.threading import ThreadLocalStack


class ClassParams:
    """A class to store extra parameters and defaults on configured classes.

    This is used by `.Config` to associate defaults and new `.Parameter`
    instances with existing objects. It should not be instantiated outside of
    `.Config.configures`.

    Parameters
    ----------
    configures : class
        The class with which to associate new defaults and parameters.
    """

    def __init__(self, configures):
        assert inspect.isclass(configures)
        self._configures = configures
        self._extra_params = {}
        self._default_params = tuple(
            attr
            for attr in dir(self._configures)
            if is_param(getattr(self._configures, attr))
        )

    def __contains__(self, key):
        return self in self.get_param(key)

    def __delattr__(self, key):
        if key.startswith("_"):
            super().__delattr__(key)
        else:
            self.get_param(key).del_default(self)

    def __getattr__(self, key):
        if key.startswith("_"):
            # If we get here, then that attribute hasn't been set
            raise AttributeError(
                "%r object has no attribute %r" % (type(self).__name__, key)
            )
        return self.get_param(key).get_default(self)

    def __setattr__(self, key, value):
        """Overridden to handle instance descriptors manually.

        Everything not starting with _ is assumed to be a parameter.
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            param = self.get_param(key)
            if not param.configurable:
                raise ConfigError("Parameter '%s' is not configurable" % key)
            param.set_default(self, value)

    def __getstate__(self):
        state = {
            "_configures": self._configures,
            "_default_params": self._default_params,
            "_extra_params": self._extra_params,
        }

        # Store all of the things we set in the params
        for attr in self.params:
            param = self.get_param(attr)
            if self in param:
                state[attr] = param.get_default(self)

        return state

    def __setstate__(self, state):
        self._configures = state["_configures"]
        self._default_params = state["_default_params"]
        self._extra_params = state["_extra_params"]

        # Restore all of the things we set in the params
        for attr in self.params:
            if attr in state:
                self.get_param(attr).set_default(self, state[attr])

    def __str__(self):
        name = self._configures.__name__
        lines = ["Parameters configured for %s:" % name]
        for attr in self.params:
            if self in self.get_param(attr):
                lines.append("  %s: %s" % (attr, getattr(self, attr)))
        if len(lines) > 1:
            return "\n".join(lines)
        else:
            return "No parameters configured for %s." % name

    def __repr__(self):
        # Only print defaults if we've configured them
        params = []
        filled_defaults = [
            attr for attr in self.default_params if self in self.get_param(attr)
        ]
        for attr in filled_defaults + sorted(self.extra_params):
            params.append("%s: %s" % (attr, getattr(self, attr)))

        return "<%s[%s]{%s}>" % (
            type(self).__name__,
            self._configures.__name__,
            ", ".join(params),
        )

    @property
    def default_params(self):
        return self._default_params

    @property
    def extra_params(self):
        return tuple(self._extra_params)

    @property
    def params(self):
        return self.default_params + self.extra_params

    def get_param(self, key):
        if key in self._extra_params:
            return self._extra_params[key]

        return getattr(self._configures, key)

    def set_param(self, key, value):
        if not is_param(value):
            raise ConfigError("'%s' is not a parameter" % key)
        elif key in dir(self._configures):
            raise ConfigError(
                "'%s' is already a parameter in %s. "
                "Please choose a different name." % (key, self._configures.__name__)
            )
        self._extra_params[key] = value

    def update(self, d):
        """Sets a number of parameters at once given a dictionary."""
        for key in d:
            setattr(self, key, d[key])


class InstanceParams:
    """A class to store parameter value on configured objects.

    In contrast to `.ClassParams`, the only thing that can be done with
    ``InstanceParams`` is get and set parameter values. Use the corresponding
    `.ClassParams` to set defaults and create new parameters.
    """

    def __init__(self, configures, clsparams):
        self._configures = configures
        self._clsparams = clsparams
        assert not inspect.isclass(configures)

    def __contains__(self, key):
        return self in self._clsparams.get_param(key)

    def __delattr__(self, key):
        if key.startswith("_"):
            super().__delattr__(key)
        elif key in dir(self._configures):
            # Disallow configuring attributes the instance already has
            raise ConfigError(
                "Cannot configure the built-in parameter '%s' on an instance "
                "of '%s'. Please delete the attribute directly on the object."
                % (key, type(self._configures).__name__)
            )
        else:
            self._clsparams.get_param(key).__delete__(self)

    def __getattr__(self, key):
        if key in self._clsparams.default_params:
            raise ConfigError(
                "Cannot configure the built-in parameter '%s' on an instance "
                "of '%s'. Please get the attribute directly from the object."
                % (key, type(self._configures).__name__)
            )
        param = self._clsparams.get_param(key)
        if self in param:
            return param.__get__(self, type(self))
        return getattr(self._clsparams, key)

    def __setattr__(self, key, value):
        """Everything not starting with _ is assumed to be a parameter."""
        if key.startswith("_"):
            super().__setattr__(key, value)
        elif key in dir(self._configures):
            # Disallow configuring attributes the instance already has
            raise ConfigError(
                "Cannot configure the built-in parameter '%s' on an instance "
                "of '%s'. Please set the attribute directly on the object."
                % (key, type(self._configures).__name__)
            )
        else:
            self._clsparams.get_param(key).__set__(self, value)

    def __getstate__(self):
        state = {}

        for key in iter_params(self._configures):
            param = self._clsparams.get_param(key)
            if self in param:
                state[key] = param.__get__(self, type(self))

        state.update(self.__dict__)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __repr__(self):
        params = []
        filled_params = [
            attr
            for attr in self._clsparams.params
            if self in self._clsparams.get_param(attr)
        ]
        for attr in filled_params:
            params.append("%s: %s" % (attr, getattr(self, attr)))

        return "<%s[%s]{%s}>" % (
            type(self).__name__,
            self._configures,
            ", ".join(params),
        )

    def __str__(self):
        lines = ["Parameters set for %s:" % str(self._configures)]
        for attr in self._clsparams.params:
            if self in self._clsparams.get_param(attr):
                lines.append("  %s: %s" % (attr, getattr(self, attr)))
        return "\n".join(lines)

    def get_param(self, key):
        raise ConfigError(
            "Cannot get parameters on an instance; use "
            "'config[%s].get_param' instead." % type(self._configures).__name__
        )

    def set_param(self, key, value):
        raise ConfigError(
            "Cannot set parameters on an instance; use "
            "'config[%s].set_param' instead." % type(self._configures).__name__
        )


class Config:
    """Configures network-level defaults and additional parameters.

    Every `.Network` contains an associated ``Config`` object which can
    be manipulated to change network-specific defaults, and to store
    additional parameters (for example, those specific to a backend).

    A ``Config`` object can configure objects of any class, but it has to be
    told the classes to configure first. This is either done on instantiation
    of the ``Config`` object or by calling the `.configures`  method.
    This sets up a mapping between configured class and a `.ClassParams`
    object that sets the default values for that class. Attempting to
    configure an instance of a configure class will create a mapping from
    that instance to an `.InstanceParams` object to configure additional
    parameters for that instance.

    Parameters
    ----------
    *configures
        The classes that this ``Config`` instance will configure.

    Attributes
    ----------
    params : dict
        Maps configured classes and instances to their `.ClassParams`
        or `.InstanceParams` object.

    Examples
    --------
    To configure defaults on a network:

    .. testcode::

       net = nengo.Network()
       net.config[nengo.Ensemble].radius = 1.5
       with net:
           ens = nengo.Ensemble(10, 1)
       ens.radius == 1.5  # True

    To add a new parameter to a Nengo object:

    .. testcode::

       net.config[nengo.Ensemble].set_param(
           'location', nengo.params.Parameter('location')
       )
       net.config[ens].location = 'cortex'

    To group together a set of parameters:

    .. testcode::

       gaba = nengo.Config(nengo.Connection)
       gaba[nengo.Connection].synapse = nengo.Lowpass(0.008)
       with net, gaba:
           conn = nengo.Connection(ens, ens)
       conn.synapse == nengo.Lowpass(0.008)  # True

    To configure a new type of object:

    .. testcode::

       class SynapseInfo:
           label = nengo.params.StringParam('label', default=None)
       gaba.configures(SynapseInfo)
       gaba[SynapseInfo].label = "GABA"  # Set default label

    """

    context = ThreadLocalStack(maxsize=100)  # static stack of Config objects

    def __init__(self, *configures):
        self.params = {}
        if len(configures) > 0:
            self.configures(*configures)

    def __enter__(self):
        Config.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Config.context) == 0:
            raise ConfigError(
                "Config.context in bad state; was empty when "
                "exiting from a 'with' block."
            )

        config = Config.context.pop()

        if config is not self:
            raise ConfigError(
                "Config.context in bad state; was expecting "
                "current context to be '%s' but instead got "
                "'%s'." % (self, config)
            )

    def __contains__(self, key):
        raise TypeError("Cannot check if %r is in a config." % (key,))

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
            raise ConfigError(
                "Type '%(name)s' is not set up for configuration. "
                "Call 'configures(%(name)s)' first." % {"name": key.__name__}
            )

        # For new instances, if we configure a class in the mro we're good
        for cls in type(key).__mro__:
            if cls in self.params:
                clsparams = self.params[cls]
                instparams = InstanceParams(key, clsparams)
                self.params[key] = instparams
                return instparams

        # If we don't configure the class, KeyError
        raise ConfigError(
            "Type '%(name)s' is not set up for configuration. Call "
            "configures('%(name)s') first." % {"name": type(key).__name__}
        )

    def __repr__(self):
        classes = [key.__name__ for key in self.params if inspect.isclass(key)]
        return "<%s(%s)>" % (type(self).__name__, ", ".join(classes))

    def __str__(self):
        return "\n".join(str(v) for v in self.params.values())

    @staticmethod
    def all_defaults(nengo_cls=None):
        """Look up all of the default values in the current context.

        Parameters
        ----------
        nengo_cls : class, optional
            If specified, only the defaults for a particular class will
            be returned. If not specified, the defaults for all configured
            classes will be returned.

        Returns
        -------
        str
        """
        lines = []
        if nengo_cls is None:
            all_configured = set()
            for config in Config.context:
                all_configured.update(
                    key for key in config.params if inspect.isclass(key)
                )
            lines.extend([Config.all_defaults(key) for key in all_configured])
        else:
            lines.append("Current defaults for %s:" % nengo_cls.__name__)
            for attr in dir(nengo_cls):
                desc = getattr(nengo_cls, attr)
                if is_param(desc) and desc.configurable:
                    val = Config.default(nengo_cls, attr)
                    lines.append("  %s: %s" % (attr, val))
        return "\n".join(lines)

    @staticmethod
    def default(nengo_cls, param):
        """Look up the current default value for a parameter.

        The default is found by going through the config stack, from most
        specific to least specific. The network that an object is in
        is the most specific; the top-level network is the least specific.
        If no default is found there, then the parameter's default value
        is returned.
        """

        # Get the descriptor
        desc = getattr(nengo_cls, param)
        if not desc.configurable:
            raise ConfigError(
                "Unconfigurable parameters have no defaults. "
                "Please ensure you are not using the 'Default' "
                "keyword with an unconfigurable parameter."
            )

        for config in reversed(Config.context):
            for cls in nengo_cls.__mro__:
                # If a default has been set for this config, return it
                if cls in config.params and config[cls] in desc:
                    return getattr(config[cls], param)

        # Otherwise, return the param default
        return desc.default

    def configures(self, *classes):
        """Start configuring a particular class and its instances."""
        if len(classes) == 0:
            raise TypeError("configures() takes 1 or more arguments (0 given)")
        for klass in classes:
            self.params[klass] = ClassParams(klass)
