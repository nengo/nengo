"""A customizable configuration system for setting default parameters and
backend-specific info.

The idea here is that a backend can create a Config and ConfigItems to
define the set of parameters that their backend supports.
Parameters are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at
https://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb

"""

import inspect
import sys

from nengo.exceptions import ConfigError, ValidationError
from nengo.params import Default, is_param, iter_params
from nengo.rc import rc
from nengo.utils.threading import ThreadLocalStack


class KeyAccessor:
    def __init__(self, config, key):
        self._config = config
        self._key = key

        self._values = {}
        self._params = {}

        self._is_class = inspect.isclass(key)
        self._key_class = key if inspect.isclass(key) else type(key)

    @property
    def _key_name(self):
        return self._key.__name__ if self._is_class else str(self._key)

    def __str__(self):
        return "%s(%s)" % (self._key_name,
                           ', '.join("%s=%s" % (k, v)
                                     for k, v in self._values.items()))

    def __contains__(self, item):
        return item in self._values or item in self._params

    def __getattr__(self, item):
        assert not item.startswith('_')

        # If we have the exact thing, we'll just return it
        if item in self._values:
            return self._values[item]
        elif item in self._params:
            return self._params[item].default

        # If key is a class return a superclass's ClassParams
        for cls in self._key_class.__mro__:
            if item in self._config[cls]:
                return getattr(self._config[cls], item)

        if self._is_class:
            return getattr(self._key, item).default
        else:
            raise ConfigError("No entry for this instance")

    def __delattr__(self, item):
        if item not in self:
            raise ConfigError("No such attribute")

        if item in self._values:
            del self._values[item]

        # if inspect.isclass(key):
        #     for cls in key.__mro__:
        #         if item in self._values.get(cls, {}):
        #             del self._values[cls][item]

    def __setattr__(self, item, value):
        if item.startswith('_'):
            return super().__setattr__(item, value)

        for cls in self._key_class.__mro__:
            param = self._config[cls]._params.get(item, None)
            if param is not None:
                self._values[item] = param.coerce(self._key, value)
                return

        if item not in dir(self._key):
            raise ConfigError("'%s' is not a parameter in %s."
                              % (item, self._key_name))

        if not self._is_class:
            raise ConfigError(
                "Cannot configure the parameter '%s' on an instance "
                "of '%s'. Please set the attribute directly on the object."
                % (item, type(self._key).__name__))
        else:
            self._values[item] = value

    def _empty(self):
        return (len(self._values) + len(self._params)) == 0

    def _has(self, item):
        if item in self:
            return True

        # If key is a class return a superclass's ClassParams
        if self._is_class:
            for cls in self._key.__mro__:
                if item in self._config[cls]:
                    return True

        return False

    def set_param(self, item, value):
        if not self._is_class:
            raise ConfigError(
                "Cannot set a parameter on an instance of '%s'. Please"
                " set the attribute directly on the object, or"
                " configure the parameter on the class."
                % (type(self._key).__name__,))
        elif not is_param(value):
            raise ConfigError("'%s' is not a parameter" % item)
        elif item in dir(self._key):
            raise ConfigError("'%s' is already a parameter in %s. "
                              "Please choose a different name."
                              % (item, self._key_name))
        self._params[item] = value


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

    To configure defaults on a network::

        net = nengo.Network()
        net.config[nengo.Ensemble].radius = 1.5
        with net:
            ens = nengo.Ensemble(10, 1)
        ens.radius == 1.5  # True

    To add a new parameter to a Nengo object::

        net.config[nengo.Ensemble].set_param(
            'location', nengo.params.Parameter('location'))
        net.config[ens].location = 'cortex'

    To group together a set of parameters::

        gaba = nengo.Config(nengo.Connection)
        gaba[nengo.Connection].synapse = nengo.Lowpass(0.008)
        with net, gaba:
            conn = nengo.Connection(ens, ens)
        conn.synapse == nengo.Lowpass(0.008)  # True

    To configure a new type of object::

        class SynapseInfo:
            label = nengo.params.StringParam('label')
        gaba.configures(SynapseInfo)
        gaba[SynapseInfo].label = "GABA"  # Set default label

    """

    context = ThreadLocalStack(maxsize=100)  # static stack of Config objects

    def __init__(self):
        self.accessors = {}

    def __enter__(self):
        Config.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Config.context) == 0:
            raise ConfigError("Config.context in bad state; was empty when "
                              "exiting from a 'with' block.")

        config = Config.context.pop()

        if config is not self:
            raise ConfigError("Config.context in bad state; was expecting "
                              "current context to be '%s' but instead got "
                              "'%s'." % (self, config))

    def __contains__(self, key):
        raise TypeError("Cannot check if %r is in a config." % (key,))

    def __getitem__(self, key):
        if key not in self.accessors:
            self.accessors[key] = KeyAccessor(self, key)

        return self.accessors[key]

    def __repr__(self):
        classes = [key.__name__ for key, acc in self.accessors.items()
                   if acc._is_class and not acc._empty()]
        return "<%s(%s)>" % (type(self).__name__, ', '.join(classes))

    def __str__(self):
        return "\n".join(str(acc) for acc in self.accessors.values()
                         if len(acc._values) > 0)

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
                    key for key, acc in config.accessors.items()
                    if inspect.isclass(key) and not acc._empty())
            lines.extend([Config.all_defaults(key) for key in all_configured])
        else:
            assert inspect.isclass(nengo_cls)
            lines.append("Current defaults for %s:" % nengo_cls.__name__)
            for attr in dir(nengo_cls):
                desc = getattr(nengo_cls, attr)
                if is_param(desc) and desc.configurable:
                    val = Config.default(nengo_cls, attr)
                    lines.append("  %s: %s" % (attr, val))

        return "\n".join(lines)

    @staticmethod
    def default(nengo_cls, key):
        """Look up the current default value for a parameter.

        The default is found by going through the config stack, from most
        specific to least specific. The network that an object is in
        is the most specific; the top-level network is the least specific.
        If no default is found there, then the parameter's default value
        is returned.
        """
        # Get the descriptor
        desc = getattr(nengo_cls, key)
        if not desc.configurable:
            raise ConfigError("Unconfigurable parameters have no defaults. "
                              "Please ensure you are not using the 'Default' "
                              "keyword with an unconfigurable parameter.")

        for config in reversed(Config.context):
            if config[nengo_cls]._has(key):
                return getattr(config[nengo_cls], key)

        # Otherwise, return the param default
        return desc.default


class SupportDefaultsMixin:
    """Mixin to support assigning `.Default` to parameters.

    Implements ``__setattr__`` to do so. If the inheriting class overrides
    this method, it has to call the mixin's ``__setattr__``.

    This mixin may simplify the exception depending on the value of the
    ``simplified`` rc option.
    """

    # TODO: test

    def __setattr__(self, name, val):
        if val is Default:
            val = Config.default(type(self), name)

        if rc.getboolean('exceptions', 'simplified'):
            try:
                super().__setattr__(name, val)
            except ValidationError:
                exc_info = sys.exc_info()
                raise exc_info[1].with_traceback(None)
        else:
            super().__setattr__(name, val)
