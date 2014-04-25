"""A customizable configuration system for setting default parameters and
backend-specific info.

The idea here is that a backend can subclass Config and ConfigItem to
define the set of parameters that their backend supports. Parameters
are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at

  http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""

import collections
import copy
import inspect
import weakref


class DefaultType:
    pass
Default = DefaultType()


class Parameter(object):
    """Simple descriptor for storing configuration parameters"""
    def __init__(self, default):
        self.default = default
        # use a WeakKey dictionary so items can still be garbage collected
        self.data = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = value


def configures(nengo_class):
    """Decorator to mark ConfigItems as to what class they configure"""
    def set_config_nengo_class(klass):
        klass._nengo_class = nengo_class
        return klass
    return set_config_nengo_class


class ConfigItem(object):
    """Base class for defining sets of parameters to configure

    Example
    -------

    @nengo.config.configures(nengo.Connection)
    class TestConfigConnection(nengo.config.ConfigItem):
        my_param = nengo.config.Parameter(None)

    """
    def __setattr__(self, key, value):
        if key not in dir(self):
            raise AttributeError('Unknown config parameter "%s"' % key)
        super(ConfigItem, self).__setattr__(key, value)

    def __getattribute__(self, key):
        val = super(ConfigItem, self).__getattribute__(key)
        if val is Default:
            raise AttributeError("No value for config parameter '%s'" % key)
        return val

    def __str__(self):
        rep = "Defaults for %s:\n" % self._nengo_class
        for d in dir(self):
            if not d.startswith("_"):
                rep += "  %s: %s\n" % (d, getattr(self, d))
        return rep


class Config(object):
    """Network-level configuration options.

    TODO: this is going to be user facing, so add in some explanation of
    how to interact with and modify network.config.

    Example
    -------

    @nengo.config.configures(nengo.Connection)
    class TestConfigConnection(nengo.config.ConfigItem):
        my_param = nengo.config.Parameter(None)

    my_config = Config([TestConfigConnection])
    """

    context = collections.deque(maxlen=100)  # static stack of Config objects

    def __init__(self, config_items=[]):
        self.items = {}
        self.configurable = {}

        for config_item in config_items:
            self.add_config(config_item)

    def __copy__(self):
        new = type(self)()
        new.configurable = dict(self.configurable)
        for key, value in self.items.items():
            new.items[key] = copy.copy(value)  # shallow-copy each ConfigItem

        return new

    def __getitem__(self, key):
        item = self.items.get(key, None)
        if item is None:
            mro = [key] if isinstance(key, type) else []
            mro.extend(list(key.__class__.__mro__))
            for cls in mro:
                if cls in self.configurable:
                    # Get the ConfigItem template from self.configurable
                    item = self.configurable[cls]
                    item = item() if inspect.isclass(item) else copy.copy(item)
                    self.items[key] = item
                    break
            else:
                raise KeyError('No parameters for %s objects' %
                               key.__class__.__name__)
        return item

    def add_config(self, config_item):
        if not hasattr(config_item, "_nengo_class"):
            raise AttributeError(
                "%s is not a decorated ConfigItem" % config_item)

        self.configurable[config_item._nengo_class] = config_item

    @classmethod
    def lookup(cls, key, conf_type):
        for conf in reversed([c for c in cls.context]):
            try:
                val = getattr(conf[conf_type], key)
                return val
            except (KeyError, AttributeError):
                # either there is no config for that object type,
                # or the config item has no entry for that attribute
                pass

        raise AttributeError("No config value found for %s" % key)

    def __enter__(self):
        Config.context.append(self)

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Config.context) == 0:
            raise RuntimeError("Config.context in bad state; was empty when "
                               "exiting from a 'with' block.")

        config = Config.context.pop()

        if len(Config.context) == 0:
            raise RuntimeError("Config.context in bad state; no default "
                               "context is currently set.")

        if config is not self:
            raise RuntimeError("Config.context in bad state; was expecting "
                               "current context to be '%s' but instead got "
                               "'%s'." % (self, config))
