"""A customizable configuration system for setting backend-specific info.

The idea here is that a backend can subclass Config and ConfigItem to
define the set of parameters that their backend supports. Parameters
are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at

  http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""

import weakref
import inspect


class Parameter(object):
    """Simple decorator for storing configuration parameters"""
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
        klass.nengo_class = nengo_class
        return klass
    return set_config_nengo_class


class ConfigItem(object):
    """Base class for defining sets of parameters to configure"""
    def __setattr__(self, key, value):
        if key not in dir(self):
            raise AttributeError('Unknown config parameter "%s"' % key)
        super(ConfigItem, self).__setattr__(key, value)


class Config(object):
    """Base class for backends to define their own Config.

    Subclasses are expected to set a class variable config_items
    to be a list of ConfigItem subclasses, each decorated with a
    @configures to indicate what nengo class these parameters are for.
    """

    def __init__(self):
        self.items = {}
        if not hasattr(self, 'config_items'):
            raise AttributeError('Config must have config_items ' +
                                 'set to be a list of ConfigItems')
        for config_item in self.config_items:
            if not (inspect.isclass(config_item) and
                    issubclass(config_item, ConfigItem) and
                    hasattr(config_item, 'nengo_class')):
                raise AttributeError('config_items must be a list of ' +
                                     'ConfigItem subclasses, each with ' +
                                     '@configures decorators ' +
                                     'indicating what class it configures')

    def __getitem__(self, key):
        item = self.items.get(key, None)
        if item is None:
            for config_item in self.config_items:
                if isinstance(key, config_item.nengo_class):
                    item = config_item()
                    self.items[key] = item
                    break
            else:
                raise KeyError('No parameters for %s objects' %
                               key.__class__.__name__)
        return item
