"""A customizable configuration system for setting backend-specific info.

The idea here is that a backend can subclass Config and ConfigItem to
define the set of parameters that their backend supports. Parameters
are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at

  http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""


class Parameter(object):
    def __init__(self, default):
        self.default = default
        self.data = {}

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = value


class ConfigItem(object):
    def __setattr__(self, key, value):
        if key not in dir(self):
            raise AttributeError('Unknown config parameter "%s"' % key)
        super(ConfigItem, self).__setattr__(key, value)


class Config(object):
    def __init__(self, item_classes):
        self.items = {}
        self.item_classes = item_classes

    def __getitem__(self, key):
        item = self.items.get(key, None)
        if item is None:
            # TODO: what if these classes are subclasses of each other?
            for (klass, item_class) in self.item_classes.items():
                if isinstance(key, klass):
                    item = item_class()
                    self.items[key] = item
                    break
            else:
                raise KeyError('No parameters for %s objects' %
                               key.__class__.__name__)
        return item
