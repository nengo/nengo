"""
This file contains classes to make parameter handling
and reporting easier in Nengo through the use of
a bit of Python magic (descriptors, etc.)

These classes should not need to be understood
by the end user, but Nengo developers should
have some understanding of how these classes work.

Some inspiration for this file is taken from Param by IOAM:
  http://ioam.github.io/param/

"""

class Parameter(object):
    """A descriptor for keeping track of parameters in Nengo.

    The primary use of these is to make the probe interface simpler.
    However, this is also handy for keeping track of default values,
    and making object implementations simpler.

    """
    def __init__(self, default, doc=None):
        self.default = default
        self.doc = doc
        self.name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._internal_name = "_%s_value" % value

    def __get__(self, instance, owner):
        """Return the value for the instance, or the Parameter itself.

        If the instance is not specified (i.e., we're accessing
        it as a class attribute) then we will return the Parameter
        in order to call other methods.

        If the instance is specified (i.e., we're accessing it
        normally) we'll return the value, or the default if one
        hasn't been set yet.
        """
        if instance is None:
            return self
        return instance.__dict__.get(self._internal_name, self.default)

    def __set__(self, instance, value):
        """Set the value for the parameter instance.

        There's a bit of asymmetry here. We disallow changing
        the type of parameter by accessing as a class
        attribute (i.e., if instance is None) like in __get__.
        Instead, setting the value in this way changes the default value.

        If called with an instance, then it sets an instance
        specific value, as you would expect.
        """
        if instance is None:
            self.default = value
        else:
            instance.__dict__[self._internal_name] = value

    def __delete__(self, instance):
        raise TypeError("Cannot delete %s: Parameter deletion not allowed." % (
            self._attrib_name))


class Parameterized(object):
    """A base class for objects that contain parameters.

    This enables two things in the subclass.
    First, Parameter class attributes are detected
    and their names are set up appropriately.
    Second, __setattr__ is overridden so that
    assignments to these class attributes
    do not replace them, but instead
    replaces their default value.
    """
    def __new__(cls, *args, **kwargs):
        for k in dir(cls):
            v = getattr(cls, k)
            if isinstance(v, Parameter):
                v.name = k
        return object.__new__(cls)

    def __setattr__(self, name, value):
        try:
            attr = getattr(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
        else:
            if isinstance(attr, Parameter):
                attr.__set__(None, value)
            else:
                object.__setattr__(self, name, value)
