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

    def __get__(self, instance, owner):
        """Return the value for the instance, or the Parameter itself.

        If the instance is not specified (i.e., we're accessing
        it as a class attribute) then we will return the Parameter
        in order to call other methods.

        If the instance is specified (i.e., we're accessing it
        normally) we'll return the value, or the default if one
        hasn't been set yet.
        """
        return self

    def __set__(self, instance, value):
        """Set the value for the parameter instance.

        There's a bit of asymmetry here. We disallow changing
        the type of parameter by accessing as a class
        attribute (i.e., if instance is None) like in __get__.
        Instead, setting the value in this way changes the default value.

        If called with an instance, then it sets an instance
        specific value, as you would expect.
        """
        instance.__dict__['_' + self.name] = value

    def __delete__(self, instance):
        raise TypeError("Cannot delete %s: Cannot delete parameters." % (
            self._attrib_name))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name + " parameter. Current value in _" + self.name + "."


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
        params = []
        for k in dir(cls):
            v = getattr(cls, k)
            if isinstance(v, Parameter):
                params.append(k)
                v.name = k
        inst = object.__new__(cls)
        for param in params:
            inst.__dict__["_" + param] = getattr(inst, param).default
        return inst
