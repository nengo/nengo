import inspect


class NengoException(Exception):
    """Base class for Nengo exceptions.

    NengoException instances should not be created; this base class exists so
    that all exceptions raised by Nengo can be caught in a try / except block.
    """


class ValidationError(NengoException, ValueError):
    """A ValueError encountered during validation of a parameter."""

    def __init__(self, msg, attr, obj=None):
        self.attr = attr
        self.obj = obj
        super(ValidationError, self).__init__(msg)

    def __str__(self):
        if self.obj is None:
            return "{}: {}".format(
                self.attr, super(ValidationError, self).__str__())
        klassname = (self.obj.__name__ if inspect.isclass(self.obj)
                     else self.obj.__class__.__name__)
        return "{}.{}: {}".format(
            klassname, self.attr, super(ValidationError, self).__str__())


class ReadonlyError(ValidationError):
    """A ValidationError occurring because a parameter is read-only."""

    def __init__(self, attr, obj=None, msg=None):
        if msg is None:
            msg = "%s is read-only and cannot be changed" % attr
        super(ReadonlyError, self).__init__(msg, attr, obj)


class BuildError(NengoException, ValueError):
    """A ValueError encountered during the build process."""
