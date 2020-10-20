import inspect


class NengoException(Exception):
    """Base class for Nengo exceptions.

    NengoException instances should not be created; this base class exists so
    that all exceptions raised by Nengo can be caught in a try / except block.
    """


class NengoWarning(Warning):
    """Base class for Nengo warnings."""


class ValidationError(NengoException, ValueError):
    """A ValueError encountered during validation of a parameter."""

    def __init__(self, msg, attr, obj=None):
        self.attr = attr
        self.obj = obj
        super().__init__(msg)

    def __str__(self):
        if self.obj is None:
            return f"{self.attr}: {super().__str__()}"
        klassname = (
            self.obj.__name__ if inspect.isclass(self.obj) else type(self.obj).__name__
        )
        return f"{klassname}.{self.attr}: {super().__str__()}"


class ConvergenceError(NengoException, RuntimeError):
    """A RuntimeError raised when an algorithm does not converge."""


class ReadonlyError(ValidationError):
    """A ValidationError occurring because a parameter is read-only."""

    def __init__(self, attr, obj=None, msg=None):
        if msg is None:
            msg = f"{attr} is read-only and cannot be changed"
        super().__init__(msg, attr, obj)


class BuildError(NengoException, ValueError):
    """A ValueError encountered during the build process."""


class ObsoleteError(NengoException):
    """A feature that has been removed in a backwards-incompatible way."""

    def __init__(self, msg, since=None, url=None):
        self.since = since
        self.url = url
        super().__init__(msg)

    def __str__(self):
        since_txt = "" if self.since is None else f" since {self.since}"
        url_txt = (
            f"\nFor more information, please visit {self.url}"
            if self.url is not None
            else ""
        )
        return f"Obsolete{since_txt}: {super().__str__()}{url_txt}"


class MovedError(NengoException):
    """A feature that has been moved elsewhere.

    .. versionadded:: 3.0.0
    """

    def __init__(self, location=None):
        self.location = location
        super().__init__()

    def __str__(self):
        return f"This feature has been moved to {self.location}"


class ConfigError(NengoException, ValueError):
    """A ValueError encountered in the config system."""


class SpaModuleError(NengoException, ValueError):
    """An error in how SPA keeps track of modules."""


class SpaParseError(NengoException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SimulatorClosed(NengoException):
    """Raised when attempting to run a closed simulator."""


class SimulationError(NengoException, RuntimeError):
    """An error encountered during simulation of the model."""


class SignalError(NengoException, ValueError):
    """An error dealing with Signals in the builder."""


class FingerprintError(NengoException, ValueError):
    """An error in fingerprinting an object for cache identification."""


class NetworkContextError(NengoException, RuntimeError):
    """An error with the Network context stack."""


class Unconvertible(NengoException, ValueError):
    """Raised a requested network conversion cannot be done."""


class CacheIOError(NengoException, IOError):
    """An IO error in reading from or writing to the decoder cache."""


class TimeoutError(NengoException):
    """A timeout occurred while waiting for a resource."""


class NotAddedToNetworkWarning(NengoWarning):
    """A NengoObject has not been added to a network."""

    def __init__(self, obj):
        self.obj = obj
        super().__init__()

    def __str__(self):
        return (
            f"{self.obj} was not added to the network. When copying objects, "
            "use the copy method on the object instead of Python's copy "
            "module. When unpickling objects, they have to be added to "
            "networks manually."
        )


class CacheIOWarning(NengoWarning):
    """A non-critical issue in accessing files in the cache."""
