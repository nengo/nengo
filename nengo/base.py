import sys
import warnings

import numpy as np

from nengo.config import Config
from nengo.exceptions import ValidationError
from nengo.params import (
    Default, is_param, IntParam, Parameter, StringParam, Unconfigurable)
from nengo.rc import rc
from nengo.utils.compat import is_integer, reraise, with_metaclass


class NetworkMember(type):
    """A metaclass used to add instances of derived classes to networks.

    Inheriting from this class means that Network.add will be invoked after
    initializing the object, unless add_to_container=False is passed to the
    derived class constructor.
    """

    def __call__(cls, *args, **kwargs):
        """Override default __call__ behavior so that Network.add is called."""
        from nengo.network import Network
        inst = cls.__new__(cls)
        add_to_container = kwargs.pop('add_to_container', True)
        # Do the __init__ before adding in case __init__ errors out
        inst.__init__(*args, **kwargs)
        if add_to_container:
            Network.add(inst)
        inst._initialized = True  # value doesn't matter, just existance
        return inst


class NengoObject(with_metaclass(NetworkMember)):
    """A base class for Nengo objects.

    This defines some functions that the Network requires
    for correct operation. In particular, list membership
    and object comparison require each object to have a unique ID.
    """

    label = StringParam('label', default=None, optional=True)
    seed = IntParam('seed', default=None, optional=True)

    def __init__(self, label, seed):
        self.label = label
        self.seed = seed

    def _str(self, include_id):
        return "<%s%s%s>" % (
            self.__class__.__name__,
            "" if not hasattr(self, 'label') else
            " (unlabeled)" if self.label is None else
            ' "%s"' % self.label,
            " at 0x%x" % id(self) if include_id else "")

    def __str__(self):
        return self._str(
            include_id=not hasattr(self, 'label') or self.label is None)

    def __repr__(self):
        return self._str(include_id=True)

    def __setattr__(self, name, val):
        if hasattr(self, '_initialized') and not hasattr(self, name):
            warnings.warn(
                "Creating new attribute '%s' on '%s'. "
                "Did you mean to change an existing attribute?" % (name, self),
                SyntaxWarning)
        if val is Default:
            val = Config.default(type(self), name)

        if rc.getboolean('exceptions', 'simplified'):
            try:
                super(NengoObject, self).__setattr__(name, val)
            except ValidationError:
                exc_info = sys.exc_info()
                reraise(exc_info[0], exc_info[1], None)
        else:
            super(NengoObject, self).__setattr__(name, val)

    def __getstate__(self):
        raise NotImplementedError("Nengo objects do not support pickling")

    def __setstate__(self, state):
        raise NotImplementedError("Nengo objects do not support pickling")

    @classmethod
    def param_list(cls):
        """Returns a list of parameter names that can be set."""
        return (attr for attr in dir(cls) if is_param(getattr(cls, attr)))

    @property
    def params(self):
        """Returns a list of parameter names that can be set."""
        return self.param_list()


class ObjView(object):
    """Container for a slice with respect to some object.

    This is used by the __getitem__ of Neurons, Node, and Ensemble, in order
    to pass slices of those objects to Connection. This is a notational
    convenience for creating transforms. See Connection for details.

    Does not currently support any other view-like operations.
    """

    def __init__(self, obj, key=slice(None)):
        self.obj = obj
        if is_integer(key):
            # single slices of the form [i] should be cast into
            # slice objects for convenience
            if key == -1:
                # special case because slice(-1, 0) gives the empty list
                key = slice(key, None)
            else:
                key = slice(key, key+1)
        self.slice = key

        # Node.size_in != size_out, so one of these can be invalid
        try:
            self.size_in = np.arange(self.obj.size_in)[self.slice].size
        except IndexError:
            self.size_in = None
        try:
            self.size_out = np.arange(self.obj.size_out)[self.slice].size
        except IndexError:
            self.size_out = None
        if self.size_in is None and self.size_out is None:
            raise ValidationError("Invalid slice '%s' of %s"
                                  % (self.slice, self.obj), attr='key')

    def __getstate__(self):
        raise NotImplementedError("Nengo objects do not support pickling")

    def __setstate__(self, state):
        raise NotImplementedError("Nengo objects do not support pickling")

    def __len__(self):
        return self.size_out

    @property
    def _slice_string(self):
        if isinstance(self.slice, slice):
            sl_start = "" if self.slice.start is None else self.slice.start
            sl_stop = "" if self.slice.stop is None else self.slice.stop
            if self.slice.step is None:
                return "%s:%s" % (sl_start, sl_stop)
            else:
                return "%s:%s:%s" % (sl_start, sl_stop, self.slice.step)
        else:
            return str(self.slice)

    def __str__(self):
        return "%s[%s]" % (self.obj, self._slice_string)

    def __repr__(self):
        return "%r[%s]" % (self.obj, self._slice_string)


class NengoObjectParam(Parameter):
    def __init__(self, name, optional=False, readonly=True,
                 nonzero_size_in=False, nonzero_size_out=False):
        default = Unconfigurable  # These can't have defaults
        self.nonzero_size_in = nonzero_size_in
        self.nonzero_size_out = nonzero_size_out
        super(NengoObjectParam, self).__init__(
            name, default, optional, readonly)

    def validate(self, instance, nengo_obj):
        from nengo.ensemble import Neurons
        from nengo.connection import LearningRule
        if not isinstance(nengo_obj, (
                NengoObject, ObjView, Neurons, LearningRule)):
            raise ValidationError("'%s' is not a Nengo object" % nengo_obj,
                                  attr=self.name, obj=instance)
        if self.nonzero_size_in and nengo_obj.size_in < 1:
            raise ValidationError("'%s' must have size_in > 0." % nengo_obj,
                                  attr=self.name, obj=instance)
        if self.nonzero_size_out and nengo_obj.size_out < 1:
            raise ValidationError("'%s' must have size_out > 0." % nengo_obj,
                                  attr=self.name, obj=instance)
        super(NengoObjectParam, self).validate(instance, nengo_obj)
