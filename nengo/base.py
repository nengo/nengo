import warnings

import numpy as np

from nengo.config import Config
from nengo.params import Default, is_param, IntParam, Parameter, SliceParam
from nengo.utils.compat import with_metaclass


class NengoObject(object):
    """A base class for Nengo objects.

    This provides some common functionality for NengoObjects,
    the most important of which is an overridden __setattr__, which hooks
    NengoObjects up with the Config system, and warns whenever an attribute
    is added to the object, as this is commonly due to typos.
    """

    def __len__(self):
        return self.size_out

    def __repr__(self):
        if self.label is not None:
            return "%s(label='%s')" % (self.__class__.__name__, self.label)
        return self.__class__.__name__

    def __setattr__(self, name, val):
        # Parameters are already defined, so only new attrs will warn
        if not hasattr(self, name):
            warnings.warn(
                "Creating new attribute '%s' on '%s'. "
                "Did you mean to change an existing attribute?" % (name, self),
                SyntaxWarning)
        if val is Default:
            val = Config.default(type(self), name)
        try:
            super(NengoObject, self).__setattr__(name, val)
        except Exception as e:
            arg0 = '' if len(e.args) == 0 else e.args[0]
            arg0 = ("Validation error when setting '%s.%s': %s"
                    % (self.__class__.__name__, name, arg0))
            e.args = (arg0,) + e.args[1:]
            raise

    @classmethod
    def param_list(cls):
        """Returns a list of parameter names that can be set."""
        return (attr for attr in dir(cls) if is_param(getattr(cls, attr)))

    @property
    def params(self):
        """Returns a list of parameter names that can be set."""
        return self.param_list()


class NetworkMemberMeta(type):
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
        return inst

NetworkMember = with_metaclass(NetworkMemberMeta, NengoObject)


class NengoObjectParam(Parameter):
    def __init__(self, default=None, disallow=None, optional=False,
                 readonly=True):
        assert default is None  # These can't have defaults
        self.disallow = [] if disallow is None else disallow
        super(NengoObjectParam, self).__init__(default, optional, readonly)

    def validate(self, instance, nengo_obj):
        if not isinstance(nengo_obj, NengoObject):
            raise ValueError("'%s' is not a Nengo object" % nengo_obj)
        for n_type in self.disallow:
            if isinstance(nengo_obj, n_type):
                raise ValueError("Objects of type '%s' disallowed." % n_type)
        super(NengoObjectParam, self).validate(instance, nengo_obj)


class ObjView(NengoObject):
    """Container for a slice with respect to some object.

    This is used by the __getitem__ of Neurons, Node, and Ensemble, in order
    to pass slices of those objects to Connect. This is a notational
    convenience for creating transforms. See Connect for details.

    Does not currently support any other view-like operations.
    """

    obj = NengoObjectParam()
    key = SliceParam(default=None, readonly=True)
    size_in = IntParam(default=None, optional=True, readonly=True)
    size_out = IntParam(default=None, optional=True, readonly=True)

    def __init__(self, obj, key=slice(None)):
        self.obj = obj
        self.key = key

        # Node.size_in != size_out, so one of these can be invalid
        try:
            self.size_in = np.arange(obj.size_in)[self.key].size
        except IndexError:
            self.size_in = None
        try:
            self.size_out = np.arange(obj.size_out)[self.key].size
        except IndexError:
            self.size_out = None

    @property
    def label(self):
        if isinstance(self.key, list):
            sl_str = self.key
        else:
            sl_start = "" if self.key.start is None else self.key.start
            sl_stop = "" if self.key.stop is None else self.key.stop
            if self.key.step is None:
                sl_str = "%s:%s" % (sl_start, sl_stop)
            else:
                sl_str = "%s:%s:%s" % (sl_start, sl_stop, self.key.step)
        return "%s[%s]" % (self.obj.label, sl_str)
