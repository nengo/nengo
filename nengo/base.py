import sys
import warnings

import numpy as np

from nengo.config import Config
from nengo.exceptions import ValidationError
from nengo.params import (
    Default, FrozenObject, is_param, IntParam, NumberParam, Parameter,
    StringParam, Unconfigurable)
from nengo.rc import rc
from nengo.utils.compat import is_integer, range, reraise, with_metaclass
from nengo.utils.numpy import as_shape, maxint


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

    Parameters
    ----------
    label : string
        A descriptive label for the object.
    seed : int
        The seed used for random number generation.

    Attributes
    ----------
    label : string
        A descriptive label for the object.
    seed : int
        The seed used for random number generation.
    """

    label = StringParam('label', default=None, optional=True)
    seed = IntParam('seed', default=None, optional=True)

    def __init__(self, label, seed):
        self.label = label
        self.seed = seed

    def __getstate__(self):
        raise NotImplementedError("Nengo objects do not support pickling")

    def __setstate__(self, state):
        raise NotImplementedError("Nengo objects do not support pickling")

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

    def __str__(self):
        return self._str(
            include_id=not hasattr(self, 'label') or self.label is None)

    def __repr__(self):
        return self._str(include_id=True)

    def _str(self, include_id):
        return "<%s%s%s>" % (
            self.__class__.__name__,
            "" if not hasattr(self, 'label') else
            " (unlabeled)" if self.label is None else
            ' "%s"' % self.label,
            " at 0x%x" % id(self) if include_id else "")

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

    def __str__(self):
        return "%s[%s]" % (self.obj, self._slice_string)

    def __repr__(self):
        return "%r[%s]" % (self.obj, self._slice_string)

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


class Process(FrozenObject):
    """A general system with input, output, and state.

    Attributes
    ----------
    default_size_in : int (Default: 0)
        Sets the default size in for nodes using this process.
    default_size_out : int (Default: 1)
        Sets the default size out for nodes running this process. Also,
        if ``d`` is not specified in `.run` or `.run_steps`, this will be used.
    default_dt : float (Default: 0.001 (1 millisecond))
        If ``dt`` is not specified in `.run`, `.run_steps`, `.ntrange`,
        or `.trange`, this will be used.
    seed : int, optional (Default: None)
        Random number seed. Ensures random factors will be the same each run.
    """

    default_size_in = IntParam('default_size_in', low=0)
    default_size_out = IntParam('default_size_out', low=0)
    default_dt = NumberParam('default_dt', low=0, low_open=True)
    seed = IntParam('seed', low=0, high=maxint, optional=True)

    def __init__(self, default_size_in=0, default_size_out=1,
                 default_dt=0.001, seed=None):
        super(Process, self).__init__()
        self.default_size_in = default_size_in
        self.default_size_out = default_size_out
        self.default_dt = default_dt
        self.seed = seed

    def apply(self, x, d=None, dt=None, rng=np.random, copy=True, **kwargs):
        """Run process on a given input."""
        shape_in = as_shape(np.asarray(x[0]).shape, min_dim=1)
        shape_out = as_shape(self.default_size_out if d is None else d)
        dt = self.default_dt if dt is None else dt
        rng = self.get_rng(rng)
        step = self.make_step(shape_in, shape_out, dt, rng, **kwargs)
        output = np.zeros((len(x),) + shape_out) if copy else x
        for i, xi in enumerate(x):
            output[i] = step((i+1) * dt, xi)
        return output

    def get_rng(self, rng):
        """Get a properly seeded independent RNG for the process step"""
        seed = rng.randint(maxint) if self.seed is None else self.seed
        return np.random.RandomState(seed)

    def make_step(self, shape_in, shape_out, dt, rng):
        """Create function that advances the process forward one time step."""
        raise NotImplementedError("Process must implement `make_step` method.")

    def run(self, t, d=None, dt=None, rng=np.random, **kwargs):
        """Run process without input for given length of time."""
        dt = self.default_dt if dt is None else dt
        n_steps = int(np.round(float(t) / dt))
        return self.run_steps(n_steps, d=d, dt=dt, rng=rng, **kwargs)

    def run_steps(self, n_steps, d=None, dt=None, rng=np.random, **kwargs):
        """Run process without input for given number of steps."""
        shape_in = as_shape(0)
        shape_out = as_shape(self.default_size_out if d is None else d)
        dt = self.default_dt if dt is None else dt
        rng = self.get_rng(rng)
        step = self.make_step(shape_in, shape_out, dt, rng, **kwargs)
        output = np.zeros((n_steps,) + shape_out)
        for i in range(n_steps):
            output[i] = step((i+1) * dt)
        return output

    def ntrange(self, n_steps, dt=None):
        """Create time points corresponding to a given number of steps."""
        dt = self.default_dt if dt is None else dt
        return dt * np.arange(1, n_steps + 1)

    def trange(self, t, dt=None):
        """Create time points corresponding to a given length of time."""
        dt = self.default_dt if dt is None else dt
        n_steps = int(np.round(float(t) / dt))
        return self.ntrange(n_steps, dt=dt)


class ProcessParam(Parameter):
    """Must be a Process."""

    def validate(self, instance, process):
        super(ProcessParam, self).validate(instance, process)
        if process is not None and not isinstance(process, Process):
            raise ValidationError(
                "Must be Process (got type %r)" % process.__class__.__name__,
                attr=self.name, obj=instance)
        return process
