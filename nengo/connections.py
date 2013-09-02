import logging

import numpy as np

from . import core
from . import decoders as decsolve

logger = logging.getLogger(__name__)


class SimpleConnection(object):
    """A SimpleConnection connects two Signals (or objects with signals)
    via a transform and a filter.

    """
    def __init__(self, pre, post, transform=1.0, filter=0.005, dt=0.001):
        if not isinstance(pre, core.SignalView):
            pre = pre.signal
        if not isinstance(post, core.SignalView):
            post = post.input_signal

        self.pre = pre
        self.post = post

        if filter is not None and filter > dt:
            name = self.pre.name + ".filtered(%f)" % filter
            self.signal = core.Signal(n=self.pre.size, name=name)
            fcoef, tcoef = core.filter_coefs(pstc=filter, dt=dt)
            if core.is_constant(self.pre):
                self.sig_transform = core.Filter(tcoef, self.pre, self.signal)
            else:
                self.sig_transform = core.Transform(
                    tcoef, self.pre, self.signal)
            self.sig_filter = core.Filter(fcoef, self.signal, self.signal)
            self.filter = core.Filter(transform, self.signal, self.post)
        else:
            self.filter = core.Filter(transform, self.pre, self.post)

    def __str__(self):
        return self.name + " (SimpleConnection)"

    @property
    def name(self):
        return self.pre.name + ">" + self.post.name

    def add_to_model(self, model):
        if hasattr(self, 'signal'):
            model.add(self.signal)
            model.add(self.sig_transform)
            model.add(self.sig_filter)
        model.add(self.filter)


class DecodedConnection(object):
    """A DecodedConnection connects a nonlinearity to a Signal
    (or objects with those) via a set of decoders, a transform, and a filter.

    """
    def __init__(self, pre, post, transform=1.0, decoders=None,
                 filter=0.005, function=None, learning_rule=None,
                 eval_points=None, modulatory=False, dt=0.001):
        if decoders is not None:
            raise NotImplementedError()
        if learning_rule is not None:
            raise NotImplementedError()

        transform = np.asarray(transform)

        if not isinstance(post, core.SignalView):
            post = post.input_signal

        self.pre = pre
        self.post = post
        self.function = function

        if eval_points is None:
            eval_points = pre.eval_points

        if function is None:
            targets = eval_points.T
        else:
            targets = np.array([function(s) for s in eval_points.T])
            if len(targets.shape) < 2:
                targets.shape = targets.shape[0], 1

        n, = targets.shape[1:]

        # -- N.B. this is only accurate for models firing well
        #    under the simulator's dt.
        activities = pre.activities(eval_points) * dt

        self.signal = core.Signal(n, name=self.name)
        self.decoders = core.Decoder(
            sig=self.signal, pop=pre.neurons,
            weights=decsolve.solve_decoders(activities, targets))
        if function is not None:
            self.decoders.desired_function = function

        if filter is not None and filter > dt:
            fcoef, tcoef = core.filter_coefs(pstc=filter, dt=dt)
            self.sig_transform = core.Transform(tcoef, self.signal, self.signal)
            self.sig_filter = core.Filter(fcoef, self.signal, self.signal)
            self.filter = core.Filter(transform, self.signal, self.post)
        else:
            self.transform = core.Transform(transform, self.signal, self.post)

    def __str__(self):
        return self.name + " (DecodedConnection)"

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        name = self.pre.name + ">" + self.post.name
        if self.function is not None:
            return name + ":" + self.function.__name__
        return name

    def add_to_model(self, model):
        model.add(self.signal)
        model.add(self.decoders)
        if hasattr(self, 'transform'):
            model.add(self.transform)
        if hasattr(self, 'sig_transform'):
            model.add(self.sig_transform)
            model.add(self.sig_filter)
            model.add(self.filter)

    def remove_from_model(self, model):
        raise NotImplementedError


class ConnectionList(object):
    """A connection made up of several other connections."""
    def __init__(self, connections):
        self.connections = connections

    def add_to_model(self, model):
        for connection in self.connections:
            model.add(connection)

    def remove_from_model(self, model):
        for connection in self.connections:
            model.remove(connection)
