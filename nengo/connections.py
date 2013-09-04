import logging

import numpy as np

from . import core
from . import decoders as decsolve

logger = logging.getLogger(__name__)


class SimpleConnection(object):
    """A SimpleConnection connects two Signals (or objects with signals)
    via a transform and a filter.

    Attributes
    ----------
    name
    filter : type
        description
    transform

    probes : type
        description

    """
    def __init__(self, pre, post, **kwargs):
        self.pre = pre
        self.post = post

        self.filter = kwargs.get('filter', 0.005)
        self.transform = kwargs.get('transform', 1.0)

        self.probes = {'signal': []}

    def __str__(self):
        return self.name + " (SimpleConnection)"

    @property
    def name(self):
        return self.pre.name + ">" + self.post.name

    @property
    def transform(self):
        """TODO"""
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = np.asarray(_transform)

    def probe(self, to_probe='signal', sample_every=0.001, filter=None):
        if filter is not None and filter > self.filter:
            raise ValueError("Cannot create filtered probes on connections; "
                             "the connection is already filtered.")

        if to_probe == 'signal':
            probe = core.Probe(None, sample_every)
            self.probes['signal'].append(probe)
        return probe

    def build(self, model, dt):
        if not isinstance(self.pre, core.SignalView):
            self.pre = pre.signal
        if not isinstance(self.post, core.SignalView):
            self.post = post.input_signal

        if self.filter is not None and self.filter > dt:
            # Set up signal
            name = self.pre.name + ".filtered(%f)" % self.filter
            self.signal = core.Signal(n=self.pre.size, name=name)
            model.add(self.signal)

            # Set up filters and transforms
            fcoef, tcoef = core.filter_coefs(pstc=self.filter, dt=dt)
            if core.is_constant(self.pre):
                self.sig_transform = core.Filter(tcoef, self.pre, self.signal)
            else:
                self.sig_transform = core.Transform(
                    tcoef, self.pre, self.signal)
            model.add(self.sig_transform)
            self.sig_filter = core.Filter(fcoef, self.signal, self.signal)
            model.add(self.sig_filter)
        else:
            # Signal should already be in the model
            self.signal = self.pre

        self.trans_filter = core.Filter(self.transform, self.signal, self.post)
        model.add(self.trans_filter)

        # Set up probes
        for probe in self.probes['signal']:
            probe.sig = self.signal
            model.add(probe)


class DecodedConnection(object):
    """A DecodedConnection connects a nonlinearity to a Signal
    (or objects with those) via a set of decoders, a transform, and a filter.

    Attributes
    ----------
    name
    pre
    post

    decoders
    eval_points
    filter : type
        description
    function : type
        description
    transform

    probes : type
        description

    """
    def __init__(self, pre, post, **kwargs):
        self.pre = pre
        self.post = post

        if 'modulatory' in kwargs:
            raise NotImplementedError('modulatory')

        self.decoders = kwargs.get('decoders', None)
        self.eval_points = kwargs.get('eval_points', None)
        self.filter = kwargs.get('filter', 0.005)
        self.function = kwargs.get('function', None)
        # self.modulatory = kwargs.get('modulatory', False)
        self.transform = kwargs.get('transform', 1.0)

        self.probes = {'signal': []}

    @property
    def decoders(self):
        return self._decoders

    @decoders.setter
    def decoders(self, _decoders):
        if _decoders is not None and self.function is not None:
            logger.warning("Setting decoders on a connection with a specified "
                           "function. May not actually compute that function.")

        if _decoders is not None:
            _decoders = np.asarray(_decoders)
            if _decoders.shape[0] != self.pre.n_neurons:
                msg = ("Decoders axis 0 must be n_neurons; in this case, "
                       "%d." % self.pre.n_neurons)
                raise core.ShapeMismatchError(msg)

        self._decoders = _decoders

    @property
    def eval_points(self):
        if self._eval_points is None:
            # OK because ensembles always build first
            return self.pre.eval_points
        return self._eval_points

    @eval_points.setter
    def eval_points(self, _eval_points):
        if _eval_points is not None:
            _eval_points = np.asarray(_eval_points)
            if len(_eval_points.shape) == 1:
                _eval_points.shape = (1, _eval_points.shape[0])
        self._eval_points = _eval_points

    @property
    def name(self):
        name = self.pre.name + ">" + self.post.name
        if self.function is not None:
            return name + ":" + self.function.__name__
        return name

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, _transform):
        self._transform = np.asarray(_transform)

    def __str__(self):
        return self.name + " (DecodedConnection)"

    def __repr__(self):
        return str(self)

    def probe(self, to_probe='signal', sample_every=0.001, filter=None):
        if filter is not None and filter > self.filter:
            raise ValueError("Cannot create filtered probes on connections; "
                             "the connection is already filtered.")

        if to_probe == 'signal':
            probe = core.Probe(sample_every)
            self.probes['signal'].append(probe)
        return probe

    def build(self, model, dt):
        if not isinstance(self.post, core.SignalView):
            self.post = post.input_signal

        # Set up signal
        dims = len(function(self.eval_points.T[0]))
        self.signal = core.Signal(n, name=self.name)
        model.add(self.signal)

        # Set up decoders
        if self.decoders is None:
            activities = self.pre.activities(self.eval_points) * dt
            if self.function is None:
                targets = self.eval_points.T
            else:
                targets = np.array([function(ep) for ep in self.eval_points.T])
                if len(targets.shape) < 2:
                    targets.shape = targets.shape[0], 1
            decoders = decsolve.solve_decoders(activities, targets)
        else:
            decoders = self.decoders
        self.decoder = core.Decoder(pre.neurons, self.signal, decoders)
        if self.function is not None:
            self.decoder.desired_function = function
        model.add(self.decoder)

        # Set up filters and transforms
        if self.filter is not None and self.filter > dt:
            fcoef, tcoef = core.filter_coefs(pstc=filter, dt=dt)
            self.sig_transform = core.Transform(tcoef, self.signal, self.signal)
            model.add(self.sig_transform)
            self.sig_filter = core.Filter(fcoef, self.signal, self.signal)
            model.add(self.sig_filter)
            self.trans_filter = core.Filter(
                self.transform, self.signal, self.post)
            model.add(self.trans_filter)
        else:
            self.trans_transform = core.Transform(
                self.transform, self.signal, self.post)
            model.add(self.trans_transform)

        # Set up probes, but don't build them
        for probe in self.probes['signal']:
            probe.sig = self.signal
            model.add(probe)


class ConnectionList(object):
    """A connection made up of several other connections."""
    def __init__(self, connections):
        self.connections = connections
        self.probes = {}

    def build(self, model, dt):
        for connection in self.connection:
            connection.build(model, dt)
