import logging

import numpy as np

from . import builder
from . import decoders as decsolve
from . import simulator


logger = logging.getLogger(__name__)


class SignalConnection(object):
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

    def __repr__(self):
        return str(self)

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
            probe = builder.Probe(None, sample_every)
            self.probes['signal'].append(probe)
        return probe

    def add_to_model(self, model):
        model.connections.append(self)

    def _add_filter(self, model, dt):
        if self.filter is not None and self.filter > dt:
            # Set up signal
            name = self.pre.name + ".filtered(%f)" % self.filter
            self.signal = builder.Signal(n=self.pre.size, name=name)
            model.add(self.signal)

            # Set up filters and transforms
            o_coef, n_coef = builder.filter_coefs(pstc=self.filter, dt=dt)
            model._operators += [simulator.ProdUpdate(builder.Constant(n_coef),
                                                 self.pre,
                                                 builder.Constant(o_coef),
                                                 self.signal)]

        else:
            # Signal should already be in the model
            self.signal = self.pre

    def _add_transform(self, model):
        model._operators += [simulator.DotInc(builder.Constant(self.transform),
                                              self.signal,
                                              self.post)]

    def _add_probes(self, model):
        for probe in self.probes['signal']:
            probe.sig = self.signal
            model.add(probe)

    def build(self, model, dt):
        # Pre / post may be high level objects (ensemble, node) or signals
        if not builder.is_signal(self.pre):
            self.pre = self.pre.signal
#            if not builder.is_constant(self.pre):
#                logger.warning("SignalConnection is usually used for "
#                               "connecting raw Signals and ConstantNodes. "
#                               "Are you sure you shouldn't be using "
#                               "DecodedConnection?")

        if not builder.is_signal(self.post):
            self.post = self.post.signal

        # Set up filters and transform
        self._add_filter(model, dt)
        self._add_transform(model)

        # Set up probes
        self._add_probes(model)


class NonlinearityConnection(SignalConnection):
    """A NonlinearityConnection connects a nonlinearity to a Signal
    (or objects with those) via a transform and a filter.

    Attributes
    ----------
    name
    pre
    post

    filter : type
        description
    transform

    probes : type
        description

    """
    def __init__(self, pre, post, **kwargs):
        SignalConnection.__init__(self, pre, post, **kwargs)

    def __str__(self):
        return self.name + " (NonlinearityConnection)"

    def build(self, model, dt):
        # Pre must be a nonlinearity
        if not isinstance(self.pre, builder.Nonlinearity):
            self.pre = self.pre.nonlinear

        # then get the output signal of the nonlinearity
        if not builder.is_signal(self.pre):
            self.pre = self.pre.output_signal

        # Post could be a node / ensemble, etc
        if isinstance(self.post, builder.Nonlinearity):
            if isinstance(self.post, builder.GainNonlinearity):
                self.transform = self.transform * self.post.gain[:,None]
            self.post = self.post.input_signal
        elif not builder.is_signal(self.post):
            self.post = self.post.signal

        # Set up filters and transform
        self._add_filter(model, dt)
        self._add_transform(model)

        # Set up probes
        self._add_probes(model)


class DecodedConnection(SignalConnection):
    """A DecodedConnection connects an ensemble to a Signal
    via a set of decoders, a transform, and a filter.

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
        SignalConnection.__init__(self, pre, post, **kwargs)

        self.decoders = kwargs.get('decoders', None)
        self.decoder_solver = kwargs.get('decoder_solver',
                                         decsolve.least_squares)
        self.eval_points = kwargs.get('eval_points', None)
        self.function = kwargs.get('function', None)
        # self.modulatory = kwargs.get('modulatory', False)
        if 'modulatory' in kwargs:
            raise NotImplementedError('modulatory')

    @property
    def decoders(self):
        return None if self._decoders is None else self._decoders.T

    @decoders.setter
    def decoders(self, _decoders):
        if _decoders is not None and self.function is not None:
            logger.warning("Setting decoders on a connection with a specified "
                           "function. May not actually compute that function.")

        if _decoders is not None:
            _decoders = np.asarray(_decoders)
            if _decoders.shape[0] != self.pre.n_neurons:
                msg = ("Decoders axis 0 must be %d; in this case it is "
                       "%d. (shape=%s)" % (self.pre.n_neurons, _decoders.shape[0], _decoders.shape))
                raise builder.ShapeMismatch(msg)

        self._decoders = None if _decoders is None else _decoders.T

    @property
    def dimensions(self):
        if self.function is None:
            return self.pre.dimensions
        else:
            return np.array(self.function(np.ones(self.pre.dimensions,))).size

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

    def __str__(self):
        return self.name + " (DecodedConnection)"

    def _add_filter(self, model, dt):
        if self.filter is not None and self.filter > dt:
            o_coef, n_coef = builder.filter_coefs(pstc=self.filter, dt=dt)

            model._operators += [simulator.ProdUpdate(builder.Constant(self._decoders*n_coef),
                                                      self.pre,
                                                      builder.Constant(o_coef),
                                                      self.signal)]
        else:
            model._operators += [simulator.ProdUpdate(builder.Constant(self._decoders),
                                                      self.pre,
                                                      builder.Constant(0),
                                                      self.signal)]

    def build(self, model, dt):
        # Pre must be an ensemble -- but, don't want to import objects
        assert self.pre.__class__.__name__ == "Ensemble"

        # Post could be a node / ensemble, etc
        if isinstance(self.post, builder.Nonlinearity):
            if isinstance(self.post, builder.GainNonlinearity):
                self.transform = self.transform * self.post.gain[:,None]
            self.post = self.post.input_signal
        elif not builder.is_signal(self.post):
            self.post = self.post.signal

        # Set up signal
        dims = self.dimensions
        self.signal = builder.Signal(dims, name=self.name)
        model.add(self.signal)

        # Set up decoders
        if self._decoders is None:
            activities = self.pre.activities(self.eval_points) * dt
            if self.function is None:
                targets = self.eval_points
            else:
                targets = np.array(
                    [self.function(ep) for ep in self.eval_points])
                if len(targets.shape) < 2:
                    targets.shape = targets.shape[0], 1
            self._decoders = self.decoder_solver(activities, targets)

        # Set up filters and transform
        self.pre = self.pre.neurons.output_signal
        self._add_filter(model, dt)
        self._add_transform(model)

        # Set up probes
        self._add_probes(model)


class ConnectionList(object):
    """A connection made up of several other connections."""
    def __init__(self, connections, transform=1.0):
        self.connections = connections
        self.transform = transform
        self.probes = {}

    def add_to_model(self, model):
        model.connections.append(self)

    def build(self, model, dt):
        self.transform = np.asarray(self.transform)

        i = 0
        for connection in self.connections:
            pre_dim = connection.dimensions

            if self.transform.ndim == 0:
                trans = np.zeros((connection.post.dimensions, pre_dim))
                np.fill_diagonal(trans[i:i+pre_dim,:], self.transform)
            elif self.transform.ndim == 2:
                trans = self.transform[:,i:i+pre_dim]
            else:
                raise NotImplementedError(
                    "Only transforms with 0 or 2 ndims are accepted")

            i += pre_dim

            connection.transform = trans
            connection.build(model, dt)
