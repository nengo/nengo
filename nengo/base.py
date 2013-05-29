import numpy as np


class DuplicateFilter(Exception):
    """A signal appears as `newsig` in multiple filters.

    This is ambiguous because at simulation time, a filter means

        signals[newsig] <- alpha * signals[oldsig]
    """


class Signal(object):
    def __init__(self, n=1):
        self.n = n


class SignalProbe(object):
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt


class Constant(Signal):
    def __init__(self, n, value):
        Signal.__init__(self, n)
        self.value = value


class Population(object):
    # XXX rename this to PopulationLIF
    def __init__(self, n, bias=None, tau_rc=.02, tau_ref=.002):
        self.n = n
        if bias is None:
            bias = np.zeros(n)
        else:
            bias = np.asarray(bias, dtype=np.float64)
            if bias.shape != (n,):
                raise ValueError('shape', (bias.shape, n))
        self.bias = bias
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref


class Transform(object):
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig


class CustomTransform(object):
    def __init__(self, func, insig, outsig):
        self.func = func
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    def __init__(self, alpha, oldsig, newsig):
        self.oldsig = oldsig
        self.newsig = newsig
        self.alpha = alpha


class Encoder(object):
    def __init__(self, sig, pop, weights=None):
        self.sig = sig
        self.pop = pop
        assert isinstance(sig, Signal)
        assert isinstance(pop, Population)
        if weights is None:
            weights = np.random.randn(pop.n, sig.n)
        else:
            weights = np.asarray(weights)
            if weights.shape != (pop.n, sig.n):
                raise ValueError('weight shape', weights.shape)
        self.weights = weights


class Decoder(object):
    def __init__(self, pop, sig, weights=None):
        self.pop = pop
        self.sig = sig
        if weights is None:
            weights = np.random.randn(sig.n, pop.n)
        else:
            weights = np.asarray(weights)
            if weights.shape != (sig.n, pop.n):
                raise ValueError('weight shape', weights.shape)
        self.weights = weights


class Model(object):
    def __init__(self, dt):
        self.dt = dt
        self.signals = []
        self.populations = []
        self.encoders = []
        self.decoders = []
        self.transforms = []
        self.filters = []
        self.custom_transforms = []
        self.signal_probes = []

    def signal(self, n=1, value=None):
        if value is None:
            rval = Signal(n)
        else:
            rval = Constant(n, value)
        self.signals.append(rval)
        return rval

    def signal_probe(self, sig, dt):
        rval = SignalProbe(sig, dt)
        self.signal_probes.append(rval)
        return rval

    def population(self, *args, **kwargs):
        rval = Population(*args, **kwargs)
        self.populations.append(rval)
        return rval

    def encoder(self, sig, pop, weights=None):
        rval = Encoder(sig, pop, weights=weights)
        self.encoders.append(rval)
        return rval

    def decoder(self, pop, sig, weights=None):
        rval = Decoder(pop, sig, weights=weights)
        self.decoders.append(rval)
        return rval

    def transform(self, alpha, insig, outsig):
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        rval = Filter(alpha, oldsig, newsig)
        self.filters.append(rval)
        return rval

    def custom_transform(self, func, insig, outsig):
        rval = CustomTransform(func, insig, outsig)
        self.custom_transforms.append(rval)
        return rval


# ----------------------------------------------------------------------
# nengo_theano interface utilities
# ----------------------------------------------------------------------


def net_get_decoders(net, obj, origin_name, idx=None):
    ensemble = net.get_object(obj)
    try:
        origin = ensemble.origin[origin_name]
    except KeyError:
        print 'Origin names:', ensemble.origin.keys()
        raise
    rval = origin.decoders.get_value().astype('float32')
    r = origin.ensemble.radius
    rval = rval * r / net.dt
    if idx is None:
        assert rval.shape[0] == 1
        return rval[0].T
    else:
        return rval[idx].T

def net_get_encoders(net, obj, idx=None):
    ensemble = net.get_object(obj)
    encoders = ensemble.shared_encoders.get_value().astype('float32')
    # -- N.B. shared encoders already have "alpha" factored in
    if idx is None:
        assert encoders.shape[0] == 1
        return encoders[0]
    else:
        return encoders[idx]

def net_get_bias(net, obj, idx=None):
    ensemble = net.get_object(obj)
    bias = ensemble.bias.astype('float32')
    if idx is None:
        assert bias.shape[0] == 1
        return bias[0]
    else:
        return bias[idx]

try:
    import nengo.nef_theano.ensemble
    import nengo.nef_theano.filter
    import nengo.nef_theano.input
    import nengo.nef_theano.probe
except ImportError:
    pass

class Nengo2Model(object):
    def __init__(self, net):
        m = Model(net.dt)

        one = m.signal(value=1.0)
        steps = m.signal()
        simtime = m.signal()

        # -- hold all constants on the line
        m.filter(1.0, one, one)

        # -- steps counts by 1.0
        m.filter(1.0, steps, steps)
        m.filter(1.0, one, steps)

        # simtime <- dt * steps
        m.filter(net.dt, steps, simtime)

        memo = {}

        self.model = m
        self.memo = memo
        self.one = one
        self.steps = steps
        self.simtime = simtime
        self.add_net(net)

    def add_net(self, net):
        if net.tick_nodes:
            raise NotImplementedError(net.tick_nodes)
        for nodename, node in net.nodes.items():
            if isinstance(node,
                    nengo.nef_theano.input.ScalarFunctionOfTime):
                self.add_scalar_function_of_time(nodename, node)
            elif isinstance(node,
                    nengo.nef_theano.ensemble.Ensemble):
                self.add_ensemble(nodename, node)
            elif isinstance(node,
                    nengo.nef_theano.probe.Probe):
                self.add_probe(nodename, node)
            else:
                raise NotImplementedError(node)

    def add_scalar_function_of_time(self, nodename, node):
        print 'importing scalar function input:', nodename
        sint = m.signal()
        # XXX actually check *WHAT* function to use
        m.custom_transform(np.sin, simtime, sint)
        memo[node] = sint

    def add_ensemble(self, nodename, node):
        print 'importing ensemble:', nodename
        if node.array_size != 1:
            raise NotImplementedError()
        else:
            pop = self.model.population(node.neurons_num,
                    bias=node.bias.astype(np.float)[0])
            termination = self.model.signal()
            self.model.encoder(termination, pop,
                    weights=node.shared_encoders.get_value()[0])
            for origin_name, origin in node.origin.items():
                print '.. origin', origin
                decoded = self.model.signal()

                dweights = origin.decoders.get_value().astype('float32')
                r = origin.ensemble.radius
                dweights = dweights * r / self.model.dt
                dweights.shape = dweights.shape[:-1]
                self.model.decoder(pop, decoded, weights=dweights)

            for name, thing in node.encoded_input.items():
                print '.. encoded input', name, thing
                raise NotImplementedError('encoded input', name, thing)

            for name, thing in node.decoded_input.items():
                print '.. decoded input', name, thing
                if isinstance(thing,
                        nengo.nef_theano.filter.Filter):
                    sig = self.model.signal()
                    if thing.pstc >= self.model.dt:
                        decay = np.exp(-self.model.dt / thing.pstc)
                        self.model.filter(decay, sig, sig)
                        self.model.transform(1 - decay, sig, sig)
                    else:
                        self.model.transform(1, sig, sig)
                else:
                    raise NotImplementedError('decoded input', name, thing)

    def add_probe(self, nodename, node):
        target = node.target
        target_signal = self.memo[target]
        raise NotImplementedError('probes')


def nengo2model(net):
    N2M = Nengo2Model(net)
    return N2M.model, N2M.memo

