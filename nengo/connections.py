import random
import warnings

import numpy as np

# -- James and Terry arrived at this by eyeballing some graphs.
#    Not clear if this should be a constant at all, it
#    may depend on fn being estimated, number of neurons, etc...
DEFAULT_RCOND=0.01

class Connection(object):
    """Describes a connection between two Nengo objects.

    The connection encapsulates a lot of information that Nengo needs
    to compute a biologically plausible connection between two networks
    that implements some mathematical function.
    Alternatively, the connection could bypass this logic and just store
    a set of connection weights between two Ensembles.

    Attributes
    ----------
    pre : Nengo object
        The Nengo object on the presynaptic side of this connection.
    post : Nengo object
        The Nengo object on the postsynaptic side of this connection.
    transform : 2D matrix of floats
        If the connection operates in vector (state) space,
        ``transform`` is a two-dimensional array of floats
        that represents the linear transformation
        between ``pre`` and ``post``.
    weights : 2D matrix of floats
        If the connection operates in neuron space,
        ``weights`` is a two-dimensional array of floats
        that represents the connection weights
        between ``pre`` neurons and ``post`` neurons.
    decoders : 2D matrix of floats
        If the connection operates in vector space,
        it will have a set of decoders defined that
        maps the neural activity to a vector representation.
    filter : dict
        A dictionary describing the filter that is applied to
        presynaptic spikes before being communicated to ``post``.
    function : function
        The function that this connection implements.
    learning_rule : dict
        A dictionary describing a learning rule that
        modifies connection's decoders, weights,
        or both during a simulation.
    modulatory : bool
        A boolean indicating if the connection is modulatory.

        Modulatory connections do not impart current in ``post``.
        Instead, it can be used by ``post`` to do other operations
        (e.g., modulate learning).

    See Also
    --------
    Model.connect : Helper to make connections
    Model.connect_neurons : Helper to make direct connections

    """

    def __init__(self, pre, post, transform=1.0, weights=None, decoders=None,
                 filter=None, function=None, learning_rule=None,
                 eval_points=None, modulatory=False):
        if weights is not None:
            raise NotImplementedError()
        if decoders is not None:
            raise NotImplementedError()
        if filter is not None:
            raise NotImplementedError()
        if learning_rule is not None:
            raise NotImplementedError()

        # if function is None:
        #     function = lambda x: x

        if eval_points is None:
            eval_points = pre.babbling_signal ## TODO: eval_points ensemble prop

        self.pre = pre
        self.post = post

        if function is None:
            targets = eval_points.T
        else:
            targets = np.array([function(s) for s in eval_points.T])
            if len(targets.shape) < 2:
                targets.shape = targets.shape[0], 1

        n, = targets.shape[1:]
        dt = pre.model.dt

        # -- N.B. this is only accurate for models firing well
        #    under the simulator's dt.
        A = pre.neurons.babbling_rate * dt
        b = targets
        weights, res, rank, s = np.linalg.lstsq(A, b, rcond=rcond)

        sig = ensemble.model.add(Signal(n=n, name='%s[%i]' % (name, ii)))
        decoder = ensemble.model.add(Decoder(
                sig=sig,
                pop=ensemble.neurons[ii],
                weights=weights.T))

        # set up self.sig as an unfiltered signal
        transform = ensemble.model.add(Transform(1.0, sig, sig))

        self.sigs.append(sig)
        self.decoders.append(decoder)
        self.transforms.append(transform)


        # if isinstance(self.pre, Ensemble):
        #     self.decoder = sim.Decoder(self.pre.nl, self.pre.sig)
        #     self.decoder.desired_function = function
        #     self.transform = sim.Transform(np.asarray(transform),
        #                                   self.pre.sig,
        #                                   self.post.sig)

        # elif isinstance(self.pre, Node):
        #     if function is None:
        #         self.transform = sim.Transform(np.asarray(transform),
        #                                       self.pre.sig,
        #                                       self.post.sig)
        #     else:
        #         raise NotImplementedError()
        # else:
        #     raise NotImplementedError()

    def __str__(self):
        ret = "Connection (id " + str(id(self)) + "): \n"
        if hasattr(self, 'decoder'):
            return ret + ("    " + str(self.decoder) + "\n"
                          "    " + str(self.transform))
        else:
            return ret + "    " + str(self.transform)

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self.pre.name + ">" + self.post.name

    def add_to_model(self, model):
        if hasattr(self, 'decoder'):
            model.add(self.decoder)
        if hasattr(self, 'transform'):
            model.add(self.transform)
        if hasattr(self, 'filter'):
            model.add(self.filter)

    def remove_from_model(self, model):
        raise NotImplementedError


# class Probe(object):
#     def __init__(self, target, sample_every=None, filter=None):
#         if pstc is not None and pstc > self.dt:
#             fcoef, tcoef = _filter_coefs(pstc=pstc, dt=self.dt)
#             probe_sig = self.signal(obj.sig.n)
#             self.filter(fcoef, probe_sig, probe_sig)
#             self.transform(tcoef, obj.sig, probe_sig)
#             self.probe = SimModel.probe(self, probe_sig, sample_every)


#     @staticmethod
#     def filter_coefs(pstc, dt):
#         pstc = max(pstc, dt)
#         decay = math.exp(-dt / pstc)
#         return decay, (1.0 - decay)


