from nengo.base import NengoObject, NengoObjectParam
from nengo.ensemble import Ensemble
from nengo.params import Default, DictParam, IntParam, NumberParam, StringParam


class Probe(NengoObject):
    """A probe is an object that receives data from the simulation.

    This is to be used in any situation where you wish to gather simulation
    data (spike data, represented values, neuron voltages, etc.) for analysis.

    Probes cannot directly affect the simulation.

    TODO: Example usage for each object.

    Parameters
    ----------
    target : Ensemble, Node, Connection
        The Nengo object to connect to the probe.
    attr : str, optional
        The quantity to probe. Refer to the target's ``probeable`` list for
        details. Defaults to the first element in the list.
    sample_every : float, optional
        Sampling period in seconds.
    conn_args : dict, optional
        Optional keyword arguments to pass to the Connection created for this
        probe. For example, passing ``synapse=pstc`` will filter the data.
    """

    target = NengoObjectParam(nonzero_size_out=True)
    attr = StringParam(default=None)
    sample_every = NumberParam(default=None, optional=True, low=1e-10)
    conn_args = DictParam(default=None)
    seed = IntParam(default=None, optional=True)

    def __init__(self, target, attr=Default, sample_every=Default,
                 **conn_args):
        if not hasattr(target, 'probeable') or len(target.probeable) == 0:
            raise TypeError(
                "Type '%s' is not probeable" % target.__class__.__name__)

        conn_args.setdefault('synapse', None)

        # We'll use the first in the list as default
        self.attr = attr if attr is not Default else target.probeable[0]

        if self.attr not in target.probeable:
            raise ValueError(
                "'%s' is not probeable for '%s'" % (self.attr, target))

        self.target = target
        self.sample_every = sample_every
        self.conn_args = conn_args
        self.seed = conn_args.get('seed', None)

    @property
    def label(self):
        return "Probe(%s.%s)" % (self.target.label, self.attr)

    @property
    def size_in(self):
        # TODO: A bit of a hack; make less hacky.
        if isinstance(self.target, Ensemble) and self.attr != "decoded_output":
            return self.target.neurons.size_out
        return self.target.size_out

    @property
    def size_out(self):
        return 0
