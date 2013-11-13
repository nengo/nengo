import nengo
from .. import objects
from . import Network

class Integrator(Network):
    def make(self, recurrent_tau, **ens_args):
        with self:
            self.input = objects.Node()
            self.ensemble = objects.Ensemble(**ens_args)
            nengo.DecodedConnection(self.ensemble, self.ensemble, filter=recurrent_tau)
            nengo.Connection(self.input, self.ensemble, transform=recurrent_tau)
