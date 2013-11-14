from .. import objects
from . import Network

class Integrator(Network):
    def make(self, recurrent_tau, **ens_args):
        self.node = self.add(objects.Node("In"))
        self.ensemble = self.add(objects.Ensemble('Integrator', **ens_args))
        self.ensemble.connect_to(self.ensemble, filter=recurrent_tau)
        self.node.connect_to(self.ensemble, transform=recurrent_tau)
