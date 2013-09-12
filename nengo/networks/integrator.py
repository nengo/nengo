from .. import objects
from . import Network

class Integrator(Network):
    def connect_to(self, post, **kwargs):
        return self.ensemble.connect_to(post, **kwargs)

    @property
    def input(self):
        return self.node

    def make(self, recurrent_tau, **ens_args):
        self.node = self.add(objects.Node('In', output=lambda x: x))
        self.ensemble = self.add(objects.Ensemble('Integrator', **ens_args))
        self.ensemble.connect_to(self.ensemble, filter=recurrent_tau)
        self.node.connect_to(ensemble, transform=recurrent_tau)
