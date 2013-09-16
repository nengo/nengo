from .. import objects
from . import Network

class Integrator(Network):
    def connect_to(self, post, **kwargs):
        return self.ensemble.connect_to(post, **kwargs)

    @property
    def signal(self):
        return self.node.signal

    def make(self, recurrent_tau, **ens_args):
        self.node = self.add(objects.PassthroughNode("In"))
        self.ensemble = self.add(objects.Ensemble('Integrator', **ens_args))
        self.ensemble.connect_to(self.ensemble, filter=recurrent_tau)
        self.node.connect_to(self.ensemble, transform=recurrent_tau)

    def probe(self, *args, **kwargs):
        return self.ensemble.probe(*args, **kwargs)
