from .. import objects
from . import Network

class Oscillator(Network):
    def make(self, recurrent_tau, frequency, **ens_args):
        self.node = self.add(objects.Node('In', dimensions=2))
        self.ensemble = self.add(
            objects.Ensemble('Oscillator', dimensions=2, **ens_args))
        tA = [[1, -frequency * recurrent_tau], [frequency * recurrent_tau, 1]]
        self.ensemble.connect_to(
            self.ensemble, filter=recurrent_tau, transform=tA)
        self.node.connect_to(self.ensemble, transform=1)
