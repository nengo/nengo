import nengo
from nengo.spa.module import Module


class Scalar(Module):
    """A module capable of representing a single scalar."""

    def __init__(
            self, n_neurons=50, label=None, seed=None, add_to_container=None):
        super(Scalar, self).__init__(label, seed, add_to_container)

        with self:
            self.scalar = nengo.Ensemble(n_neurons, 1)

        self.input = self.scalar
        self.output = self.scalar
        self.inputs = dict(default=(self.input, None))
        self.outputs = dict(default=(self.output, None))
