import nengo
from nengo.params import Default, IntParam
from nengo.spa.module import Module


class Scalar(Module):
    """A module capable of representing a single scalar.

    Parameters
    ----------
    n_neurons : int, optional (Default: 50)
        Number of neurons to represent the scalar.
    """

    n_neurons = IntParam('n_neurons', default=50, low=1, readonly=True)

    def __init__(self, n_neurons=Default, **kwargs):
        super(Scalar, self).__init__(**kwargs)

        self.n_neurons = n_neurons

        with self:
            self.scalar = nengo.Ensemble(self.n_neurons, 1)

        self.input = self.scalar
        self.output = self.scalar
        self.inputs = dict(default=(self.input, None))
        self.outputs = dict(default=(self.output, None))
