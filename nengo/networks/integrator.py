from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.exceptions import ObsoleteError
from nengo.network import Network
from nengo.node import Node


class Integrator(Network):
    """An ensemble that accumulates input and maintains state.

    This is accomplished through scaling the input signal and recurrently
    connecting an ensemble to itself to maintain state.

    Parameters
    ----------
    recurrent_tau : float
        Time constant on the recurrent connection.
    n_neurons : int
        Number of neurons in the recurrently connected ensemble.
    dimensions : int
        Dimensionality of the input signal and ensemble.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    ensemble : Ensemble
        The recurrently connected ensemble.
    input : Node
        Provides the input signal.
    """

    def __init__(self, recurrent_tau, n_neurons, dimensions, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Integrator")
        super().__init__(**kwargs)

        with self:
            self.input = Node(size_in=dimensions)
            self.ensemble = Ensemble(n_neurons, dimensions=dimensions)
            Connection(self.ensemble, self.ensemble, synapse=recurrent_tau)
            Connection(self.input, self.ensemble, transform=recurrent_tau, synapse=None)

        self.output = self.ensemble
