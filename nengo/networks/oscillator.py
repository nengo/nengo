from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.exceptions import ObsoleteError
from nengo.network import Network
from nengo.node import Node


class Oscillator(Network):
    """A two-dimensional ensemble with interacting recurrent connections.

    The ensemble connects to itself in a manner similar to the integrator;
    however, here the two dimensions interact with each other to implement
    a cyclic oscillator.

    Parameters
    ----------
    recurrent_tau : float
        Time constant on the recurrent connection.
    frequency : float
        Desired frequency, in radians per second, of the cyclic oscillation.
    n_neurons : int
        Number of neurons in the recurrently connected ensemble.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    ensemble : Ensemble
        The recurrently connected oscillatory ensemble.
    input : Node
        Provides the input signal.
    """

    def __init__(self, recurrent_tau, frequency, n_neurons, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Oscillator")
        super().__init__(**kwargs)

        with self:
            self.input = Node(label="In", size_in=2)
            self.ensemble = Ensemble(n_neurons, dimensions=2, label="Oscillator")

            tA = [[1, -frequency * recurrent_tau], [frequency * recurrent_tau, 1]]
            Connection(
                self.ensemble, self.ensemble, synapse=recurrent_tau, transform=tA
            )
            Connection(self.input, self.ensemble, synapse=None)

        self.output = self.ensemble
