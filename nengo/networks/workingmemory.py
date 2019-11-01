import numpy as np

import nengo
from nengo.exceptions import ObsoleteError
from nengo.networks import EnsembleArray


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate.

    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.
    dimensions : int
        Dimensionality of the vector.

    feedback : float, optional
        Strength of the recurrent connection from the memory to itself.
    difference_gain : float, optional
        Strength of the connection from the difference ensembles to the
        memory ensembles.
    recurrent_synapse : float, optional

    difference_synapse : Synapse
        If None, ...
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    diff : EnsembleArray
        Represents the difference between the desired vector and
        the current vector represented by ``mem``.
    gate : Node
        With input of 0, the network is not gated, and ``mem`` will be updated
        to minimize ``diff``. With input greater than 0, the network will be
        increasingly gated such that ``mem`` will retain its current value,
        and ``diff`` will be inhibited.
    input : Node
        The desired vector.
    mem : EnsembleArray
        Integrative population that stores the vector.
    output : Node
        The vector currently represented by ``mem``.
    reset : Node
        With positive input, the ``mem`` population will be inhibited,
        effectively wiping out the vector currently being remembered.
    """

    def __init__(
        self,
        n_neurons,
        dimensions,
        feedback=1.0,
        difference_gain=1.0,
        recurrent_synapse=0.1,
        difference_synapse=None,
        **kwargs
    ):

        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Input gated memory")
        super().__init__(**kwargs)

        if difference_synapse is None:
            difference_synapse = recurrent_synapse

        n_total_neurons = n_neurons * dimensions

        with self:
            # integrator to store value
            self.mem = EnsembleArray(n_neurons, dimensions, label="mem")
            nengo.Connection(
                self.mem.output,
                self.mem.input,
                transform=feedback,
                synapse=recurrent_synapse,
            )

            # calculate difference between stored value and input
            self.diff = EnsembleArray(n_neurons, dimensions, label="diff")
            nengo.Connection(self.mem.output, self.diff.input, transform=-1)

            # feed difference into integrator
            nengo.Connection(
                self.diff.output,
                self.mem.input,
                transform=difference_gain,
                synapse=difference_synapse,
            )

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Node(size_in=1)
            self.diff.add_neuron_input()
            nengo.Connection(
                self.gate,
                self.diff.neuron_input,
                transform=np.ones((n_total_neurons, 1)) * -10,
                synapse=None,
            )

            # reset input (if reset=1, remove all values, and set to 0)
            self.reset = nengo.Node(size_in=1)
            nengo.Connection(
                self.reset,
                self.mem.add_neuron_input(),
                transform=np.ones((n_total_neurons, 1)) * -3,
                synapse=None,
            )

        self.input = self.diff.input
        self.output = self.mem.output
