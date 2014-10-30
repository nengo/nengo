import numpy as np

import nengo
from nengo.networks import EnsembleArray


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, mem_synapse=0.1, fdbk_scale=1.0,
                 difference_gain=1.0, gate_gain=10, reset_gain=3,
                 **mem_args):
        # integrator to store value
        self.mem = EnsembleArray(n_neurons, dimensions,
                                 neuron_nodes=True, label="mem", **mem_args)
        nengo.Connection(self.mem.output, self.mem.input,
                         synapse=mem_synapse, transform=fdbk_scale)

        # calculate difference between stored value and input
        self.diff = EnsembleArray(n_neurons, dimensions,
                                  neuron_nodes=True, label="diff")
        nengo.Connection(self.mem.output, self.diff.input, transform=-1)

        # feed difference into integrator
        nengo.Connection(self.diff.output, self.mem.input,
                         transform=difference_gain,
                         synapse=mem_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        neuron_trans = np.ones((n_neurons * dimensions, 1))
        self.gate = nengo.Node(size_in=1)
        nengo.Connection(self.gate, self.diff.neuron_input,
                         transform=neuron_trans * -gate_gain)

        # reset input (if reset=1, remove all values stored, and set values=0)
        self.reset = nengo.Node(size_in=1)
        nengo.Connection(self.reset, self.mem.neuron_input,
                         transform=neuron_trans * -reset_gain)

        self.input = self.diff.input
        self.output = self.mem.output
