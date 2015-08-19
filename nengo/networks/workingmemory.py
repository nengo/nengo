import numpy as np

import nengo
from nengo.networks import EnsembleArray


def InputGatedMemory(n_neurons, dimensions, feedback=1.0,
                     difference_gain=1.0, recurrent_synapse=0.1,
                     difference_synapse=None, net=None):
    """Stores a given vector in memory, with input controlled by a gate."""
    if net is None:
        net = nengo.Network(label="Input Gated Memory")

    if difference_synapse is None:
        difference_synapse = recurrent_synapse

    n_total_neurons = n_neurons * dimensions

    with net:
        # integrator to store value
        net.mem = EnsembleArray(n_neurons, dimensions,
                                neuron_nodes=True, label="mem")
        nengo.Connection(net.mem.output, net.mem.input,
                         transform=feedback,
                         synapse=recurrent_synapse)

        # calculate difference between stored value and input
        net.diff = EnsembleArray(n_neurons, dimensions,
                                 neuron_nodes=True, label="diff")
        nengo.Connection(net.mem.output, net.diff.input, transform=-1)

        # feed difference into integrator
        nengo.Connection(net.diff.output, net.mem.input,
                         transform=difference_gain,
                         synapse=difference_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        net.gate = nengo.Node(size_in=1)
        nengo.Connection(net.gate, net.diff.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -10,
                         synapse=None)

        # reset input (if reset=1, remove all values, and set to 0)
        net.reset = nengo.Node(size_in=1)
        nengo.Connection(net.reset, net.mem.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -3,
                         synapse=None)

    net.input = net.diff.input
    net.output = net.mem.output

    return net
