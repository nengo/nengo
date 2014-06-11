import numpy as np

import nengo
from nengo.networks import EnsembleArray


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, mem_synapse=0.1, fdbk_scale=1.0,
                 difference_gain=1.0, gate_gain=10, reset_gain=3,
                 **mem_args):

        self.input = nengo.Node(size_in=dimensions)
        self.output = nengo.Node(size_in=dimensions)

        # integrator to store value
        self.mem = EnsembleArray(n_neurons, dimensions, label="mem",
                                 **mem_args)
        nengo.Connection(self.mem.output, self.mem.input, synapse=mem_synapse,
                         transform=np.eye(dimensions) * fdbk_scale)

        # calculate difference between stored value and input
        self.diff = EnsembleArray(n_neurons, dimensions, label="diff")
        nengo.Connection(self.input, self.diff.input, synapse=None)
        nengo.Connection(self.mem.output, self.diff.input,
                         transform=np.eye(dimensions) * -1)

        # feed difference into integrator
        nengo.Connection(self.diff.output, self.mem.input,
                         transform=np.eye(dimensions) * difference_gain)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        self.gate = nengo.Node(size_in=1)
        for e in self.diff.ensembles:
            nengo.Connection(self.gate, e.neurons,
                             transform=[[-gate_gain]] * e.n_neurons)

        # reset input (if reset=1, remove all values stored, and set values
        # to 0)
        self.reset_node = nengo.Node(size_in=1)
        for e in self.mem.ensembles:
            nengo.Connection(self.reset_node, e.neurons,
                             transform=[[-reset_gain]] * e.n_neurons)

        nengo.Connection(self.mem.output, self.output, synapse=None)


class FeedbackGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, fdbk_synapse=0.1,
                 conn_synapse=0.005, fdbk_scale=1.0, gate_gain=2.0,
                 reset_gain=3, **ens_args):

        self.input = nengo.Node(size_in=dimensions)
        self.output = nengo.Node(size_in=dimensions)

        # gate control signal (if gate==0, update stored value, otherwise
        # retain stored value)
        self.gate = nengo.Node(size_in=1)

        # integrator to store value
        self.mem = EnsembleArray(n_neurons, dimensions, label="mem",
                                 **ens_args)

        # ensemble to gate feedback
        self.fdbk = EnsembleArray(n_neurons, dimensions, label="fdbk",
                                  **ens_args)

        # ensemble to gate input
        self.in_gate = EnsembleArray(n_neurons, dimensions, label="in_gate",
                                     **ens_args)

        # calculate gating control signal
        self.ctrl = nengo.Ensemble(n_neurons, 1, label="ctrl")

        # Connection from mem to fdbk, and from fdbk to mem
        nengo.Connection(self.mem.output, self.fdbk.input,
                         synapse=fdbk_synapse - conn_synapse)
        nengo.Connection(self.fdbk.output, self.mem.input,
                         transform=np.eye(dimensions) * fdbk_scale,
                         synapse=conn_synapse)

        # Connection from input to in_gate, and from in_gate to mem
        nengo.Connection(self.input, self.in_gate.input, synapse=None)
        nengo.Connection(self.in_gate.output, self.mem.input,
                         synapse=conn_synapse)

        # Connection from gate to ctrl
        nengo.Connection(self.gate, self.ctrl, synapse=None)

        # Connection from ctrl to fdbk and in_gate
        for e in self.fdbk.ensembles:
            nengo.Connection(self.ctrl, e.neurons,
                             function=lambda x: [1 - x[0]],
                             transform=[[-gate_gain]] * e.n_neurons)
        for e in self.in_gate.ensembles:
            nengo.Connection(self.ctrl, e.neurons,
                             transform=[[-gate_gain]] * e.n_neurons)

        # Connection from mem to output
        nengo.Connection(self.mem.output, self.output, synapse=None)

        # reset input (if reset=1, remove all values stored, and set values
        # to 0)
        self.reset = nengo.Node(size_in=1)
        for e in self.mem.ensembles:
            nengo.Connection(self.reset, e.neurons,
                             transform=[[-reset_gain]] * e.n_neurons)
