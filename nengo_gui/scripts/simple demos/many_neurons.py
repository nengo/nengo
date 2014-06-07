# # Nengo Example: Many neurons
#
# This demo shows how to construct and manipulate a population of neurons.
#
# These are 100 leaky integrate-and-fire (LIF) neurons. The neuron tuning
# properties have been randomly selected.
#
# The input is a sine wave to show the effects of increasing or decreasing
# input. As a population, these neurons do a good job of representing a
# single scalar value. This can be seen by the fact that the input graph and
# neurons graphs match well.

import nengo
import numpy as np

model = nengo.Network(label='Many Neurons')
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1, label="A")

    sin = nengo.Node(lambda t: np.sin(8 * t), label="sin")  # Input is a sine

    # Connect the input to the population
    nengo.Connection(sin, A, synapse=0.01) # 10ms filter

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)  # 10ms filter
    A_spikes = nengo.Probe(A, 'spikes') # Collect the spikes


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.9855255998510186
gui[model].offset = 289.5077061203747,208.13879272738876
gui[A].pos = 150.000, 0.000
gui[A].scale = 1.000
gui[sin].pos = 0.000, 0.000
gui[sin].scale = 1.000
