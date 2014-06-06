# # Nengo Example: A Single Neuron
# This demo shows you how to construct and manipulate a single leaky integrate-and-fire (LIF) neuron. The LIF neuron is a simple, standard neuron model, and here it resides inside a neural population, even though there is only one neuron.

import nengo
from nengo.objects import Uniform
model = nengo.Network(label='A Single Neuron')
with model:
    neuron = nengo.Ensemble(1, dimensions=1, # Represent a scalar
                            intercepts=Uniform(-.5, -.5),  # Set intercept to 0.5
                            max_rates=Uniform(100, 100),  # Set the maximum firing rate of the neuron to 100hz
                            encoders=[[1]], # Sets the neurons firing rate to increase for positive input
                            label = "A")  

import numpy as np
with model:
    cos = nengo.Node(lambda t: np.cos(8 * t), label = "cos")

    # Connect the input signal to the neuron
    nengo.Connection(cos, neuron)

    cos_probe = nengo.Probe(cos)  # The original input
    spikes = nengo.Probe(neuron, 'spikes')  # The raw spikes from the neuron
    voltage = nengo.Probe(neuron, 'voltage')  # Subthreshold soma voltage of the neuron
    filtered = nengo.Probe(neuron, synapse=0.01) # Spikes filtered by a 10ms post-synaptic filter
