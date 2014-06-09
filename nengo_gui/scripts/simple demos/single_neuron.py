# # Nengo Example: A Single Neuron
# This demo shows you how to construct and manipulate a single leaky
# integrate-and-fire (LIF) neuron. The LIF neuron is a simple, standard
# neuron model, and here it resides inside a neural population, even though
# there is only one neuron.

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
    input = nengo.Node(0)

    # Connect the input signal to the neuron
    nengo.Connection(input, neuron)

    nengo.Probe(input)  # The original input
    nengo.Probe(neuron)  # The original input
    nengo.Probe(neuron, 'spikes')  # The raw spikes from the neuron


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 4.40367654885547
gui[model].offset = -92.41029646566608,41.51336865650782
gui[neuron].pos = 175.000, 50.000
gui[neuron].scale = 1.000
gui[input].pos = 50.000, 50.000
gui[input].scale = 1.000
