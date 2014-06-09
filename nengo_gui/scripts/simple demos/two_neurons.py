# # Nengo Example: Two Neurons
#
# This demo shows how to construct and manipulate a complementary pair of
# neurons.
#
# These are leaky integrate-and-fire (LIF) neurons. The neuron tuning
# properties have been selected so there is one on and one off neuron.
#
# One neuron will increase for positive input, and the other will decrease.
# This can be thought of as the simplest population that is able to give a
# reasonable representation of a scalar value.

import nengo
from nengo.objects import Uniform
import numpy as np

model = nengo.Network(label='Two Neurons')
with model:
    neurons = nengo.Ensemble(2, dimensions=1,  # Representing a scalar
                             intercepts=Uniform(-.5, -.5),  # Set the intercepts at .5
                             max_rates=Uniform(100,100),  # Set the max firing rate at 100hz
                             encoders=[[1],[-1]])  # One 'on' and one 'off' neuron

    input = nengo.Node([0])

    nengo.Connection(input, neurons, synapse=0.01)

    nengo.Probe(input)  # The original input
    nengo.Probe(neurons, 'spikes')  # Raw spikes from each neuron
    nengo.Probe(neurons)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 4.4036765488554686
gui[model].offset = -92.41029646566369,74.81617255722708
gui[neurons].pos = 175.000, 50.000
gui[neurons].scale = 1.000
gui[input].pos = 50.000, 50.000
gui[input].scale = 1.000
