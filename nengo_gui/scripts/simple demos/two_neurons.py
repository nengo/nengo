# # Nengo Example: Two Neurons
#
# This demo shows how to construct and manipulate a complementary pair of neurons.
#
# These are leaky integrate-and-fire (LIF) neurons. The neuron tuning properties have been selected so there is one on and one off neuron.
#
# One neuron will increase for positive input, and the other will decrease. This can be thought of as the simplest population that is able to give a reasonable representation of a scalar value.

import nengo
from nengo.objects import Uniform
import numpy as np

model = nengo.Network(label='Two Neurons')
with model:
    neurons = nengo.Ensemble(2, dimensions=1,  # Representing a scalar
                             intercepts=Uniform(-.5, -.5),  # Set the intercepts at .5
                             max_rates=Uniform(100,100),  # Set the max firing rate at 100hz
                             encoders=[[1],[-1]])  # One 'on' and one 'off' neuron

    sin = nengo.Node(lambda t: np.sin(8 * t), label = "sin")

    nengo.Connection(sin, neurons, synapse=0.01)

    sin_probe = nengo.Probe(sin)  # The original input
    spikes = nengo.Probe(neurons, 'spikes')  # Raw spikes from each neuron
    voltage = nengo.Probe(neurons, 'voltage')  # Subthreshold soma voltages of the neurons
    filtered = nengo.Probe(neurons, synapse=0.01)  # Spikes filtered by a 10ms post-synaptic filter
