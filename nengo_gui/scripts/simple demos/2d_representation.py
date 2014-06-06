# # Nengo Example: 2-Dimensional Representation
# 
# Ensembles of neurons represent information. In Nengo, we represent that information with real-valued vectors -- lists of numbers. In this example, we will represent a two-dimensional vector with a single ensemble of leaky integrate-and-fire neurons.

import nengo
import numpy as np

model = nengo.Network(label='2D Representation')
with model:
    neurons = nengo.Ensemble(100, dimensions=2, label="neurons")

    sin = nengo.Node(output=np.sin, label="sin")
    cos = nengo.Node(output=np.cos, label="cos")

    nengo.Connection(sin, neurons[0])
    nengo.Connection(cos, neurons[1])

    sin_probe = nengo.Probe(sin, 'output')
    cos_probe = nengo.Probe(cos, 'output')
    neurons_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.01)  

