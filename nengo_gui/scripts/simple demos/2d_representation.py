# # Nengo Example: 2-Dimensional Representation
#
# Ensembles of neurons represent information. In Nengo, we represent that
# information with real-valued vectors -- lists of numbers. In this example,
# we will represent a two-dimensional vector with a single ensemble of leaky
# integrate-and-fire neurons.

import nengo
import numpy as np

model = nengo.Network(label='2D Representation')
with model:
    neurons = nengo.Ensemble(100, dimensions=2, label="neurons")

    input = nengo.Node([0, 0])
    nengo.Connection(input, neurons)

    nengo.Probe(input)
    nengo.Probe(neurons)
    nengo.Probe(neurons, 'spikes')



import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 4.4036765488554686
gui[model].offset = -92.41029646566392,43.50878459270723
gui[neurons].pos = 175.000, 50.000
gui[neurons].scale = 1.000
gui[input].pos = 50.000, 50.000
gui[input].scale = 1.000
