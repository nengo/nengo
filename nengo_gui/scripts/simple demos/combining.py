# # Nengo Example: Combining
#
# This example demonstrates how to create a neuronal ensemble that will
# combine two 1-D inputs into one 2-D representation.

import nengo
import numpy as np

model = nengo.Network(label='Combining')
with model:
    # Our input ensembles consist of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1, label = "A")
    B = nengo.Ensemble(100, dimensions=1, label = "B")

    # The output ensemble consists of 200 leaky integrate-and-fire neurons,
    # representing a two-dimensional signal
    output = nengo.Ensemble(200, dimensions=2, label='2D Population')

    # Create input nodes generating the sine and cosine
    sin = nengo.Node(output=np.sin, label = "sin")
    cos = nengo.Node(output=np.cos, label = "cos")

    nengo.Connection(sin, A)
    nengo.Connection(cos, B)

    # The square brackets define which dimension the input will project to
    nengo.Connection(A, output[1])
    nengo.Connection(B, output[0])

    sin_probe = nengo.Probe(sin)
    cos_probe = nengo.Probe(cos)
    A_probe = nengo.Probe(A, synapse=0.01)  # 10ms filter
    B_probe = nengo.Probe(B, synapse=0.01)  # 10ms filter
    out_probe = nengo.Probe(output, synapse=0.01)  # 10ms filter



import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.7724901181981698
gui[model].offset = 270.09581952027884,149.19433282117512
gui[A].pos = 100.000, 0.000
gui[A].scale = 1.000
gui[B].pos = 100.000, 100.000
gui[B].scale = 1.000
gui[output].pos = 200.000, 50.000
gui[output].scale = 1.000
gui[sin].pos = 0.000, 0.000
gui[sin].scale = 1.000
gui[cos].pos = 0.000, 100.000
gui[cos].scale = 1.000
