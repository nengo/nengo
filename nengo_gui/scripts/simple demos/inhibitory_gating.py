# # Nengo Example: Inhibitory Gating of Ensembles

# ## Step 1: Create the network
#
# Our model consists of two ensembles (called A and B) that receive inputs from
# a common sine wave signal generator.
#
# Ensemble A is gated using the output of a node, while Ensemble B is gated
# using the output of a third ensemble (C). This is to demonstrate that
# ensembles can be gated using either node outputs, or decoded outputs from
# ensembles.

import nengo
import numpy as np
from nengo.utils.functions import piecewise

model = nengo.Network(label="Inhibitory Gating")

n_neurons = 30

with model:
    A = nengo.Ensemble(n_neurons, dimensions=1, label="A")
    B = nengo.Ensemble(n_neurons, dimensions=1, label="B")
    C = nengo.Ensemble(n_neurons, dimensions=1, label="C")

    sin = nengo.Node(np.sin, label="sin")
    inhib = nengo.Node(
        piecewise({0: 0, 2.5: 1, 5: 0, 7.5: 1, 10: 0, 12.5: 1}),
        label="inhibition")

    nengo.Connection(sin, A)
    nengo.Connection(sin, B)
    nengo.Connection(inhib, A.neurons, transform=[[-2.5]] * n_neurons)
    nengo.Connection(inhib, C)
    nengo.Connection(C, B.neurons, transform=[[-2.5]] * n_neurons)

    sin_probe = nengo.Probe(sin)
    inhib_probe = nengo.Probe(inhib)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    C_probe = nengo.Probe(C, synapse=0.01)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.7442636788443602
gui[model].offset = 212.12253790762998,24.930856767521504
gui[A].pos = 175.000, 50.000
gui[A].scale = 1.000
gui[B].pos = 175.000, 125.000
gui[B].scale = 1.000
gui[C].pos = 143.468, 186.814
gui[C].scale = 1.000
gui[sin].pos = 16.175, 54.248
gui[sin].scale = 1.000
gui[inhib].pos = 55.160, 198.618
gui[inhib].scale = 1.000
