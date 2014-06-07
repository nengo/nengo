# # Nengo Network: Ensemble Array
#
# An ensemble array is a group of ensembles that each represent a part of the
# overall signal.
#
# Ensemble arrays are similar to normal ensembles, but expose a slightly
# different interface. Additionally, in an ensemble array, the components of
# the overall signal are not related. As a result, network arrays cannot be
# used to compute nonlinear functions that mix the dimensions they represent.

import nengo
import numpy as np

model = nengo.Network(label='Ensemble Array')
with model:
    # Make an input node
    sin = nengo.Node(output=lambda t: [np.cos(t), np.sin(t)], label="sin")

    # Make ensembles to connect
    A = nengo.networks.EnsembleArray(100, n_ensembles=2, label="A")
    B = nengo.Ensemble(100, dimensions=2, label="B")
    C = nengo.networks.EnsembleArray(100, n_ensembles=2, label="C")

    # Connect the model elements, just feedforward
    nengo.Connection(sin, A.input)
    nengo.Connection(A.output, B)
    nengo.Connection(B, C.input)

    # Setup the probes for plotting
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A.output, synapse=0.02)
    B_probe = nengo.Probe(B, synapse=0.02)
    C_probe = nengo.Probe(C.output, synapse=0.02)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9215625000000003
gui[model].offset = 67.58124999999984,131.2675634765625
gui[B].pos = 496.914, 47.999
gui[B].scale = 1.000
gui[sin].pos = 0.000, 50.000
gui[sin].scale = 1.000
gui[A].pos = 202.438, 50.000
gui[A].scale = 1.000
gui[A].size = 302.875, 212.000
gui[A.ea_ensembles[0]].pos = 200.000, 0.000
gui[A.ea_ensembles[0]].scale = 1.000
gui[A.ea_ensembles[1]].pos = 200.000, 100.000
gui[A.ea_ensembles[1]].scale = 1.000
gui[A.input].pos = 100.000, 50.000
gui[A.input].scale = 1.000
gui[A.output].pos = 300.000, 50.000
gui[A.output].scale = 1.000
gui[C].pos = 773.139, 45.829
gui[C].scale = 1.000
gui[C].size = 302.875, 212.000
gui[C.ea_ensembles[0]].pos = 770.702, -4.171
gui[C.ea_ensembles[0]].scale = 1.000
gui[C.ea_ensembles[1]].pos = 770.702, 95.829
gui[C.ea_ensembles[1]].scale = 1.000
gui[C.input].pos = 670.702, 45.829
gui[C.input].scale = 1.000
gui[C.output].pos = 870.702, 45.829
gui[C.output].scale = 1.000
