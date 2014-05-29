import nengo

import numpy as np


model = nengo.Network(label='Visualization Test')
with model:
    input = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label='input')

    a = nengo.Ensemble(100, dimensions=2, label='A')
    b = nengo.Ensemble(100, dimensions=1, label='B')
    nengo.Connection(input, a, synapse=None)
    nengo.Connection(a, b, synapse=0.01, function=lambda x: x[0]*x[1])

    def printout(t, x):
        print t, x
    output = nengo.Node(printout, size_in=1)
    nengo.Connection(b, output, synapse=0.01)



import nengo_gui.javaviz
nengo_gui.javaviz.View(model)

sim = nengo.Simulator(model)
sim.run(100000)


