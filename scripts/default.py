import nengo

model = nengo.Network(label="My Network")
with model:
    a = nengo.Ensemble(n_neurons=10, dimensions=2, label="My Ensemble")

    nengo.Connection(a, a)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.0
gui[model].offset = 0.0,0.0
gui[a].pos = 300.0, 200.0
gui[a].scale = 1.000
