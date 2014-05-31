import nengo

model = nengo.Network(label="My Network")
with model:
    input = nengo.Node([0, 0])

    a = nengo.Ensemble(100, dimensions=2, label="My Ensemble")
    nengo.Probe(a)

    nengo.Connection(input, a)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.0
gui[model].offset = 0.0,0.0
gui[a].pos = 300.0, 200.0
gui[a].scale = 1.000
gui[input].pos = 100.0, 200.0
gui[input].scale = 1.000
