import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")
    c = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")

    vis = nengo.Network(label="tester")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)

    nengo.Connection(b,b)
    nengo.Connection(a,a)
    nengo.Connection(a,b)
    nengo.Connection(c,a)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.7695036100804341
gui[model].offset = -73.39594735024002,-76.28744694489154
gui[a].pos = 556.130, 335.834
gui[a].scale = 1.000
gui[c].pos = 738.648, 320.237
gui[c].scale = 1.000
gui[b].pos = 370.773, 401.369
gui[b].scale = 2.156
