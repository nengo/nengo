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
gui[model].scale = 0.7463891916484794
gui[model].offset = -56.59276172926343,-64.20350054516632
gui[a].pos = 556.130, 335.834
gui[a].scale = 1.000
gui[c].pos = 343.588, 304.643
gui[c].scale = 1.000
gui[b].pos = 636.355, 488.398
gui[b].scale = 1.650
