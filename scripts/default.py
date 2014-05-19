import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")

    vis = nengo.Network(label="tester")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)

    nengo.Connection(b,b)
    nengo.Connection(a,a)
    nengo.Connection(a,b)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9619274546647756
gui[model].offset = 16.561557220822635,7.728726703050569
gui[a].pos = 376.000, 157.000
gui[b].pos = 320.646, 226.910
