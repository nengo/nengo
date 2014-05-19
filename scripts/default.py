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
gui[model].scale = 1.2834258975629058
gui[model].offset = -186.53468160342118,-74.91992289607458
gui[a].pos = 376.000, 157.000
gui[c].pos = 264.916, 126.225
gui[b].pos = 327.283, 230.583
