import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")
    c = nengo.Ensemble(n_neurons=80, dimensions=2, label="less")

    vis = nengo.Network(label="tester")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)
        
        subvis = nengo.Network(label="subnet")
        with subvis:
             d = nengo.Ensemble(n_neurons=80, dimensions=2, label="d_ens")

    nengo.Connection(b,b)
    nengo.Connection(a,a)
    nengo.Connection(a,b)
    nengo.Connection(c,a)
    nengo.Connection(c,d)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.7463891916484794
gui[model].offset = 396.40723827073657,331.7964994548337
gui[a].pos = -96.725, -169.064
gui[a].scale = 1.000
gui[c].pos = -309.267, -200.255
gui[c].scale = 1.000
gui[b].pos = 294.646, 61.642
gui[b].scale = 1.803
gui[d].pos = -122.924, 110.167
gui[d].scale = 1.884
