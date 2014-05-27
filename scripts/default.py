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
gui[model].scale = 1.0027764359010838
gui[model].offset = 267.87923007337326,154.09134847188193
gui[a].pos = -96.725, -169.064
gui[a].scale = 1.000
gui[c].pos = -309.267, -200.255
gui[c].scale = 1.000
gui[b].pos = 231.236, 82.501
gui[b].scale = 1.038
gui[d].pos = -9.261, 110.449
gui[d].scale = 1.085
