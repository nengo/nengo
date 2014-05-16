import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")

    vis = nengo.Network(label="tester")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)


    nengo.Connection(b,b)
    nengo.Connection(a,a)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1
gui[model].offset = 0,0
gui[a].pos = 200.000, 94.000
gui[b].pos = 113.875, 227.280

