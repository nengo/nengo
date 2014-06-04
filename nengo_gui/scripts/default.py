import nengo

model = nengo.Network()
with model:

    c = nengo.Ensemble(n_neurons=10, dimensions=1)

    v1 = nengo.Network()
    with v1:
        b = nengo.Ensemble(n_neurons=10, dimensions=1)
        b = nengo.Ensemble(n_neurons=10, dimensions=1)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1
gui[model].offset = 127,85
gui[c].pos = -16.760, -127.002
gui[c].scale = 1.000
gui[v1].pos = 197.427, 138.230
gui[v1].scale = 1.000
gui[v1].size = 304.854, 113.541
gui[v1.ensembles[0]].pos = 309.854, 121.459
gui[v1.ensembles[0]].scale = 1.000
gui[b].pos = 85.000, 155.000
gui[b].scale = 1.000
