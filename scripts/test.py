import nengo

model = nengo.Network(label="My Network")
with model:
    a = nengo.Ensemble(n_neurons=10, dimensions=2, label="A")
    b = nengo.Ensemble(n_neurons=10, dimensions=2, label="B")

    nengo.Connection(a, a)
    nengo.Connection(a, b)

    sub = nengo.Network(label="My Network")
    with sub:
        c = nengo.Ensemble(n_neurons=10, dimensions=2, label="B")


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1
gui[model].offset = 0,0
