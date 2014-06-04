import nengo

model = nengo.Network()
with model:

    vis = nengo.Network(label="1")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)
        b = nengo.Ensemble(n_neurons=80, dimensions=2)


