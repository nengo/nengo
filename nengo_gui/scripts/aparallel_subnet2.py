import nengo

model = nengo.Network()
with model:

    vis = nengo.Network(label="1")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)

        v1 = nengo.Network(label="2")
        with v1:
            d = nengo.Ensemble(n_neurons=80, dimensions=2)

        v2 = nengo.Network(label="3")
        with v2:
            f = nengo.Ensemble(n_neurons=80, dimensions=2)

    vis = nengo.Network(label="4")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)

        v1 = nengo.Network(label="5")
        with v1:
            d = nengo.Ensemble(n_neurons=80, dimensions=2)

        v2 = nengo.Network(label="6")
        with v2:
            f = nengo.Ensemble(n_neurons=80, dimensions=2)
