import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2)

    vis = nengo.Network()
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2)

        v1 = nengo.Network()
        with v1:
            c = nengo.Ensemble(n_neurons=80, dimensions=2)

        v2 = nengo.Network()
        with v2:
            d = nengo.Ensemble(n_neurons=80, dimensions=2)
            e = nengo.Ensemble(n_neurons=80, dimensions=2)

            v4 = nengo.Network()
            with v4:
                h = nengo.Ensemble(n_neurons=80, dimensions=2)
                i = nengo.Ensemble(n_neurons=80, dimensions=2)

    v3 = nengo.Network()
    with v3:
        f = nengo.Ensemble(n_neurons=80, dimensions=2)
        g = nengo.Ensemble(n_neurons=80, dimensions=2)

