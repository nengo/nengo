import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="aba")

    vis = nengo.Network(label="vis")
    with vis:
        b = nengo.Ensemble(n_neurons=80, dimensions=2, label="bab")

        v1 = nengo.Network(label="v1")
        with v1:
            c = nengo.Ensemble(n_neurons=80, dimensions=2, label="c")

        v2 = nengo.Network(label="v2")
        with v2:
            d = nengo.Ensemble(n_neurons=80, dimensions=2, label="d")
            e = nengo.Ensemble(n_neurons=80, dimensions=2, label="e")

            v4 = nengo.Network(label="v4")
            with v4:
                h = nengo.Ensemble(n_neurons=80, dimensions=2, label="h")
                i = nengo.Ensemble(n_neurons=80, dimensions=2, label="i")

    v3 = nengo.Network(label="v3")
    with v3:
        f = nengo.Ensemble(n_neurons=80, dimensions=2, label="f")
        g = nengo.Ensemble(n_neurons=80, dimensions=2, label="g")

    nengo.Connection(a,b)
    nengo.Connection(h,h)
    nengo.Connection(f,g)
    nengo.Connection(i,g)
    nengo.Connection(d,e)
    nengo.Connection(b,c)
    nengo.Connection(c,c)
    nengo.Connection(c,d)