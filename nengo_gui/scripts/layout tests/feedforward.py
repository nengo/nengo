import nengo

model = nengo.Network(label="Feedforward")
with model:
    input = nengo.Node([0, 0])

    a = nengo.Ensemble(100, dimensions=2, label="a")
    b = nengo.Ensemble(100, dimensions=2, label="b")
    c = nengo.Ensemble(100, dimensions=2, label="c")
    j = nengo.Ensemble(100, dimensions=2, label="j")

    subnet = nengo.Network(label="subnet")
    with subnet:
        d = nengo.Ensemble(100, dimensions=2, label="d")
        f = nengo.Ensemble(100, dimensions=2, label="f")
        g = nengo.Ensemble(100, dimensions=2, label="g")
        h = nengo.Ensemble(100, dimensions=2, label="h")
        z = nengo.Ensemble(100, dimensions=2, label="z")

    subnet2 = nengo.Network(label="subnet1")
    with subnet2:
        d0 = nengo.Ensemble(100, dimensions=2, label="d0")
        d1 = nengo.Ensemble(100, dimensions=2, label="d1")
        f1 = nengo.Ensemble(100, dimensions=2, label="f1")
        g1 = nengo.Ensemble(100, dimensions=2, label="g1")
        h1 = nengo.Ensemble(100, dimensions=2, label="h1")

    e = nengo.Ensemble(100, dimensions=2, label="e")
    i = nengo.Ensemble(100, dimensions=2, label="i")

    nengo.Connection(input, a)
    nengo.Connection(a, b)
    nengo.Connection(b, c)
    nengo.Connection(b, d)
    nengo.Connection(c, e)
    nengo.Connection(e, e)
    nengo.Connection(j, i)
    nengo.Connection(b, j)

    nengo.Connection(d, f)
    nengo.Connection(d, g)
    nengo.Connection(f, h)
    nengo.Connection(g, h)
    nengo.Connection(h, z)
    nengo.Connection(h, i)
    nengo.Connection(g, i)
    nengo.Connection(d, e)

    nengo.Connection(d0, d1)
    nengo.Connection(d1, f1)
    nengo.Connection(d1, g1)
    nengo.Connection(f1, h1)
    nengo.Connection(g1, h1)
    nengo.Connection(h1, i)
    nengo.Connection(g1, i)
    nengo.Connection(d1, e)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9027551020408164
gui[model].offset = 85.2602040816326,138.14858269671998
gui[a].pos = 400.000, 50.000
gui[a].scale = 1.000
gui[b].pos = 500.000, 150.000
gui[b].scale = 1.000
gui[c].pos = 750.000, 0.000
gui[c].scale = 1.000
gui[j].pos = 750.000, 100.000
gui[j].scale = 1.000
gui[e].pos = 400.000, 150.000
gui[e].scale = 1.000
gui[i].pos = 400.000, 250.000
gui[i].scale = 1.000
gui[input].pos = 150.000, 50.000
gui[input].scale = 1.000
gui[subnet].pos = 750.000, 250.000
gui[subnet].scale = 1.000
gui[subnet].size = 380.000, 212.000
gui[d].pos = 600.000, 250.000
gui[d].scale = 1.000
gui[f].pos = 700.000, 200.000
gui[f].scale = 1.000
gui[g].pos = 700.000, 300.000
gui[g].scale = 1.000
gui[h].pos = 800.000, 250.000
gui[h].scale = 1.000
gui[z].pos = 900.000, 250.000
gui[z].scale = 1.000
gui[subnet2].pos = 150.000, 200.000
gui[subnet2].scale = 1.000
gui[subnet2].size = 380.000, 212.000
gui[d0].pos = 0.000, 200.000
gui[d0].scale = 1.000
gui[d1].pos = 100.000, 200.000
gui[d1].scale = 1.000
gui[f1].pos = 200.000, 150.000
gui[f1].scale = 1.000
gui[g1].pos = 200.000, 250.000
gui[g1].scale = 1.000
gui[h1].pos = 300.000, 200.000
gui[h1].scale = 1.000
