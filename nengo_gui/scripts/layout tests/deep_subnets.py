import nengo

model = nengo.Network()
with model:
    v0 = nengo.Network()
    z = nengo.Ensemble(n_neurons=80, dimensions=2)
    with v0:
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
                    v3 = nengo.Network()
                    with v3:
                        e = nengo.Ensemble(n_neurons=80, dimensions=2)

    nengo.Connection(a, a)
    nengo.Connection(a, b)
    nengo.Connection(a, c)
    nengo.Connection(a, d)
    nengo.Connection(a, d)
    nengo.Connection(a, e)
    nengo.Connection(z, e)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.37727622068188804
gui[model].offset = 579.6424203783863,352.91540377423536
gui[z].pos = 1208.639, 869.067
gui[z].scale = 1.000
gui[v0].pos = -438.138, 162.843
gui[v0].scale = 1.000
gui[v0].size = 1905.730, 1074.108
gui[a].pos = -1351.003, 659.897
gui[a].scale = 1.000
gui[vis].pos = -327.642, 154.250
gui[vis].scale = 1.000
gui[vis].size = 1604.738, 976.922
gui[b].pos = -1090.011, 602.711
gui[b].scale = 1.000
gui[v1].pos = 43.386, -49.621
gui[v1].scale = 1.000
gui[v1].size = 782.681, 489.181
gui[c].pos = 394.727, -254.211
gui[c].scale = 1.000
gui[v2].pos = -113.977, 37.485
gui[v2].scale = 1.000
gui[v2].size = 387.954, 234.970
gui[d].pos = -267.954, 114.970
gui[d].scale = 1.000
gui[v3].pos = 0.000, 0.000
gui[v3].scale = 1.000
gui[v3].size = 80.000, 80.000
gui[e].pos = 0.000, 0.000
gui[e].scale = 1.000
