import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")
    c = nengo.Ensemble(n_neurons=80, dimensions=2, label="less")

    vis = nengo.Network(label="tester")
    with vis:
        bag = nengo.Ensemble(n_neurons=80, dimensions=2)
        
        subvis = nengo.Network(label="subnet")
        with subvis:
             d = nengo.Ensemble(n_neurons=80, dimensions=2, label="d_ens")

    nengo.Connection(bag,bag)
    nengo.Connection(a,a)
    nengo.Connection(a,bag)
    nengo.Connection(c,a)
    nengo.Connection(c,d)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.0027764359010838
gui[model].offset = 536.8792300733733,411.09134847188193
gui[a].pos = -141.739, -323.147
gui[a].scale = 1.000
gui[c].pos = -321.373, -322.427
gui[c].scale = 1.000
gui[bag].pos = -6.565, -157.136
gui[bag].scale = 0.942
gui[d].pos = -226.083, -130.435
gui[d].scale = 0.697
