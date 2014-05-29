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
gui[model].scale = 0.7867325516300206
gui[model].offset = 298.9476776123047,332.0541534423828
gui[a].pos = -141.739, -323.147
gui[a].scale = 1.000
gui[c].pos = -321.373, -322.427
gui[c].scale = 1.000
gui[bag].pos = -1.960, -146.623
gui[bag].scale = 0.942
gui[d].pos = -221.478, -119.922
gui[d].scale = 0.697
