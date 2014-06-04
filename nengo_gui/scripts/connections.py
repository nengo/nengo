import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=100, dimensions=2, label='int')
    b = nengo.Ensemble(n_neurons=80, dimensions=2, label='wait')
    c = nengo.Ensemble(n_neurons=80, dimensions=2)
    d = nengo.Ensemble(n_neurons=80, dimensions=2)
    e = nengo.networks.EnsembleArray(80,2, label="array")

    vis = nengo.Network(label="vision")
    with vis:

		v1 = nengo.Network(label="primary")
		with v1:
			f = nengo.Ensemble(n_neurons=80, dimensions=2)
			g = nengo.Ensemble(n_neurons=80, dimensions=2)
			nengo.Connection(f, g)
			nengo.Connection(f, f)
			
		r = nengo.Ensemble(n_neurons=80, dimensions=2)
		t = nengo.Ensemble(n_neurons=80, dimensions=2)
		nengo.Connection(r, t)

    nengo.Connection(a, a, synapse = 0.01)
    nengo.Connection(a, b, synapse=0.01)
    nengo.Connection(b, c, synapse=0.01)
    nengo.Connection(c, d, synapse=0.01)
    nengo.Connection(c, r, synapse=0.01)
    nengo.Connection(b, d, synapse=0.01)
    nengo.Connection(d, e.input[:2], synapse=0.01)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.6870369268888727
gui[model].offset = -2973.832579070175,-2662.104663232688
gui[a].pos = 4786.724, 4517.055
gui[a].scale = 1.000
gui[b].pos = 4752.862, 4338.535
gui[b].scale = 1.000
gui[c].pos = 5017.618, 4405.442
gui[c].scale = 1.000
gui[d].pos = 5103.881, 4278.200
gui[d].scale = 1.000
gui[e].pos = 4543.174, 4032.532
gui[e].scale = 1.000
gui[e].size = 156.946, 219.766
gui[e.ea_ensembles[0]].pos = 4516.242, 4102.415
gui[e.ea_ensembles[0]].scale = 1.000
gui[e.ea_ensembles[1]].pos = 4573.465, 4081.571
gui[e.ea_ensembles[1]].scale = 1.000
gui[e.input].pos = 4581.647, 3997.969
gui[e.input].scale = 1.000
gui[e.output].pos = 4504.701, 3962.649
gui[e.output].scale = 1.000
gui[vis].pos = 5226.591, 4609.252
gui[vis].scale = 1.000
gui[vis].size = 407.844, 312.937
gui[r].pos = 5062.669, 4725.720
gui[r].scale = 1.000
gui[t].pos = 5380.724, 4492.783
gui[t].scale = 1.000
gui[v1].pos = 5226.757, 4618.358
gui[v1].scale = 1.000
gui[v1].size = 327.512, 107.050
gui[f].pos = 5350.513, 4604.833
gui[f].scale = 1.000
gui[g].pos = 5103.001, 4631.883
gui[g].scale = 1.000
