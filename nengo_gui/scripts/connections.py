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
gui[model].scale = 0.6213981623131563
gui[model].offset = -2743.9397298506215,-2327.5847593250305
gui[a].pos = 4514.757, 4341.644
gui[a].scale = 1.000
gui[b].pos = 4752.862, 4338.535
gui[b].scale = 1.000
gui[c].pos = 5017.618, 4405.442
gui[c].scale = 1.000
gui[d].pos = 5103.881, 4278.200
gui[d].scale = 1.000
gui[e].pos = 5503.911, 4182.195
gui[e].scale = 1.000
gui[e].size = 156.946, 219.766
gui[e.ea_ensembles[0]].pos = 5476.979, 4252.078
gui[e.ea_ensembles[0]].scale = 1.000
gui[e.ea_ensembles[1]].pos = 5534.202, 4231.234
gui[e.ea_ensembles[1]].scale = 1.000
gui[e.input].pos = 5542.384, 4147.632
gui[e.input].scale = 1.000
gui[e.output].pos = 5465.438, 4112.312
gui[e.output].scale = 1.000
gui[vis].pos = 5554.011, 4602.009
gui[vis].scale = 1.000
gui[vis].size = 698.989, 246.957
gui[r].pos = 5244.517, 4685.488
gui[r].scale = 1.000
gui[t].pos = 5863.506, 4518.531
gui[t].scale = 1.000
gui[v1].pos = 5527.691, 4599.047
gui[v1].scale = 1.000
gui[v1].size = 327.512, 107.050
gui[f].pos = 5651.447, 4585.522
gui[f].scale = 1.000
gui[g].pos = 5403.935, 4612.572
gui[g].scale = 1.000
