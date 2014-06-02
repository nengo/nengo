import nengo

model = nengo.Network()
with model:
	b = nengo.Ensemble(n_neurons=80, dimensions=2)

	vis = nengo.Network()
	with vis:
		a = nengo.Ensemble(n_neurons=80, dimensions=2)

	nengo.Connection(b, a, synapse=0.01)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.6162295514775604
gui[model].offset = 397.9893680765997,232.48144533734717
gui[b].pos = -29.762, -40.340
gui[b].scale = 1.000
gui[vis].pos = 114.599, -37.881
gui[vis].scale = 1.000
gui[vis].size = 80.000, 80.000
gui[a].pos = 114.599, -37.881
gui[a].scale = 1.000
