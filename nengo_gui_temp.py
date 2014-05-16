import nengo

model = nengo.Network()
with model:
	a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")

	vis = nengo.Network(label="tester")
	with vis:
		b = nengo.Ensemble(n_neurons=80, dimensions=2)

		v1 = nengo.Network()
		with v1:
		    c = nengo.Ensemble(n_neurons=80, dimensions=2)

	v2 = nengo.Network(label="tester")
	with v2:
		d = nengo.Ensemble(n_neurons=80, dimensions=2)
		e = nengo.Ensemble(n_neurons=80, dimensions=2)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9986146661010299
gui[model].offset = 0.5457427618872543,0.24795939341674966
gui[a].pos = 200.000, 94.000
gui[b].pos = 113.875, 227.280
gui[c].pos = 300.614, 227.072
gui[d].pos = 313.046, 344.705
gui[e].pos = 115.772, 344.706

