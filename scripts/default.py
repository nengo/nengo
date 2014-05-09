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


import nengo_gui
gui = nengo_gui.Config()
gui[a].pos = 200, 100
gui[b].pos = 100, 200
gui[c].pos = 300, 200
