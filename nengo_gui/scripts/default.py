import nengo

model = nengo.Network()
with model:

	c = nengo.Ensemble(n_neurons=10, dimensions=1)

	v1 = nengo.Network()
	with v1:
	    b = nengo.Ensemble(n_neurons=10, dimensions=1)
