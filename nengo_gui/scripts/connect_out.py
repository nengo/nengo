import nengo

model = nengo.Network()
with model:
	b = nengo.Ensemble(n_neurons=80, dimensions=2)

	vis = nengo.Network()
	with vis:
		a = nengo.Ensemble(n_neurons=80, dimensions=2)

	nengo.Connection(b, a, synapse=0.01)