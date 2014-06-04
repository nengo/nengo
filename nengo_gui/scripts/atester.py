import nengo

model = nengo.Network()
with model:
	a = nengo.Ensemble(n_neurons=80, dimensions=2, label="testasdfasdfasdfasfd")


