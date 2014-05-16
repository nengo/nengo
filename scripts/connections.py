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
gui[model].scale = 0.7856725166675906
gui[model].offset = -416.625720600102,-452.458217922657
gui[a].pos = 231.023, 726.690
gui[b].pos = 469.128, 723.581
gui[c].pos = 1066.681, 897.112
gui[d].pos = 820.147, 663.246
gui[e.ea_ensembles[0]].pos = 1193.245, 782.520
gui[e.ea_ensembles[1]].pos = 1250.468, 761.676
gui[e.input].pos = 1258.650, 678.074
gui[e.output].pos = 1181.704, 642.754
gui[r].pos = 724.288, 1237.965
gui[t].pos = 1343.277, 1071.008
gui[f].pos = 1131.218, 1137.999
gui[g].pos = 883.706, 1165.049
