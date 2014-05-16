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
gui[model].scale = 0.46716367344159154
gui[model].offset = 143.27502350406473,105.83837725652398
gui[a].pos = 641.214, 715.919
gui[b].pos = 128.779, 325.756
gui[c].pos = -126.534, 302.431
gui[d].pos = 58.743, 109.789
gui[e.ea_ensembles[0]].pos = 896.185, 397.857
gui[e.ea_ensembles[1]].pos = 953.408, 377.013
gui[e.input].pos = 961.590, 293.411
gui[e.output].pos = 884.644, 258.091
gui[r].pos = 39.790, 1042.660
gui[t].pos = 744.572, 818.879
gui[f].pos = 532.513, 885.870
gui[g].pos = 285.001, 912.920
