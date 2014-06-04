import nengo
import numpy as np
from nengo.networks import CircularConvolution, EnsembleArray

model = nengo.Network(label="Neural Extraction")

D = 4
num_items = 10

threshold_func = lambda x: 1.0 if x > 0.3 else 0.0

with model:
    extract = nengo.Network(label="Extract")
    with extract:
        a = EnsembleArray(n_neurons=80, n_ensembles=D)
        b = EnsembleArray(n_neurons=80, n_ensembles=D)
        c = CircularConvolution(n_neurons=80, dimensions=D)
        d = nengo.Ensemble(n_neurons=80, dimensions=D)

        nengo.Connection(a.output, c.A)
        nengo.Connection(b.output, c.B)
        nengo.Connection(c.output, d)

    assoc = nengo.Network(label="Associate")
    assoc_nodes = []
    with assoc:
        assoc_input = nengo.Node(size_in=D)
        assoc_output = nengo.Node(size_in=D)

        for item in range(num_items):
            assoc_nodes.append(nengo.Ensemble(n_neurons=20, dimensions=1))
            nengo.Connection(
                assoc_input, assoc_nodes[-1], transform=np.ones((1, D)))
            nengo.Connection(
                assoc_nodes[-1], assoc_output, transform=np.ones((D, 1)),
                function=threshold_func)

    nengo.Connection(d, assoc_input)

    output = nengo.Network(label="output")
    with output:
        out = nengo.Ensemble(n_neurons=80, dimensions=D)

    nengo.Connection(assoc_output, out)
