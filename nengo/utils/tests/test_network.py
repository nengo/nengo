import nengo


def test_withself():
    model = nengo.Network(label='test_withself')
    with model:
        n1 = nengo.Node(0.5)
        assert n1 in model.nodes
        e1 = nengo.Ensemble(10, dimensions=1)
        assert e1 in model.ensembles
        c1 = nengo.Connection(n1, e1)
        assert c1 in model.connections
        ea1 = nengo.networks.EnsembleArray(10, n_ensembles=2)
        assert ea1 in model.networks
        assert len(ea1.ensembles) == 2
        n2 = ea1.add_output("out", None)
        assert n2 in ea1.nodes
        with ea1:
            e2 = nengo.Ensemble(10, dimensions=1)
            assert e2 in ea1.ensembles
    assert len(nengo.Network.context) == 0
