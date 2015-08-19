import nengo
from nengo.utils.network import config_with_default_synapse

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

def test_synapseconfig():

    def synapse_net(syn_conf=None):
        model = nengo.Network(label="synapse_config")
        with model:
            syn_conf, override = config_with_default_synapse(
                syn_conf, nengo.Lowpass(0.1))
            with syn_conf:
                tmp_a = nengo.Node(1)
                tmp_b = nengo.Node(size_in=1)
                model.conn = nengo.Connection(tmp_a, tmp_b)

            if override:
                del syn_conf[nengo.Connection].synapse
        return model

    # no config should use the default config
    net = synapse_net()
    assert net.conn.synapse.tau == 0.1
    # use a passed config
    conf = nengo.Config(nengo.Connection)
    conf[nengo.Connection].synapse = nengo.Lowpass(0.01)
    net = synapse_net(conf)
    assert net.conn.synapse.tau == 0.01
