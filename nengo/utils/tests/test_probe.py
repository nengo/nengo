import nengo
import pytest
from nengo.utils.probe import probe_all


@pytest.mark.parametrize("recursive", [False, True])
def test_probe_all_recursive(recursive):

    model = nengo.Network(label='test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        conn = nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node2 = nengo.Node(output=[0])

    probes = probe_all(model, recursive=recursive)

    # test top level probing
    total_number_probes1 = len(
        ens1.probeable) + len(node1.probeable) + len(conn.probeable)
    assert(len(model.probes) == total_number_probes1)

    # test dictionary
    assert(len(probes[ens1]) == len(ens1.probeable))
    assert(len(probes[node1]) == len(node1.probeable))
    assert(len(probes[conn]) == len(conn.probeable))

    # test recursive probing
    if recursive:
        total_number_probes2 = len(ens2.probeable) + len(node2.probeable)
        assert(len(subnet.probes) == total_number_probes2)
        assert(len(probes[ens2]) == len(ens2.probeable))
        assert(len(probes[node2]) == len(node2.probeable))
    else:
        assert(len(subnet.probes) == 0)
        assert ens2 not in probes


def test_probe_all_options():
    model = nengo.Network(label='test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            nengo.Ensemble(n_neurons=1, dimensions=1)
            nengo.Node(output=[0])

    probe_all(model, recursive=True, probe_options={
        nengo.Ensemble: ['decoded_output', 'spikes']})

    # only probes spikes and decoded output of the ensembles
    assert(len(model.probes) == 2)
    assert(len(subnet.probes) == 2)


def test_probe_all_kwargs():
    model = nengo.Network(label='test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            nengo.Ensemble(n_neurons=1, dimensions=1)
            nengo.Node(output=[0])

    probe_all(model, recursive=True, sample_every=0.1, seed=10)
    for probe in model.probes + subnet.probes:
        assert probe.sample_every == 0.1
        assert probe.seed == 10

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
