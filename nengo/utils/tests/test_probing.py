import nengo
import pytest
from nengo.utils.probing import probe


@pytest.mark.parametrize("recursive", [False, True])
def test_probing(recursive):

    model = nengo.Network(label='_test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        conn = nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node2 = nengo.Node(output=[0])

    object_probe_dict = probe(model, recursive=recursive)

    # test top level probing
    total_number_probes1 = len(
        ens1.probeable) + len(node1.probeable) + len(conn.probeable)
    assert(len(model.probes) == total_number_probes1)

    # test dictionary
    assert(len(object_probe_dict[ens1]) == len(ens1.probeable))
    assert(len(object_probe_dict[node1]) == len(node1.probeable))
    assert(len(object_probe_dict[conn]) == len(conn.probeable))

    # test recursive probing
    if recursive:
        total_number_probes2 = len(ens2.probeable) + len(node2.probeable)
        assert(len(subnet.probes) == total_number_probes2)
        assert(len(object_probe_dict[ens2]) == len(ens2.probeable))
        assert(len(object_probe_dict[node2]) == len(node2.probeable))
    else:
        assert(len(subnet.probes) == 0)
        assert ens2 not in object_probe_dict


def test_probe_options():
    model = nengo.Network(label='_test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        conn = nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node2 = nengo.Node(output=[0])

    probe_options = {nengo.Ensemble: ['decoded_output', 'spikes']}
    object_probe_dict = probe(
        model,
        recursive=True,
        probe_options=probe_options)

    # only probes spikes and decoded output of the ensembles
    assert(len(model.probes) == 2)
    assert(len(subnet.probes) == 2)

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
