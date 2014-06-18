import pytest

import nengo


def test_basic_context(Simulator):
    # We must give the two Networks different labels because object comparison
    # is done using identifiers that stem from the top-level Network label.
    model1 = nengo.Network(label="test1")
    model2 = nengo.Network(label="test2")

    with model1:
        e = nengo.Ensemble(1, dimensions=1)
        n = nengo.Node([0])
        assert e in model1.ensembles
        assert n in model1.nodes

        con = nengo.Network()
        with con:
            e2 = nengo.Ensemble(1, dimensions=1)
        assert e2 in con.ensembles
        assert e2 not in model1.ensembles

        e3 = nengo.Ensemble(1, dimensions=1)
        assert e3 in model1.ensembles

    with model2:
        e4 = nengo.Ensemble(1, dimensions=1)
        assert e4 not in model1.ensembles
        assert e4 in model2.ensembles


def test_nested_context(Simulator):
    model = nengo.Network()
    with model:
        con1 = nengo.Network()
        con2 = nengo.Network()
        con3 = nengo.Network()

        with con1:
            e1 = nengo.Ensemble(1, dimensions=1)
            assert e1 in con1.ensembles

            with con2:
                e2 = nengo.Ensemble(1, dimensions=1)
                assert e2 in con2.ensembles
                assert e2 not in con1.ensembles

                with con3:
                    e3 = nengo.Ensemble(1, dimensions=1)
                    assert e3 in con3.ensembles
                    assert e3 not in con2.ensembles \
                        and e3 not in con1.ensembles

                e4 = nengo.Ensemble(1, dimensions=1)
                assert e4 in con2.ensembles
                assert e4 not in con3.ensembles

            e5 = nengo.Ensemble(1, dimensions=1)
            assert e5 in con1.ensembles

        e6 = nengo.Ensemble(1, dimensions=1)
        assert e6 not in con1.ensembles


def test_context_errors(Simulator):
    def add_something():
        nengo.Ensemble(1, dimensions=1)

    # Error if adding before Network creation
    with pytest.raises(RuntimeError):
        add_something()

    model = nengo.Network()
    # Error if adding before a `with network` block
    with pytest.raises(RuntimeError):
        add_something()

    # Error if adding after a `with network` block
    with model:
        add_something()
    with pytest.raises(RuntimeError):
        add_something()

    # Okay if add_to_container=False
    nengo.Ensemble(1, dimensions=1, add_to_container=False)
    nengo.Node(output=[0], add_to_container=False)


def test_get_objects(Simulator):
    model = nengo.Network()
    with model:
        ens1 = nengo.Ensemble(10, 1)
        node1 = nengo.Node([0])
        conn1 = nengo.Connection(node1, ens1)
        pr1 = nengo.Probe(ens1)

        subnet = nengo.Network()

        with subnet:
            ens2 = nengo.Ensemble(10, 1)
            node2 = nengo.Node([0])
            conn2 = nengo.Connection(node2, ens2)
            pr2 = nengo.Probe(ens2)

    all_objects = [ens1, pr1, node1, conn1,
                   subnet, ens2, node2, conn2,
                   pr2]

    # can't use == for lists because order matters
    def is_equal(list1, list2):
        if set(list1) == set(list2) and len(list1) == len(list2):
            return True
        else:
            return False

    assert is_equal(all_objects, model.all_objects)
    assert is_equal([ens1, ens2], model.all_ensembles)
    assert is_equal([node1, node2], model.all_nodes)
    assert is_equal([conn1, conn2], model.all_connections)
    assert is_equal([pr1, pr2], model.all_probes)
    assert is_equal([subnet], model.all_networks)

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
