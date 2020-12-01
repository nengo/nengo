from collections import Counter

import pytest

import nengo
from nengo.exceptions import ConfigError, NetworkContextError, ReadonlyError
from nengo.utils.testing import ThreadedAssertion


def test_basic_context():
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


def test_nested_context():
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
                    assert e3 not in con2.ensembles and e3 not in con1.ensembles

                e4 = nengo.Ensemble(1, dimensions=1)
                assert e4 in con2.ensembles
                assert e4 not in con3.ensembles

            e5 = nengo.Ensemble(1, dimensions=1)
            assert e5 in con1.ensembles

        e6 = nengo.Ensemble(1, dimensions=1)
        assert e6 not in con1.ensembles


def test_context_errors(request):
    """Most of these errors are in ``Network.add``; one is in ``Network.__exit__``"""

    def clear_contexts():  # clear anything we've added to the contexts
        nengo.Network.context.clear()
        nengo.Config.context.clear()

    request.addfinalizer(clear_contexts)

    def add_something():
        nengo.Ensemble(1, dimensions=1)

    # Error if adding before Network creation
    with pytest.raises(NetworkContextError, match="must either be created inside a"):
        add_something()

    model = nengo.Network()
    # Error if adding before a `with network` block
    with pytest.raises(NetworkContextError, match="must either be created inside a"):
        add_something()

    # Error if adding after a `with network` block
    with model:
        add_something()
    with pytest.raises(NetworkContextError, match="must either be created inside a"):
        add_something()

    # Error if adding a bad object type
    with model:
        with pytest.raises(NetworkContextError, match="Objects of type.*added to net"):
            model.add(object())

    # Okay if add_to_container=False
    nengo.Ensemble(1, dimensions=1, add_to_container=False)
    nengo.Node(output=[0], add_to_container=False)

    # Errors if context is in a bad state (one in `Network.__exit__`)
    with pytest.raises(NetworkContextError, match="bad state; was expecting current"):
        with nengo.Network():
            nengo.Network.context.append("bad")
            with pytest.raises(NetworkContextError, match="context (bad) is not"):
                nengo.Ensemble(10, 1)

    clear_contexts()

    # Error if config is in a bad state
    with pytest.raises(ConfigError, match="bad state; was expecting current"):
        with nengo.Network():
            nengo.Config.context.append(nengo.Config(nengo.Ensemble))

    clear_contexts()

    # Error if context is empty when exiting
    with pytest.raises(NetworkContextError, match="bad state; was empty when exiting"):
        with nengo.Network():
            nengo.Network.context.clear()


def test_context_is_threadsafe():
    class CheckIndependence(ThreadedAssertion):
        def init_thread(self, worker):
            setattr(worker, "model", nengo.Network())
            worker.model.__enter__()

        def assert_thread(self, worker):
            assert list(nengo.Network.context) == [worker.model]

        def finish_thread(self, worker):
            worker.model.__exit__(*worker.exc_info)

    CheckIndependence(n_threads=2).run()


def test_get_objects():
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
            subnet2 = nengo.Network()

            with subnet2:
                ens3 = nengo.Ensemble(10, 1)

    all_objects = [
        ens1,
        pr1,
        node1,
        conn1,
        subnet,
        ens2,
        node2,
        conn2,
        pr2,
        ens3,
        subnet2,
    ]

    # Test that the lists contain the same elements, but order doesn't matter.
    # Counter is like a set, but also keeps track of the number of objects.
    assert Counter(all_objects) == Counter(model.all_objects)
    assert Counter([ens1, ens2, ens3]) == Counter(model.all_ensembles)
    assert Counter([node1, node2]) == Counter(model.all_nodes)
    assert Counter([conn1, conn2]) == Counter(model.all_connections)
    assert Counter([pr1, pr2]) == Counter(model.all_probes)
    assert Counter([subnet, subnet2]) == Counter(model.all_networks)
    # Make sure it works a second time
    assert Counter([ens1, ens2, ens3]) == Counter(model.all_ensembles)


def test_raises_exception_on_incompatiple_type_arguments():
    with pytest.raises(ValueError):
        nengo.Network(label=1)
    with pytest.raises(ValueError):
        nengo.Network(seed="label")


def test_n_neurons():
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)
        assert net.n_neurons == 10
        with nengo.Network() as subnet:
            nengo.Ensemble(30, 1)
            assert subnet.n_neurons == 30
        assert net.n_neurons == 40


def test_readonly_attributes():
    def test_attr(net, attr):
        with pytest.raises(ReadonlyError, match=f"Network.{attr}: {attr} is read-only"):
            setattr(net, attr, [])

    with nengo.Network() as net:
        for attr in (
            "objects",
            "ensembles",
            "nodes",
            "networks",
            "connections",
            "probes",
            "config",
        ):
            test_attr(net, attr)


def test_network_contains():
    out_ens = nengo.Ensemble(10, 1, add_to_container=False)
    with nengo.Network() as net:
        ens = nengo.Ensemble(10, 1)
        assert ens in net
        assert out_ens not in net
        assert "str" not in net
