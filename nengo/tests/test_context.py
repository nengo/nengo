import pytest

import nengo


def test_basic_context(Simulator):
    model1 = nengo.Network("test1")
    with model1:
        e = nengo.Ensemble(nengo.LIF(1), 1)
        n = nengo.Node([0])
        assert e in model1.ensembles
        assert n in model1.nodes

        con = nengo.Network()
        with con:
            e2 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e2 in con.ensembles
        assert not e2 in model1.ensembles

        e3 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e3 in model1.ensembles

    with nengo.Network("test") as model2:
        e4 = nengo.Ensemble(nengo.LIF(1), 1)
        assert not e4 in model1.ensembles
        assert e4 in model2.ensembles


def test_nested_context(Simulator):
    model = nengo.Network()
    with model:
        con1 = nengo.Network()
        con2 = nengo.Network()
        con3 = nengo.Network()

        with con1:
            e1 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e1 in con1.ensembles

            with con2:
                e2 = nengo.Ensemble(nengo.LIF(1), 1)
                assert e2 in con2.ensembles
                assert not e2 in con1.ensembles

                with con3:
                    e3 = nengo.Ensemble(nengo.LIF(1), 1)
                    assert e3 in con3.ensembles
                    assert not e3 in con2.ensembles \
                        and not e3 in con1.ensembles

                e4 = nengo.Ensemble(nengo.LIF(1), 1)
                assert e4 in con2.ensembles
                assert not e4 in con3.ensembles

            e5 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e5 in con1.ensembles

        e6 = nengo.Ensemble(nengo.LIF(1), 1)
        assert not e6 in con1.ensembles


def test_context_errors(Simulator):
    def add_something():
        nengo.Ensemble(nengo.LIF(1), 1)

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

    # Okay if add_to_network=False
    nengo.Ensemble(nengo.LIF(1), 1, add_to_network=False)
    nengo.Node(output=[0], add_to_network=False)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
