import pytest

import nengo
from nengo.builder.builder import Builder


def test_seed_override(seed, allclose):
    """Test that seeds are not overwritten by the seeding function"""
    with nengo.Network(seed=seed - 1) as net:
        a = nengo.Ensemble(10, 1, seed=seed - 2)
        b = nengo.Ensemble(10, 1, seed=seed + 2)

    model = nengo.builder.Model()
    model.seeds[net] = seed + 1
    model.seeds[a] = seed + 2

    # note: intentionally setting this to the 'wrong' value, to check that
    # it isn't being overridden (things with seeds set should have seeded=True)
    model.seeded[net] = False
    model.seeded[a] = False

    model.build(net)

    assert model.seeds[net] == seed + 1
    assert model.seeds[a] == seed + 2
    assert not model.seeded[net]
    assert not model.seeded[a]
    assert allclose(model.params[a].gain, model.params[b].gain)


def test_build_twice():
    """Tests warning for building an object twice"""
    model = nengo.builder.Model()
    ens = nengo.Ensemble(10, 1, add_to_container=False)
    model.seeds[ens] = 0
    model.build(ens)
    built_ens = model.params[ens]

    with pytest.warns(UserWarning, match="has already been built"):
        assert model.build(ens) is None
    assert model.params[ens] is built_ens


def test_register_builder_twice_warning():
    """Tests warning for registering a builder twice"""

    class Test:
        pass

    my_builder = Builder.register(Test)
    my_builder(1)
    with pytest.warns(Warning, match="Type .* already has a builder. Overwriting"):
        my_builder(1)  # repeat setup warning
