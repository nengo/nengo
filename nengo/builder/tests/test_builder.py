import logging

import pytest
import nengo
from nengo.builder import Model
from nengo.builder.builder import Builder
from nengo.builder.ensemble import BuiltEnsemble


def test_seeding(Simulator, allclose):
    """Test that setting the model seed fixes everything"""

    #  TODO: this really just checks random parameters in ensembles.
    #   Are there other objects with random parameters that should be
    #   tested? (Perhaps initial weights of learned connections)

    m = nengo.Network(label="test_seeding")
    with m:
        input = nengo.Node(output=1, label="input")
        A = nengo.Ensemble(40, 1, label="A")
        B = nengo.Ensemble(20, 1, label="B")
        nengo.Connection(input, A)
        C = nengo.Connection(A, B, function=lambda x: x ** 2)

    m.seed = 872
    with Simulator(m) as sim:
        m1 = sim.data
    with Simulator(m) as sim:
        m2 = sim.data
    m.seed = 873
    with Simulator(m) as sim:
        m3 = sim.data

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = allclose(getattr(obj1, attr), getattr(obj2, attr)) == equal
            if not check:
                logging.info("%s: %s", attr, getattr(obj1, attr))
                logging.info("%s: %s", attr, getattr(obj2, attr))
            assert check

    ens_attrs = BuiltEnsemble._fields
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    conn_attrs = ("eval_points", "weights")
    Cs = [mi[C] for mi in [m1, m2, m3]]
    compare_objs(Cs[0], Cs[1], conn_attrs)
    compare_objs(Cs[0], Cs[2], conn_attrs, equal=False)


def test_hierarchical_seeding():
    """Changes to subnetworks shouldn't affect seeds in top-level network"""

    def create(make_extra, seed):
        objs = []
        with nengo.Network(seed=seed, label="n1") as model:
            objs.append(nengo.Ensemble(10, 1, label="e1"))
            with nengo.Network(label="n2"):
                objs.append(nengo.Ensemble(10, 1, label="e2"))
                if make_extra:
                    # This shouldn't affect any seeds
                    objs.append(nengo.Ensemble(10, 1, label="e3"))
            objs.append(nengo.Ensemble(10, 1, label="e4"))
        return model, objs

    same1, same1objs = create(False, 9)
    same2, same2objs = create(True, 9)
    diff, diffobjs = create(True, 10)

    m1 = Model()
    m1.build(same1)
    same1seeds = m1.seeds

    m2 = Model()
    m2.build(same2)
    same2seeds = m2.seeds

    m3 = Model()
    m3.build(diff)
    diffseeds = m3.seeds

    for diffobj, same2obj in zip(diffobjs, same2objs):
        # These seeds should all be different
        assert diffseeds[diffobj] != same2seeds[same2obj]

    # Skip the extra ensemble
    same2objs = same2objs[:2] + same2objs[3:]

    for same1obj, same2obj in zip(same1objs, same2objs):
        # These seeds should all be the same
        assert same1seeds[same1obj] == same2seeds[same2obj]


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
    model = nengo.builder.Model()
    ens = nengo.Ensemble(10, 1, add_to_container=False)
    model.seeds[ens] = 0
    model.build(ens)
    built_ens = model.params[ens]

    with pytest.warns(UserWarning, match="has already been built"):
        assert model.build(ens) is None
    assert model.params[ens] is built_ens


def test_register_builder_warning():
    """tests a warning for register_builder"""

    class Test:
        pass

    my_builder = Builder.register(Test)
    my_builder(1)
    with pytest.warns(Warning):
        my_builder(1)  # repeat setup warning
