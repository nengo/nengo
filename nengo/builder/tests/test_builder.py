import os
import subprocess
import sys
import textwrap

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


@pytest.mark.skipif(sys.version_info < (3, 6, 0), reason="requires ordered dicts")
def test_deterministic_op_order(tmp_path):
    code = textwrap.dedent(
        """
    import nengo

    with nengo.Network(seed=0) as net:
        # use ensemblearrays as they create a lot of parallel ops
        ens0 = nengo.networks.EnsembleArray(1, 100)
        ens1 = nengo.networks.EnsembleArray(1, 100)
        nengo.Connection(ens0.output, ens1.input)
        nengo.Probe(ens1.output)

    # the optimizer uses WeakSets, which seem incompatible with ordering
    with nengo.Simulator(net, progress_bar=False, optimize=False) as sim:
        ops = sim.step_order

    for op in ops:
        print(type(op))
        for s in op.all_signals:
            print(s._name)
            print(s.shape)
            print(s.initial_value)
    """
    )
    tmp_path = tmp_path / "test.py"
    tmp_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()

    env["PYTHONHASHSEED"] = "0"
    output0 = subprocess.run(
        [sys.executable, str(tmp_path)],
        stdout=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        check=True,
    )

    env["PYTHONHASHSEED"] = "1"
    output1 = subprocess.run(
        [sys.executable, str(tmp_path)],
        stdout=subprocess.PIPE,
        env=env,
        encoding="utf-8",
        check=True,
    )

    assert output0.stdout == output1.stdout
