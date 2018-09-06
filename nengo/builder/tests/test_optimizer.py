import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import nengo
from nengo.builder.optimizer import SigMerger
from nengo.builder.signal import Signal
from nengo.spa.tests.test_thalamus import thalamus_net
from nengo.tests.test_learning_rules import learning_net


def test_sigmerger_check():
    # 0-d signals
    assert SigMerger.check([Signal(0), Signal(0)])
    assert not SigMerger.check([Signal(0), Signal(1)])

    # compatible along first axis
    assert SigMerger.check(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    assert SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    assert not SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    assert not SigMerger.check(
        [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    assert not SigMerger.check(
        [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    s1 = Signal(np.empty(5))
    s2 = Signal(np.empty(5))

    # mixed signal and view
    assert not SigMerger.check([s1, s1[:3]])

    # mixed bases
    assert not SigMerger.check([s1[:2], s2[2:]])

    # compatible views
    assert SigMerger.check([s1[:2], s1[2:]])


def test_sigmerger_check_signals():
    # 0-d signals
    SigMerger.check_signals([Signal(0), Signal(0)])
    with pytest.raises(ValueError):
        SigMerger.check_signals([Signal(0), Signal(1)])

    # compatible along first axis
    SigMerger.check_signals(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    SigMerger.check_signals(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    # compatible views
    s = Signal(np.empty(5))
    with pytest.raises(ValueError):
        SigMerger.check_signals([s[:2], s[2:]])


def test_sigmerger_check_views():
    s1 = Signal(np.empty((5, 5)))
    s2 = Signal(np.empty((5, 5)))

    # compatible along first axis
    SigMerger.check_views([s1[:1], s1[1:]])

    # compatible along second axis
    SigMerger.check_views([s1[:1, :1], s1[:1, 1:]], axis=1)
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[:1, :1], s1[:1, 1:]], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[:1], s1[1:, 0]])

    # different bases
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[:2], s2[2:]])


def test_sigmerger_merge(allclose):
    s1 = Signal(np.array([[0, 1], [2, 3]]))
    s2 = Signal(np.array([[4, 5]]))

    sig, replacements = SigMerger.merge([s1, s2])
    assert allclose(sig.initial_value, np.array([[0, 1], [2, 3], [4, 5]]))
    assert allclose(replacements[s1].initial_value, s1.initial_value)
    assert allclose(replacements[s2].initial_value, s2.initial_value)


def test_sigmerger_merge_views(allclose):
    s = Signal(np.array([[0, 1], [2, 3], [4, 5]]))
    v1, v2 = s[:2], s[2:]
    merged, _ = SigMerger.merge_views([v1, v2])

    assert allclose(merged.initial_value, s.initial_value)
    assert v1.base is s
    assert v2.base is s


@pytest.mark.parametrize("net", (thalamus_net, learning_net))
def test_optimizer_does_not_change_result(seed, net):
    model = net()
    model.seed = seed

    with model:
        # Add the default probe for every non-Probe object
        probes = [nengo.Probe(obj) for obj in model.all_objects
                  if not isinstance(obj, (nengo.Probe, nengo.Network))]
        # Also probe learning rules and neurons
        probes.extend(nengo.Probe(ens.neurons) for ens in model.all_ensembles)
        probes.extend(nengo.Probe(conn.learning_rule) for conn in
                      model.all_connections if conn.learning_rule is not None)

    with nengo.Simulator(model, optimize=False) as sim:
        sim.run(0.1)
    with nengo.Simulator(model, optimize=True) as sim_opt:
        sim_opt.run(0.1)

    for probe in probes:
        assert_almost_equal(sim.data[probe], sim_opt.data[probe])
