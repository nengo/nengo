from __future__ import print_function

import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.ensemble import BuiltEnsemble
from nengo.builder.operator import DotInc, PreserveValue
from nengo.builder.signal import Signal
from nengo.utils.compat import itervalues


def test_seeding(RefSimulator):
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
    m1 = RefSimulator(m).model.params
    m2 = RefSimulator(m).model.params
    m.seed = 873
    m3 = RefSimulator(m).model.params

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.allclose(getattr(obj1, attr), getattr(obj2, attr)) ==
                     equal)
            if not check:
                print(attr, getattr(obj1, attr))
                print(attr, getattr(obj2, attr))
            assert check

    ens_attrs = BuiltEnsemble._fields
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    conn_attrs = ('decoders', 'eval_points')  # transform is static, unchecked
    Cs = [mi[C] for mi in [m1, m2, m3]]
    compare_objs(Cs[0], Cs[1], conn_attrs)
    compare_objs(Cs[0], Cs[2], conn_attrs, equal=False)


def test_hierarchical_seeding(RefSimulator):
    """Changes to subnetworks shouldn't affect seeds in top-level network"""

    def create(make_extra, seed):
        objs = []
        with nengo.Network(seed=seed, label='n1') as model:
            objs.append(nengo.Ensemble(10, 1, label='e1'))
            with nengo.Network(label='n2'):
                objs.append(nengo.Ensemble(10, 1, label='e2'))
                if make_extra:
                    # This shouldn't affect any seeds
                    objs.append(nengo.Ensemble(10, 1, label='e3'))
            objs.append(nengo.Ensemble(10, 1, label='e4'))
        return model, objs

    same1, same1objs = create(False, 9)
    same2, same2objs = create(True, 9)
    diff, diffobjs = create(True, 10)

    same1seeds = RefSimulator(same1).model.seeds
    same2seeds = RefSimulator(same2).model.seeds
    diffseeds = RefSimulator(diff).model.seeds

    for diffobj, same2obj in zip(diffobjs, same2objs):
        # These seeds should all be different
        assert diffseeds[diffobj] != same2seeds[same2obj]

    # Skip the extra ensemble
    same2objs = same2objs[:2] + same2objs[3:]

    for same1obj, same2obj in zip(same1objs, same2objs):
        # These seeds should all be the same
        assert same1seeds[same1obj] == same2seeds[same2obj]


def test_signal():
    """Make sure assert_named_signals works."""
    Signal(np.array(0.))
    Signal.assert_named_signals = True
    with pytest.raises(AssertionError):
        Signal(np.array(0.))

    # So that other tests that build signals don't fail...
    Signal.assert_named_signals = False


def test_signal_values():
    """Make sure Signal.value and SignalView.value work."""
    two_d = Signal([[1], [1]])
    assert np.allclose(two_d.value, np.array([[1], [1]]))
    two_d_view = two_d[0, :]
    assert np.allclose(two_d_view.value, np.array([1]))
    two_d.value[...] = np.array([[0.5], [-0.5]])
    assert np.allclose(two_d_view.value, np.array([0.5]))


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    m = Model(dt=0)
    m.operators += [PreserveValue(five),
                    PreserveValue(array),
                    DotInc(zero, zero, five),
                    DotInc(zeroarray, one, array)]

    sim = RefSimulator(None, model=m)
    assert zero.value[0] == 0
    assert one.value[0] == 1
    assert five.value[0] == 5.0
    assert np.all(np.array([1, 2, 3]) == array.value)
    sim.step()
    assert zero.value[0] == 0
    assert one.value[0] == 1
    assert five.value[0] == 5.0
    assert np.all(np.array([1, 2, 3]) == array.value)


def test_signal_views():
    """Tests Signal view/slicing properties."""

    scalar = Signal(1)

    assert np.allclose(scalar.value, np.array(1.))
    # __getitem__ handles scalars
    assert scalar.value.shape == ()

    one_d = Signal([1])
    assert np.allclose(one_d.value, np.array([1.]))
    assert one_d.value.shape == (1,)

    two_d = Signal([[1], [1]])
    assert np.allclose(two_d.value, np.array([[1.], [1.]]))
    assert two_d.value.shape == (2, 1)

    # __getitem__ handles views
    two_d_view = two_d[0, :]
    assert np.allclose(two_d_view.value, np.array([1.]))
    assert two_d_view.value.shape == (1,)

    # __setitem__ ensures memory location stays the same
    memloc = scalar.value.__array_interface__['data'][0]
    scalar.value = np.array(0.)
    assert np.allclose(scalar.value, np.array(0.))
    assert scalar.value.__array_interface__['data'][0] == memloc

    memloc = one_d.value.__array_interface__['data'][0]
    one_d.value = np.array([0.])
    assert np.allclose(one_d.value, np.array([0.]))
    assert one_d.value.__array_interface__['data'][0] == memloc

    memloc = two_d.value.__array_interface__['data'][0]
    two_d.value = np.array([[0.], [0.]])
    assert np.allclose(two_d.value, np.array([[0.], [0.]]))
    assert two_d.value.__array_interface__['data'][0] == memloc


def test_signal_reshape():
    """Tests Signal.reshape"""
    three_d = Signal(np.ones((2, 2, 2)))
    assert three_d.reshape((8,)).shape == (8,)
    assert three_d.reshape((4, 2)).shape == (4, 2)
    assert three_d.reshape((2, 4)).shape == (2, 4)
    assert three_d.reshape(-1).shape == (8,)
    assert three_d.reshape((4, -1)).shape == (4, 2)
    assert three_d.reshape((-1, 4)).shape == (2, 4)
    assert three_d.reshape((2, -1, 2)).shape == (2, 2, 2)
    assert three_d.reshape((1, 2, 1, 2, 2, 1)).shape == (1, 2, 1, 2, 2, 1)


def test_commonsig_readonly(RefSimulator):
    """Test that the common signals cannot be modified."""
    net = nengo.Network(label="test_commonsig")
    sim = RefSimulator(net)
    for sig in itervalues(sim.model.sig['common']):
        with pytest.raises((ValueError, RuntimeError)):
            sig.value = np.array([-1])
        with pytest.raises((ValueError, RuntimeError)):
            sig.value[...] = np.array([-1])
