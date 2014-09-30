from __future__ import print_function

import numpy as np
import pytest

import nengo
import nengo.builder as nb


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

    ens_attrs = nb.BuiltEnsemble._fields
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
    nb.Signal(np.array(0.))
    nb.Signal.assert_named_signals = True
    with pytest.raises(AssertionError):
        nb.Signal(np.array(0.))

    # So that other tests that build signals don't fail...
    nb.Signal.assert_named_signals = False


def test_signal_values():
    """Make sure Signal.value and SignalView.value work."""
    two_d = nb.Signal([[1], [1]])
    assert np.allclose(two_d.value, np.array([[1], [1]]))
    two_d_view = two_d[0, :]
    assert np.allclose(two_d_view.value, np.array([1]))
    two_d.value[...] = np.array([[0.5], [-0.5]])
    assert np.allclose(two_d_view.value, np.array([0.5]))


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = nb.Signal([0])
    one = nb.Signal([1])
    five = nb.Signal([5.0])
    zeroarray = nb.Signal([[0], [0], [0]])
    array = nb.Signal([1, 2, 3])

    m = nb.Model(dt=0)
    m.operators += [nb.PreserveValue(five),
                    nb.PreserveValue(array),
                    nb.DotInc(zero, zero, five),
                    nb.DotInc(zeroarray, one, array)]

    sim = RefSimulator(None, model=m)
    assert sim.signals[zero][0] == 0
    assert sim.signals[one][0] == 1
    assert sim.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == sim.signals[array])
    sim.step()
    assert sim.signals[zero][0] == 0
    assert sim.signals[one][0] == 1
    assert sim.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == sim.signals[array])


def test_signaldict():
    """Tests SignalDict's dict overrides."""
    signaldict = nb.SignalDict()

    scalar = nb.Signal(1)

    # Both __getitem__ and __setitem__ raise KeyError
    with pytest.raises(KeyError):
        signaldict[scalar]
    with pytest.raises(KeyError):
        signaldict[scalar] = np.array(1.)

    signaldict.init(scalar)
    assert np.allclose(signaldict[scalar], np.array(1.))
    # __getitem__ handles scalars
    assert signaldict[scalar].shape == ()

    one_d = nb.Signal([1])
    signaldict.init(one_d)
    assert np.allclose(signaldict[one_d], np.array([1.]))
    assert signaldict[one_d].shape == (1,)

    two_d = nb.Signal([[1], [1]])
    signaldict.init(two_d)
    assert np.allclose(signaldict[two_d], np.array([[1.], [1.]]))
    assert signaldict[two_d].shape == (2, 1)

    # __getitem__ handles views
    two_d_view = two_d[0, :]
    assert np.allclose(signaldict[two_d_view], np.array([1.]))
    assert signaldict[two_d_view].shape == (1,)

    # __setitem__ ensures memory location stays the same
    memloc = signaldict[scalar].__array_interface__['data'][0]
    signaldict[scalar] = np.array(0.)
    assert np.allclose(signaldict[scalar], np.array(0.))
    assert signaldict[scalar].__array_interface__['data'][0] == memloc

    memloc = signaldict[one_d].__array_interface__['data'][0]
    signaldict[one_d] = np.array([0.])
    assert np.allclose(signaldict[one_d], np.array([0.]))
    assert signaldict[one_d].__array_interface__['data'][0] == memloc

    memloc = signaldict[two_d].__array_interface__['data'][0]
    signaldict[two_d] = np.array([[0.], [0.]])
    assert np.allclose(signaldict[two_d], np.array([[0.], [0.]]))
    assert signaldict[two_d].__array_interface__['data'][0] == memloc

    # __str__ pretty-prints signals and current values
    # Order not guaranteed for dicts, so we have to loop
    for k in signaldict:
        assert "%s %s" % (repr(k), repr(signaldict[k])) in str(signaldict)


def test_signaldict_reset():
    """Tests SignalDict's reset function."""
    signaldict = nb.SignalDict()
    two_d = nb.Signal([[1], [1]])
    signaldict.init(two_d)

    two_d_view = two_d[0, :]
    signaldict[two_d_view] = -0.5
    assert np.allclose(signaldict[two_d], np.array([[-0.5], [1]]))

    signaldict[two_d] = np.array([[-1], [-1]])
    assert np.allclose(signaldict[two_d], np.array([[-1], [-1]]))
    assert np.allclose(signaldict[two_d_view], np.array([-1]))

    signaldict.reset(two_d_view)
    assert np.allclose(signaldict[two_d_view], np.array([1]))
    assert np.allclose(signaldict[two_d], np.array([[1], [-1]]))

    signaldict.reset(two_d)
    assert np.allclose(signaldict[two_d], np.array([[1], [1]]))


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
