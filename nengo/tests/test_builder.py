from __future__ import print_function

import numpy as np
import pytest

import nengo
import nengo.builder


def test_seeding():
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
    m1 = nengo.Simulator(m).model.params
    m2 = nengo.Simulator(m).model.params
    m.seed = 873
    m3 = nengo.Simulator(m).model.params

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.allclose(getattr(obj1, attr), getattr(obj2, attr)) ==
                     equal)
            if not check:
                print(attr, getattr(obj1, attr))
                print(attr, getattr(obj2, attr))
            assert check

    ens_attrs = nengo.builder.BuiltEnsemble._fields
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


def test_hierarchical_seeding():
    """Test that changes to subnetworks don't affect RNGs in other subnetworks
    """
    seed = 9

    def create(make_b):
        with nengo.Network():
            a = nengo.Ensemble(40, 1, label="a")
            if make_b:
                b = nengo.Ensemble(40, 1, label="b")

        with nengo.Network():
            c = nengo.Ensemble(20, 1, label="c")

        return c

    m1 = nengo.Network(seed=seed)
    with m1:
        c1 = create(False)

    m2 = nengo.Network(seed=seed)
    with m2:
        c2 = create(True)

    params1 = nengo.Simulator(m1).model.params[c1]
    params2 = nengo.Simulator(m2).model.params[c2]
    for key in ['gain', 'bias', 'encoders']:
        assert np.array_equal(getattr(params1, key), getattr(params2, key))


def test_signal():
    """Make sure assert_named_signals works."""
    nengo.builder.Signal(np.array(0.))
    nengo.builder.Signal.assert_named_signals = True
    with pytest.raises(AssertionError):
        nengo.builder.Signal(np.array(0.))

    # So that other tests that build signals don't fail...
    nengo.builder.Signal.assert_named_signals = False


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = nengo.builder.Signal([0])
    one = nengo.builder.Signal([1])
    five = nengo.builder.Signal([5.0])
    zeroarray = nengo.builder.Signal([[0], [0], [0]])
    array = nengo.builder.Signal([1, 2, 3])

    m = nengo.builder.Model(dt=0)
    m.operators += [nengo.builder.ProdUpdate(zero, zero, one, five),
                    nengo.builder.ProdUpdate(zeroarray, one, one, array)]

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
    """Tests simulator.SignalDict's dict overrides."""
    signaldict = nengo.builder.SignalDict()

    scalar = nengo.builder.Signal(1)

    # Both __getitem__ and __setitem__ raise KeyError
    with pytest.raises(KeyError):
        signaldict[scalar]
    with pytest.raises(KeyError):
        signaldict[scalar] = np.array(1.)

    signaldict.init(scalar, scalar.value)
    assert np.allclose(signaldict[scalar], np.array(1.))
    # __getitem__ handles scalars
    assert signaldict[scalar].shape == ()

    one_d = nengo.builder.Signal([1])
    signaldict.init(one_d, one_d.value)
    assert np.allclose(signaldict[one_d], np.array([1.]))
    assert signaldict[one_d].shape == (1,)

    two_d = nengo.builder.Signal([[1], [1]])
    signaldict.init(two_d, two_d.value)
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


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
