import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.ensemble import BuiltEnsemble
from nengo.builder.operator import DotInc, PreserveValue
from nengo.builder.signal import Signal, SignalDict
from nengo.exceptions import ObsoleteError, SignalError
from nengo.utils.compat import itervalues, range


def test_seeding(RefSimulator, logger):
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
    with RefSimulator(m) as sim:
        m1 = sim.model.params
    with RefSimulator(m) as sim:
        m2 = sim.model.params
    m.seed = 873
    with RefSimulator(m) as sim:
        m3 = sim.model.params

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.allclose(getattr(obj1, attr), getattr(obj2, attr)) ==
                     equal)
            if not check:
                logger.info("%s: %s", attr, getattr(obj1, attr))
                logger.info("%s: %s", attr, getattr(obj2, attr))
            assert check

    ens_attrs = BuiltEnsemble._fields
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    conn_attrs = ('eval_points', 'weights')
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

    with RefSimulator(same1) as sim:
        same1seeds = sim.model.seeds
    with RefSimulator(same2) as sim:
        same2seeds = sim.model.seeds
    with RefSimulator(diff) as sim:
        diffseeds = sim.model.seeds

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
    """Make sure Signal.initial_value works."""
    two_d = Signal([[1.], [1.]])
    assert np.allclose(two_d.initial_value, np.array([[1], [1]]))
    two_d_view = two_d[0, :]
    assert np.allclose(two_d_view.initial_value, np.array([1]))

    # cannot change signal value after creation
    with pytest.raises(SignalError):
        two_d.initial_value = np.array([[0.5], [-0.5]])
    with pytest.raises((ValueError, RuntimeError)):
        two_d.initial_value[...] = np.array([[0.5], [-0.5]])


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    m = Model(RefSimulator, dt=0)
    m.operators += [PreserveValue(five), PreserveValue(array),
                    DotInc(zero, zero, five), DotInc(zeroarray, one, array)]

    with RefSimulator(None, model=m) as sim:
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
    signaldict = SignalDict()

    scalar = Signal(1.)

    # Both __getitem__ and __setitem__ raise KeyError
    with pytest.raises(KeyError):
        signaldict[scalar]
    with pytest.raises(KeyError):
        signaldict[scalar] = np.array(1.)

    signaldict.init(scalar)
    assert np.allclose(signaldict[scalar], np.array(1.))
    # __getitem__ handles scalars
    assert signaldict[scalar].shape == ()

    one_d = Signal([1.])
    signaldict.init(one_d)
    assert np.allclose(signaldict[one_d], np.array([1.]))
    assert signaldict[one_d].shape == (1,)

    two_d = Signal([[1.], [1.]])
    signaldict.init(two_d)
    assert np.allclose(signaldict[two_d], np.array([[1.], [1.]]))
    assert signaldict[two_d].shape == (2, 1)

    # __getitem__ handles views implicitly (note no .init)
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
    signaldict = SignalDict()
    two_d = Signal([[1.], [1.]])
    signaldict.init(two_d)

    two_d_view = two_d[0, :]
    signaldict.init(two_d_view)

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


def test_signal_slicing(rng):
    slices = [0, 1, slice(None, -1), slice(1, None), slice(1, -1),
              slice(None, None, 3), slice(1, -1, 2)]

    x = np.arange(12, dtype=float)
    y = np.arange(24, dtype=float).reshape(4, 6)
    a = Signal(x.copy())
    b = Signal(y.copy())

    for i in range(100):
        si0, si1 = rng.randint(0, len(slices), size=2)
        s0, s1 = slices[si0], slices[si1]
        assert np.array_equiv(a[s0].initial_value, x[s0])
        assert np.array_equiv(b[s0, s1].initial_value, y[s0, s1])

    with pytest.raises(ValueError):
        a[[0, 2]]
    with pytest.raises(ValueError):
        b[[0, 1], [3, 4]]


def test_commonsig_readonly(RefSimulator):
    """Test that the common signals cannot be modified."""
    net = nengo.Network(label="test_commonsig")
    with RefSimulator(net) as sim:
        for sig in itervalues(sim.model.sig['common']):
            sim.signals.init(sig)
            with pytest.raises((ValueError, RuntimeError)):
                sim.signals[sig] = np.array([-1])
            with pytest.raises((ValueError, RuntimeError)):
                sim.signals[sig][...] = np.array([-1])


def test_obsolete_params(RefSimulator):
    with nengo.Network() as net:
        e = nengo.Ensemble(10, 1)
        c = nengo.Connection(e, e)
    with RefSimulator(net) as sim:
        pass
    with pytest.raises(ObsoleteError):
        sim.data[c].decoders
