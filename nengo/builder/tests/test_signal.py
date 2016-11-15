import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.signal import Signal, SignalDict
from nengo.exceptions import SignalError
from nengo.utils.compat import itervalues


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


def test_assert_named_signals():
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


def test_commonsig_readonly():
    """Test that the common signals cannot be modified."""
    net = nengo.Network(label="test_commonsig")
    model = Model()
    model.build(net)
    signals = SignalDict()

    for sig in itervalues(model.sig['common']):
        signals.init(sig)
        with pytest.raises((ValueError, RuntimeError)):
            signals[sig] = np.array([-1])
        with pytest.raises((ValueError, RuntimeError)):
            signals[sig][...] = np.array([-1])


def test_signal_offset():
    value = np.eye(3)
    s = Signal(value)
    assert s.offset == 0
    assert s[1].offset == value.strides[0]

    value_view = value[1]
    s = Signal(value_view)
    assert s.offset == 0
    assert s[0:].offset == 0
    assert s[1:].offset == value.strides[1]
