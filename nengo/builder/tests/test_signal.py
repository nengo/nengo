import pickle

import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.signal import Signal, SignalDict
from nengo.exceptions import SignalError
from nengo.utils.numpy import scipy_sparse


def test_signaldict(allclose):
    """Tests SignalDict's dict overrides."""
    signaldict = SignalDict()

    scalar = Signal(1.0)

    # Both __getitem__ and __setitem__ raise KeyError
    with pytest.raises(KeyError):
        print(signaldict[scalar])
    with pytest.raises(KeyError):
        signaldict[scalar] = np.array(1.0)

    signaldict.init(scalar)

    # tests repeat init
    with pytest.raises(SignalError, match="Cannot add signal twice"):
        signaldict.init(scalar)

    assert allclose(signaldict[scalar], np.array(1.0))
    # __getitem__ handles scalars
    assert signaldict[scalar].shape == ()

    one_d = Signal([1.0])
    signaldict.init(one_d)
    assert allclose(signaldict[one_d], np.array([1.0]))
    assert signaldict[one_d].shape == (1,)

    two_d = Signal([[1.0], [1.0]])
    signaldict.init(two_d)
    assert allclose(signaldict[two_d], np.array([[1.0], [1.0]]))
    assert signaldict[two_d].shape == (2, 1)

    # __getitem__ handles views implicitly (note no .init)
    two_d_view = two_d[0, :]
    assert allclose(signaldict[two_d_view], np.array([1.0]))
    assert signaldict[two_d_view].shape == (1,)

    # __setitem__ ensures memory location stays the same
    memloc = signaldict[scalar].__array_interface__["data"][0]
    signaldict[scalar] = np.array(0.0)
    assert allclose(signaldict[scalar], np.array(0.0))
    assert signaldict[scalar].__array_interface__["data"][0] == memloc

    memloc = signaldict[one_d].__array_interface__["data"][0]
    signaldict[one_d] = np.array([0.0])
    assert allclose(signaldict[one_d], np.array([0.0]))
    assert signaldict[one_d].__array_interface__["data"][0] == memloc

    memloc = signaldict[two_d].__array_interface__["data"][0]
    signaldict[two_d] = np.array([[0.0], [0.0]])
    assert allclose(signaldict[two_d], np.array([[0.0], [0.0]]))
    assert signaldict[two_d].__array_interface__["data"][0] == memloc

    # __str__ pretty-prints signals and current values
    # Order not guaranteed for dicts, so we have to loop
    for k in signaldict:
        assert "%s %s" % (repr(k), repr(signaldict[k])) in str(signaldict)


def test_signaldict_reset(allclose):
    """Tests SignalDict's reset function."""
    signaldict = SignalDict()
    two_d = Signal([[1.0], [1.0]])
    signaldict.init(two_d)

    two_d_view = two_d[0, :]
    signaldict.init(two_d_view)

    signaldict[two_d_view] = -0.5
    assert allclose(signaldict[two_d], np.array([[-0.5], [1]]))

    signaldict[two_d] = np.array([[-1], [-1]])
    assert allclose(signaldict[two_d], np.array([[-1], [-1]]))
    assert allclose(signaldict[two_d_view], np.array([-1]))

    signaldict.reset(two_d_view)
    assert allclose(signaldict[two_d_view], np.array([1]))
    assert allclose(signaldict[two_d], np.array([[1], [-1]]))

    signaldict.reset(two_d)
    assert allclose(signaldict[two_d], np.array([[1], [1]]))


def test_assert_named_signals():
    """Make sure assert_named_signals works."""
    Signal(np.array(0.0))
    Signal.assert_named_signals = True
    with pytest.raises(AssertionError):
        Signal(np.array(0.0))

    # So that other tests that build signals don't fail...
    Signal.assert_named_signals = False


def test_signal_values(allclose):
    """Make sure Signal.initial_value works."""
    two_d = Signal([[1.0], [1.0]])
    assert allclose(two_d.initial_value, np.array([[1], [1]]))
    two_d_view = two_d[0, :]
    assert allclose(two_d_view.initial_value, np.array([1]))

    # cannot change signal value after creation
    with pytest.raises(SignalError):
        two_d.initial_value = np.array([[0.5], [-0.5]])
    with pytest.raises((ValueError, RuntimeError)):
        two_d.initial_value[...] = np.array([[0.5], [-0.5]])


def test_signal_reshape():
    """Tests Signal.reshape"""
    # check proper shape after reshape
    three_d = Signal(np.ones((2, 2, 2)))
    assert three_d.reshape((8,)).shape == (8,)
    assert three_d.reshape((4, 2)).shape == (4, 2)
    assert three_d.reshape((2, 4)).shape == (2, 4)
    assert three_d.reshape(-1).shape == (8,)
    assert three_d.reshape((4, -1)).shape == (4, 2)
    assert three_d.reshape((-1, 4)).shape == (2, 4)
    assert three_d.reshape((2, -1, 2)).shape == (2, 2, 2)
    assert three_d.reshape((1, 2, 1, 2, 2, 1)).shape == (1, 2, 1, 2, 2, 1)

    # check with non-contiguous arrays (and with offset)
    value = np.arange(20).reshape(5, 4)
    s = Signal(np.array(value), name="s")

    s0slice = slice(0, 3), slice(None, None, 2)
    s0shape = 2, 3
    s0 = s[s0slice].reshape(*s0shape)
    assert s.offset == 0
    assert np.array_equal(s0.initial_value, value[s0slice].reshape(*s0shape))

    s1slice = slice(1, None), slice(None, None, 2)
    s1shape = 2, 4
    s1 = s[s1slice].reshape(s1shape)
    assert s1.offset == 4 * s1.dtype.itemsize
    assert np.array_equal(s1.initial_value, value[s1slice].reshape(s1shape))

    # check error if non-contiguous array cannot be reshaped without copy
    s2slice = slice(None, None, 2), slice(None, None, 2)
    s2shape = 2, 3
    s2 = s[s2slice]
    with pytest.raises(SignalError):
        s2.reshape(s2shape)

    # check that views are working properly (incrementing `s` effects views)
    values = SignalDict()
    values.init(s)
    values.init(s0)
    values.init(s1)

    values[s] += 1
    assert np.array_equal(values[s0], value[s0slice].reshape(s0shape) + 1)
    assert np.array_equal(values[s1], value[s1slice].reshape(s1shape) + 1)


def test_signal_slicing(rng):
    slices = [
        0,
        1,
        slice(None, -1),
        slice(1, None),
        slice(1, -1),
        slice(None, None, 3),
        slice(1, -1, 2),
    ]

    x = np.arange(12, dtype=float)
    y = np.arange(24, dtype=float).reshape(4, 6)
    a = Signal(x.copy())
    b = Signal(y.copy())

    for _ in range(100):
        si0, si1 = rng.randint(0, len(slices), size=2)
        s0, s1 = slices[si0], slices[si1]
        assert np.array_equiv(a[s0].initial_value, x[s0])
        assert np.array_equiv(b[s0, s1].initial_value, y[s0, s1])

    with pytest.raises(ValueError):
        print(a[[0, 2]])
    with pytest.raises(ValueError):
        print(b[[0, 1], [3, 4]])


def test_commonsig_readonly():
    """Test that the common signals cannot be modified."""
    net = nengo.Network(label="test_commonsig")
    model = Model()
    model.build(net)
    signals = SignalDict()

    for sig in model.sig["common"].values():
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


def make_signal(sig_type, shape, indices, data):
    dense = np.zeros(shape)
    dense[indices[:, 0], indices[:, 1]] = data
    if sig_type == "sparse_scipy":
        m = scipy_sparse.csr_matrix((data, indices.T), shape=shape)
    elif sig_type == "sparse_nengo":
        m = nengo.transforms.SparseMatrix(data=data, indices=indices, shape=shape)
    else:
        m = dense
    return Signal(m, name="MySignal"), dense


@pytest.mark.parametrize("sig_type", ("dense", "sparse_scipy", "sparse_nengo"))
def test_signal_initial_value(sig_type, tmpdir, allclose):
    if sig_type == "sparse_scipy":
        pytest.importorskip("scipy.sparse")

    sig, dense = make_signal(
        sig_type,
        shape=(3, 3),
        indices=np.asarray([[0, 0], [0, 2], [1, 1], [2, 2]]),
        data=[1.0, 2.0, 1.0, 1.5],
    )

    # check initial_value equality
    assert allclose(
        sig.initial_value.toarray()
        if sig_type.startswith("sparse")
        else sig.initial_value,
        dense,
    )

    # cannot change once set
    with pytest.raises(SignalError, match="Cannot change initial value"):
        sig.initial_value = sig.initial_value

    # check signal pickles correctly
    pkl_path = str(tmpdir.join("tmp.pkl"))
    with open(pkl_path, "wb") as f:
        pickle.dump(sig, f)

    with open(pkl_path, "rb") as f:
        pkl_sig = pickle.load(f)

    # initial_value still matches after pickle/unpickle
    assert allclose(
        sig.initial_value.toarray()
        if sig_type.startswith("sparse")
        else sig.initial_value,
        pkl_sig.initial_value.toarray()
        if sig_type.startswith("sparse")
        else pkl_sig.initial_value,
    )


@pytest.mark.parametrize("sig_type", ("dense", "sparse_scipy", "sparse_nengo"))
def test_signal_slice_reshape(sig_type):
    if sig_type == "sparse_scipy":
        pytest.importorskip("scipy.sparse")

    sig, _ = make_signal(
        sig_type,
        shape=(3, 3),
        indices=np.asarray([[0, 0], [0, 2], [1, 1], [2, 2]]),
        data=[1.0, 2.0, 1.0, 1.5],
    )

    # check slicing
    if sig_type == "dense":
        sig_slice = sig[:2]
        assert sig_slice.shape == (2, 3)
        assert sig_slice.base is sig
        assert sig.may_share_memory(sig_slice)
    else:
        with pytest.raises(SignalError, match="sparse Signal"):
            print(sig[:2])

    # check reshaping
    if sig_type == "dense":
        sig_reshape = sig.reshape((1, 9))
        assert sig_reshape.shape == (1, 9)
        assert sig_reshape.base is sig
        assert sig.may_share_memory(sig_reshape)
    else:
        with pytest.raises(SignalError, match="sparse Signal"):
            sig.reshape((1, 9))


@pytest.mark.parametrize("sig_type", ("dense", "sparse_scipy", "sparse_nengo"))
def test_signal_properties(sig_type):
    if sig_type == "sparse_scipy":
        pytest.importorskip("scipy.sparse")

    sig, _ = make_signal(
        sig_type,
        shape=(3, 3),
        indices=np.asarray([[0, 0], [0, 2], [1, 1], [2, 2]]),
        data=[1.0, 2.0, 1.0, 1.5],
    )

    # check properties
    assert sig.base is sig
    assert sig.dtype == sig.initial_value.dtype == np.float64
    assert sig.elemoffset == 0
    assert sig.elemstrides == ((3, 1) if sig_type == "dense" else None)
    assert not sig.is_view
    assert sig.itemsize == 8
    assert sig.name == "MySignal"
    assert sig.nbytes == sig.itemsize * sig.size
    assert sig.ndim == 2
    assert sig.offset == 0
    assert not sig.readonly
    assert sig.shape == (3, 3)
    assert sig.size == (
        np.sum(sig.initial_value.toarray() != 0)
        if sig_type.startswith("sparse")
        else np.prod((3, 3))
    )
    assert sig.strides == (
        (3 * sig.itemsize, sig.itemsize) if sig_type == "dense" else None
    )
    assert sig.sparse if sig_type.startswith("sparse") else not sig.sparse

    # modifying properties
    sig.name = "NewName"
    assert sig.name == "NewName"
    sig.readonly = True
    assert sig.readonly


@pytest.mark.parametrize("sig_type", ("dense", "sparse_scipy", "sparse_nengo"))
def test_signal_init(sig_type):
    if sig_type == "sparse_scipy":
        pytest.importorskip("scipy.sparse")

    sig, dense = make_signal(
        sig_type,
        shape=(3, 3),
        indices=np.asarray([[0, 0], [0, 2], [1, 1], [2, 2]]),
        data=[1.0, 2.0, 1.0, 1.5],
    )
    signals = SignalDict()
    signals.init(sig)
    assert np.all(signals[sig] == dense)

    sig.readonly = True
    signals = SignalDict()
    signals.init(sig)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        signals[sig].data[0] = -1


def test_signal_shape():
    shape = (3, 4)
    sig = Signal(shape=shape)
    assert sig.shape == shape

    Signal(np.zeros(shape), shape=shape)
    with pytest.raises(AssertionError):
        Signal(np.zeros((2, 3)), shape=shape)


def tests_signal_nan():
    with_nan = np.ones(4)
    with_nan[1] = np.nan
    with pytest.raises(SignalError, match="contains NaNs"):
        Signal(initial_value=with_nan)
