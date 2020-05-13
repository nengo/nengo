import numpy as np
import pytest

import nengo
from nengo.exceptions import BuildError, ValidationError
from nengo.transforms import ChannelShape, NoTransform, SparseMatrix
from nengo._vendor.npconv2d import conv2d


@pytest.mark.parametrize("dimensions", (1, 2))
@pytest.mark.parametrize("padding", ("same", "valid"))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("fixed_kernel", (True, False))
def test_convolution(
    dimensions, padding, channels_last, fixed_kernel, Simulator, allclose, rng, seed
):
    input_d = 4
    input_channels = 2
    output_channels = 5
    kernel_d = 3
    kernel_size = (kernel_d,) if dimensions == 1 else (kernel_d, kernel_d)
    output_d = input_d - kernel_d // 2 * 2 if padding == "valid" else input_d

    input_shape = (input_d, input_channels)
    kernel_shape = (kernel_d, input_channels, output_channels)
    output_shape = (output_d, output_channels)

    if dimensions == 2:
        input_shape = (input_d,) + input_shape
        kernel_shape = (kernel_d,) + kernel_shape
        output_shape = (output_d,) + output_shape

    if not channels_last:
        input_shape = tuple(np.roll(input_shape, 1))
        output_shape = tuple(np.roll(output_shape, 1))

    with nengo.Network(seed=seed) as net:
        x = rng.randn(*input_shape)
        w = rng.randn(*kernel_shape) if fixed_kernel else nengo.dists.Uniform(-0.1, 0.1)

        a = nengo.Node(np.ravel(x))
        b = nengo.Node(size_in=np.prod(output_shape))
        conn = nengo.Connection(
            a,
            b,
            synapse=None,
            transform=nengo.Convolution(
                output_channels,
                input_shape,
                init=w,
                padding=padding,
                kernel_size=kernel_size,
                strides=(1,) if dimensions == 1 else (1, 1),
                channels_last=channels_last,
            ),
        )
        p = nengo.Probe(b)

        # check error handling
        bad_in = nengo.Node([0])
        bad_out = nengo.Node(size_in=5)
        with pytest.raises(ValidationError):
            nengo.Connection(bad_in, b, transform=conn.transform)
        with pytest.raises(ValidationError):
            nengo.Connection(a, bad_out, transform=conn.transform)

    assert conn.transform.output_shape.shape == output_shape
    assert conn.transform.kernel_shape == kernel_shape

    with Simulator(net) as sim:
        sim.step()

    weights = sim.data[conn].weights
    if not channels_last:
        x = np.moveaxis(x, 0, -1)
    if dimensions == 1:
        x = x[:, None, :]
        weights = weights[:, None, :, :]
    truth = conv2d.conv2d(x[None, ...], weights, pad=padding.upper())[0]
    if not channels_last:
        truth = np.moveaxis(truth, -1, 0)

    assert allclose(sim.data[p][0], np.ravel(truth))


@pytest.mark.parametrize("encoders", (True, False))
@pytest.mark.parametrize("decoders", (True, False))
def test_convolution_nef(encoders, decoders, Simulator):
    with nengo.Network() as net:
        transform = nengo.transforms.Convolution(n_filters=2, input_shape=(3, 3, 1))
        a = nengo.Ensemble(9, 9)
        b = nengo.Ensemble(2, 2)
        nengo.Connection(
            a if decoders else a.neurons,
            b if encoders else b.neurons,
            transform=transform,
        )

    if decoders:
        # error if decoders
        with pytest.raises(BuildError, match="decoded connection"):
            with nengo.Simulator(net):
                pass
    else:
        # no error
        with nengo.Simulator(net):
            pass


@pytest.mark.parametrize("use_dist", (False, True))
@pytest.mark.parametrize("use_scipy", (False, True))
def test_sparse(use_dist, use_scipy, Simulator, rng, seed, plt, monkeypatch, allclose):
    if use_scipy:
        scipy_sparse = pytest.importorskip("scipy.sparse")
    else:
        monkeypatch.setattr(nengo.transforms, "scipy_sparse", None)
        monkeypatch.setattr(nengo.utils.numpy, "scipy_sparse", None)
        monkeypatch.setattr(nengo.utils.numpy, "is_spmatrix", lambda obj: False)

    input_d = 4
    output_d = 2
    shape = (output_d, input_d)

    inds = np.asarray([[0, 0], [1, 1], [0, 2], [1, 3]])
    weights = rng.uniform(0.25, 0.75, size=4)
    if use_dist:
        init = nengo.dists.Uniform(0.25, 0.75)
        indices = inds
    elif use_scipy:
        init = scipy_sparse.csr_matrix((weights, inds.T), shape=shape)
        indices = None
    else:
        init = weights
        indices = inds

    transform = nengo.transforms.Sparse(shape, indices=indices, init=init)

    sim_time = 1.0
    with nengo.Network(seed=seed) as net:
        x = nengo.processes.WhiteSignal(period=sim_time, high=10, seed=seed + 1)
        u = nengo.Node(x, size_out=4)
        a = nengo.Ensemble(100, 2)
        conn = nengo.Connection(u, a, synapse=None, transform=transform)
        ap = nengo.Probe(a, synapse=0.03)

    def run_sim():
        with Simulator(net) as sim:
            sim.run(sim_time)
        return sim

    if use_scipy:
        sim = run_sim()
    else:
        with pytest.warns(UserWarning, match="require Scipy"):
            sim = run_sim()

    actual_weights = sim.data[conn].weights

    full_transform = np.zeros(shape)
    full_transform[inds[:, 0], inds[:, 1]] = weights
    if use_dist:
        actual_weights = actual_weights.toarray()
        assert np.array_equal(actual_weights != 0, full_transform != 0)
        full_transform[:] = actual_weights

    conn.transform = full_transform
    with Simulator(net) as ref_sim:
        ref_sim.run(sim_time)

    plt.plot(ref_sim.trange(), ref_sim.data[ap], ":")
    plt.plot(sim.trange(), sim.data[ap])

    assert allclose(sim.data[ap], ref_sim.data[ap])


@pytest.mark.parametrize("encoders", (True, False))
@pytest.mark.parametrize("decoders", (True, False))
def test_sparse_nef(encoders, decoders, Simulator):
    # Sparse transforms currently don't work with NEF connections,
    # so just check the errors

    with nengo.Network() as net:
        transform = nengo.transforms.Sparse((2, 2), indices=[[0, 1], [1, 0]])
        a = nengo.Ensemble(2, 2)
        b = nengo.Ensemble(2, 2)
        nengo.Connection(
            a if decoders else a.neurons,
            b if encoders else b.neurons,
            transform=transform,
        )

    if decoders:
        # error if decoders
        with pytest.raises(BuildError, match="decoded connection"):
            with nengo.Simulator(net):
                pass
    else:
        # no error
        with nengo.Simulator(net):
            pass


def test_argreprs():
    """Test repr() for each transform type."""
    assert repr(nengo.Dense((1, 2), init=[[1, 1]])) == "Dense(shape=(1, 2))"

    assert (
        repr(nengo.Convolution(3, (1, 2, 3)))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3))"
    )
    assert (
        repr(nengo.Convolution(3, (1, 2, 3), kernel_size=(3, 2)))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3), "
        "kernel_size=(3, 2))"
    )
    assert (
        repr(nengo.Convolution(3, (1, 2, 3), channels_last=False))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3), "
        "channels_last=False)"
    )
    assert (
        repr(nengo.Sparse((1, 1), indices=((1, 1), (1, 1)))) == "Sparse(shape=(1, 1))"
    )
    assert (
        repr(nengo.Sparse((1, 1), indices=((1, 1), (1, 1), (1, 1)), init=2))
        == "Sparse(shape=(1, 1))"
    )


def test_SparseMatrix_str():
    """test repr() for SparseMatrix"""
    assert (
        repr(SparseMatrix(((1, 2), (3, 4)), (5, 6), (7, 8)))
        == "SparseMatrix(indices=array([[1, 2],\n       [3, 4]], dtype=int64), data=array([5, 6]), shape=(7, 8))"
    )


def test_channelshape_str():
    assert (
        repr(ChannelShape((1, 2, 3)))
        == "ChannelShape(shape=(1, 2, 3), channels_last=True)"
    )
    assert (
        repr(ChannelShape((1, 2, 3), channels_last=False))
        == "ChannelShape(shape=(1, 2, 3), channels_last=False)"
    )

    # `str` always has channels last
    assert str(ChannelShape((1, 2, 3))) == "(1, 2, ch=3)"
    assert str(ChannelShape((1, 2, 3), channels_last=False)) == "(ch=1, 2, 3)"


@pytest.mark.parametrize("dimensions", (1, 2))
def test_NoTransform(dimensions):
    """test repr() for NoTransform"""
    assert (
        repr(NoTransform(dimensions)) == "NoTransform(size_in=" + str(dimensions) + ")"
    )
