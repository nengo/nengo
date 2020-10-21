import numpy as np
import pytest

import nengo
from nengo._vendor.npconv2d import conv2d
from nengo.exceptions import BuildError, ValidationError


@pytest.mark.parametrize("x_mul", (1, 2, 3, 4))
@pytest.mark.parametrize("k_size", (1, 2, 3, 4))
@pytest.mark.parametrize("stride", (1, 2, 3, 4))
@pytest.mark.parametrize("padding", ("same", "valid"))
def test_convolution_shape(padding, stride, k_size, x_mul, rng, allclose):
    tf = pytest.importorskip("tensorflow")

    in_channels = 2
    out_channels = 3

    for i in range(2 * k_size):
        x_size = k_size + stride * (x_mul - 1) + i
        x_shape = (x_size, x_size, in_channels)
        k_shape = (k_size, k_size, in_channels, out_channels)

        x = rng.uniform(-1, 1, size=x_shape)
        kernel = rng.uniform(-1, 1, size=k_shape)
        y_tf = tf.nn.conv2d(
            x[None, ...], kernel, stride, padding=padding.upper()
        ).numpy()[0]

        y_np = conv2d.conv2d(
            x[None, ...], kernel, pad=padding.upper(), stride=(stride, stride)
        )[0]

        transform = nengo.Convolution(
            out_channels,
            x_shape,
            kernel_size=(k_size, k_size),
            strides=(stride, stride),
            padding=padding,
        )

        assert transform.output_shape.shape == y_tf.shape
        assert y_np.shape == y_tf.shape
        assert allclose(y_np, y_tf)


@pytest.mark.parametrize("dimensions", (1, 2))
@pytest.mark.parametrize("padding", ("same", "valid"))
@pytest.mark.parametrize("channels_last", (True, False))
@pytest.mark.parametrize("fixed_kernel", (True, False))
@pytest.mark.parametrize("transpose", (True, False))
def test_convolution(
    dimensions,
    padding,
    channels_last,
    fixed_kernel,
    transpose,
    Simulator,
    allclose,
    rng,
    seed,
):
    input_d = 4
    input_channels = 2
    output_channels = 5
    kernel_d = 3
    output_d = (
        input_d
        if padding == "same"
        else input_d + (1 if transpose else -1) * (kernel_d // 2 * 2)
    )

    kernel_size = (kernel_d,) if dimensions == 1 else (kernel_d, kernel_d)
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

    x = rng.randn(*input_shape)
    w = rng.randn(*kernel_shape) if fixed_kernel else nengo.dists.Uniform(-0.1, 0.1)

    if transpose:
        transform = nengo.transforms.ConvolutionTranspose(
            output_channels,
            input_shape,
            init=w,
            padding=padding,
            kernel_size=kernel_size,
            strides=(1,) if dimensions == 1 else (1, 1),
            channels_last=channels_last,
        )
    else:
        transform = nengo.Convolution(
            output_channels,
            input_shape,
            init=w,
            padding=padding,
            kernel_size=kernel_size,
            strides=(1,) if dimensions == 1 else (1, 1),
            channels_last=channels_last,
        )

    assert transform.output_shape.shape == output_shape

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(np.ravel(x))
        b = nengo.Node(size_in=np.prod(output_shape))
        conn = nengo.Connection(a, b, synapse=None, transform=transform)
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

    if transpose:
        outsize = (output_d, 1) if dimensions == 1 else (output_d, output_d)
        truth = conv2d.conv2d_gradx(
            weights, x[None, ...], xsize=outsize, pad=padding.upper()
        )[0]
    else:
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
            with Simulator(net):
                pass
    else:
        # no error
        with Simulator(net):
            pass


def test_convolution_invalid():
    with pytest.raises(ValidationError, match="exceeds the spatial size"):
        nengo.transforms.Convolution(n_filters=2, input_shape=(3, 2, 1))

    # valid output shape
    nengo.transforms.ConvolutionTranspose(
        n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 2)
    )
    with pytest.raises(ValidationError, match="number of dimensions"):
        # too many dims in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 2, 1)
        )
    with pytest.raises(ValidationError, match="number of channels"):
        # too many channels in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(5, 4, 3)
        )
    with pytest.raises(ValidationError, match="not a valid output shape"):
        # too many rows in output shape
        nengo.transforms.ConvolutionTranspose(
            n_filters=2, input_shape=(3, 2, 1), output_shape=(6, 4, 2)
        )


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
    """Sparse transforms currently don't work with NEF connections."""

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
            with Simulator(net):
                pass
    else:
        # no error
        with Simulator(net):
            pass
