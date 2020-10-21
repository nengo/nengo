import numpy as np
import pytest

import nengo
from nengo.exceptions import BuildError, ValidationError


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


def test_sparse_validation_errors():
    with pytest.raises(ValidationError, match="Either `init` must be a `scipy.sparse"):
        nengo.Sparse((1, 1))

    nengo.Sparse((3, 3), indices=[(0, 0), (1, 2)], init=[1, 2])
    with pytest.raises(ValidationError, match="Must be a vector.*length as `ind"):
        nengo.Sparse((3, 3), indices=[(0, 0), (1, 2)], init=[1, 2, 3])


def test_sparseinitparam_errors():
    class TestClass:
        sparse = nengo.transforms.SparseInitParam("sparse")

    test = TestClass()
    with pytest.raises(ValidationError, match="Must be `.*SparseMatrix` or .*spmatrix"):
        test.sparse = "a"


def test_notransform():
    transform = nengo.transforms.NoTransform(3)
    assert transform.size_in == transform.size_out == 3

    with pytest.raises(TypeError, match="Cannot sample a NoTransform"):
        transform.sample()
