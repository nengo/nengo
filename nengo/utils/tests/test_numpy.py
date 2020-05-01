import itertools
import os
import time

import numpy as np
import pytest

import nengo.utils.numpy
from nengo.utils.numpy import (
    array_hash,
    betaincinv22_lookup,
    _make_betaincinv22_table,
    meshgrid_nd,
)
from nengo._vendor.scipy import expm


def test_meshgrid_nd(allclose):
    a = [0, 0, 1]
    b = [1, 2, 3]
    c = [23, 42]
    expected = [
        np.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        ),
        np.array(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
            ]
        ),
        np.array(
            [
                [[23, 42], [23, 42], [23, 42]],
                [[23, 42], [23, 42], [23, 42]],
                [[23, 42], [23, 42], [23, 42]],
            ]
        ),
    ]
    actual = meshgrid_nd(a, b, c)
    assert allclose(expected, actual)


@pytest.mark.parametrize("nnz", [7, 300])
def test_array_hash_sparse(nnz, rng):
    scipy_sparse = pytest.importorskip("scipy.sparse")

    if nnz == 7:
        shape = (5, 5)
        idxs_a = ([0, 0, 1, 2, 3, 3, 4], [0, 2, 3, 4, 2, 4, 0])
        idxs_b = ([0, 1, 1, 2, 3, 3, 4], [1, 2, 3, 4, 2, 4, 0])

        data_a = [1.0, 2.0, 1.5, 2.3, 1.2, 2.5, 1.8]
        data_b = [1.0, 1.0, 1.5, 2.3, 1.2, 2.5, 1.8]
    else:
        shape = (100, 100)

        idxs_a = np.unravel_index(rng.permutation(np.prod(shape))[:nnz], shape)
        idxs_b = np.unravel_index(rng.permutation(np.prod(shape))[:nnz], shape)

        data_a = rng.uniform(-1, 1, size=nnz)
        data_b = rng.uniform(-1, 1, size=nnz)

    matrices = [[] for _ in range(6)]

    for (rows, cols), data in itertools.product((idxs_a, idxs_b), (data_a, data_b)):

        csr = scipy_sparse.csr_matrix((data, (rows, cols)), shape=shape)
        matrices[0].append(csr)
        matrices[1].append(csr.tocsc())
        matrices[2].append(csr.tocoo())
        matrices[3].append(csr.tobsr())
        matrices[4].append(csr.todok())
        matrices[5].append(csr.tolil())
        # matrices[6].append(csr.todia())  # warns about inefficiency

    # ensure hash is reproducible
    for matrix in (m for kind in matrices for m in kind):
        assert array_hash(matrix) == array_hash(matrix)

    # ensure hash is different for different matrices
    for kind in matrices:
        hashes = [array_hash(matrix) for matrix in kind]
        assert len(np.unique(hashes)) == len(kind), (
            "Different matrices should have different hashes: %s" % hashes
        )


def test_expm(rng, allclose):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    for a in [np.eye(3), rng.randn(10, 10), -10 + rng.randn(10, 10)]:
        assert allclose(scipy_linalg.expm(a), expm(a))


def betaincinv22_test(plt, allclose, filename=None):
    scipy_special = pytest.importorskip("scipy.special")

    kwargs = {}
    if filename is not None:
        kwargs["filename"] = filename

    # call once to load table, so that doesn't effect timing
    betaincinv22_lookup(5, [0.1], **kwargs)

    dims = np.concatenate(
        [np.arange(1, 50), np.round(np.logspace(np.log10(51), 3.1)).astype(np.int64)]
    )
    x = np.linspace(0, 1, 1000)

    results = []
    for dim in dims:
        ref_timer = time.time()
        yref = scipy_special.betaincinv(dim / 2, 0.5, x)
        ref_timer = time.time() - ref_timer

        timer = time.time()
        y = betaincinv22_lookup(dim, x, **kwargs)
        timer = time.time() - timer

        results.append((yref, y, ref_timer, timer))

    n_show = 5
    resultsT = list(zip(*results))
    errors = np.abs(np.array(resultsT[0]) - np.array(resultsT[1])).max(axis=1)
    show_inds = np.argsort(errors)[-n_show:]

    subplots = plt.subplots(nrows=2, sharex=True)
    if isinstance(subplots, tuple):
        fig, ax = subplots

        for i in show_inds:
            yref, y, ref_timer, timer = results[i]
            dim = dims[i]

            ax[0].plot(x, y, label="dims=%d" % dim)
            ax[1].plot(x, y - yref)

        speedups = np.array(resultsT[2]) / np.array(resultsT[3])
        ax[0].set_title("average speedup = %0.1f times" % speedups.mean())
        ax[0].set_ylabel("value")
        ax[1].set_xlabel("input")
        ax[1].set_ylabel("error")
        ax[0].legend()

    for i, (yref, y, ref_timer, timer) in enumerate(results):
        # allow error to increase for higher dimensions (to 5e-3 when dims=1000)
        atol = 1e-3 + (np.log10(dims[i]) / 3) * 4e-3
        assert allclose(y, yref, atol=atol), "dims=%d" % dims[i]


def test_make_betaincinv22_table(monkeypatch, tmpdir, plt, allclose):
    pytest.importorskip("scipy.special")
    monkeypatch.setattr(nengo.utils.numpy, "_betaincinv22_table", None)

    filename = os.path.join(str(tmpdir), "betaincinv22_test_table.npz")

    _make_betaincinv22_table(filename=filename, n_interp=200, n_dims=50)
    betaincinv22_test(filename=filename, plt=plt, allclose=allclose)


def test_betaincinv22_lookup(monkeypatch, plt, allclose):
    pytest.importorskip("scipy.special")
    monkeypatch.setattr(nengo.utils.numpy, "_betaincinv22_table", None)

    betaincinv22_test(plt=plt, allclose=allclose)
