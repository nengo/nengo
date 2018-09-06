import itertools

import numpy as np
import pytest

from nengo.utils.numpy import array_hash, meshgrid_nd
from nengo._vendor.scipy import expm


def test_meshgrid_nd(allclose):
    a = [0, 0, 1]
    b = [1, 2, 3]
    c = [23, 42]
    expected = [
        np.array([[[0, 0], [0, 0], [0, 0]],
                  [[0, 0], [0, 0], [0, 0]],
                  [[1, 1], [1, 1], [1, 1]]]),
        np.array([[[1, 1], [2, 2], [3, 3]],
                  [[1, 1], [2, 2], [3, 3]],
                  [[1, 1], [2, 2], [3, 3]]]),
        np.array([[[23, 42], [23, 42], [23, 42]],
                  [[23, 42], [23, 42], [23, 42]],
                  [[23, 42], [23, 42], [23, 42]]])]
    actual = meshgrid_nd(a, b, c)
    assert allclose(expected, actual)


@pytest.mark.parametrize('nnz', [7, 300])
def test_array_hash_sparse(nnz, rng):
    scipy_sparse = pytest.importorskip('scipy.sparse')

    if nnz == 7:
        shape = (5, 5)
        rows_a = [0, 0, 1, 2, 3, 3, 4]
        rows_b = [0, 1, 1, 2, 3, 3, 4]

        cols_a = [0, 2, 3, 4, 2, 4, 0]
        cols_b = [1, 2, 3, 4, 2, 4, 0]

        data_a = [1., 2., 1.5, 2.3, 1.2, 2.5, 1.8]
        data_b = [1., 1., 1.5, 2.3, 1.2, 2.5, 1.8]
    else:
        shape = (100, 100)

        inds_a = rng.permutation(np.prod(shape))[:nnz]
        inds_b = rng.permutation(np.prod(shape))[:nnz]
        rows_a, cols_a = np.unravel_index(inds_a, shape)
        rows_b, cols_b = np.unravel_index(inds_b, shape)

        data_a = rng.uniform(-1, 1, size=nnz)
        data_b = rng.uniform(-1, 1, size=nnz)

    matrices = [[] for _ in range(6)]

    for (rows, cols), data in itertools.product(
            ((rows_a, cols_a), (rows_b, cols_b)), (data_a, data_b)):

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
            "Different matrices should have different hashes: %s" % hashes)


def test_expm(rng, allclose):
    scipy_linalg = pytest.importorskip('scipy.linalg')
    for a in [np.eye(3), rng.randn(10, 10), -10 + rng.randn(10, 10)]:
        assert allclose(scipy_linalg.expm(a), expm(a))
