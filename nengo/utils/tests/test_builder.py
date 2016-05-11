import numpy as np
import pytest

import nengo
from nengo.exceptions import BuildError
from nengo.params import Deferral
from nengo.utils.builder import assert_no_deferred_params, full_transform


def test_full_transform():
    N = 30

    with nengo.Network():
        neurons3 = nengo.Ensemble(3, dimensions=1).neurons
        ens1 = nengo.Ensemble(N, dimensions=1)
        ens2 = nengo.Ensemble(N, dimensions=2)
        ens3 = nengo.Ensemble(N, dimensions=3)
        node1 = nengo.Node(output=[0])
        node2 = nengo.Node(output=[0, 0])
        node3 = nengo.Node(output=[0, 0, 0])

        # Pre slice with default transform -> 1x3 transform
        conn = nengo.Connection(node3[2], ens1)
        assert np.all(conn.transform == np.array(1))
        assert np.all(full_transform(conn) == np.array([[0, 0, 1]]))

        # Post slice with 1x1 transform -> 1x2 transform
        conn = nengo.Connection(node2[0], ens1, transform=-2)
        assert np.all(conn.transform == np.array(-2))
        assert np.all(full_transform(conn) == np.array([[-2, 0]]))

        # Post slice with 2x1 tranfsorm -> 3x1 transform
        conn = nengo.Connection(node1, ens3[::2], transform=[[1], [2]])
        assert np.all(conn.transform == np.array([[1], [2]]))
        assert np.all(full_transform(conn) == np.array([[1], [0], [2]]))

        # Both slices with 2x1 transform -> 3x2 transform
        conn = nengo.Connection(ens2[-1], neurons3[1:], transform=[[1], [2]])
        assert np.all(conn.transform == np.array([[1], [2]]))
        assert np.all(full_transform(conn) == np.array(
            [[0, 0], [0, 1], [0, 2]]))

        # Full slices that can be optimized away
        conn = nengo.Connection(ens3[:], ens3, transform=2)
        assert np.all(conn.transform == np.array(2))
        assert np.all(full_transform(conn) == np.array(2))

        # Pre slice with 1x1 transform on 2x2 slices -> 2x3 transform
        conn = nengo.Connection(neurons3[:2], ens2, transform=-1)
        assert np.all(conn.transform == np.array(-1))
        assert np.all(full_transform(conn) == np.array(
            [[-1, 0, 0], [0, -1, 0]]))

        # Both slices with 1x1 transform on 2x2 slices -> 3x3 transform
        conn = nengo.Connection(neurons3[1:], neurons3[::2], transform=-1)
        assert np.all(conn.transform == np.array(-1))
        assert np.all(full_transform(conn) == np.array([[0, -1, 0],
                                                       [0, 0, 0],
                                                       [0, 0, -1]]))

        # Both slices with 2x2 transform -> 3x3 transform
        conn = nengo.Connection(node3[[0, 2]], neurons3[1:],
                                transform=[[1, 2], [3, 4]])
        assert np.all(conn.transform == np.array([[1, 2], [3, 4]]))
        assert np.all(full_transform(conn) == np.array([[0, 0, 0],
                                                       [1, 0, 2],
                                                       [3, 0, 4]]))

        # Both slices with 2x3 transform -> 3x3 transform... IN REVERSE!
        conn = nengo.Connection(neurons3[::-1], neurons3[[2, 0]],
                                transform=[[1, 2, 3], [4, 5, 6]])
        assert np.all(conn.transform == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(full_transform(conn) == np.array([[6, 5, 4],
                                                       [0, 0, 0],
                                                       [3, 2, 1]]))

        # Both slices using lists
        conn = nengo.Connection(neurons3[[1, 0, 2]], neurons3[[2, 1]],
                                transform=[[1, 2, 3], [4, 5, 6]])
        assert np.all(conn.transform == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(full_transform(conn) == np.array([[0, 0, 0],
                                                       [5, 4, 6],
                                                       [2, 1, 3]]))

        # using vector
        conn = nengo.Connection(ens3[[1, 0, 2]], ens3[[2, 0, 1]],
                                transform=[1, 2, 3])
        assert np.all(conn.transform == np.array([1, 2, 3]))
        assert np.all(full_transform(conn) == np.array([[2, 0, 0],
                                                       [0, 0, 3],
                                                       [0, 1, 0]]))

        # using vector 1D
        conn = nengo.Connection(ens1, ens1, transform=[5])
        assert full_transform(conn).ndim != 1
        assert np.all(full_transform(conn) == 5)

        # using vector and lists
        conn = nengo.Connection(ens3[[1, 0, 2]], ens3[[2, 0, 1]],
                                transform=[1, 2, 3])
        assert np.all(conn.transform == np.array([1, 2, 3]))
        assert np.all(full_transform(conn) == np.array([[2, 0, 0],
                                                       [0, 0, 3],
                                                       [0, 1, 0]]))

        # using multi-index lists
        conn = nengo.Connection(ens3, ens2[[0, 1, 0]])
        assert np.all(full_transform(conn) == np.array([[1, 0, 1],
                                                       [0, 1, 0]]))


def test_assert_no_deferred_params():
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)

    assert_no_deferred_params(model)

    ens.radius = Deferral(lambda model, ens: 0.5)

    with pytest.raises(BuildError):
        assert_no_deferred_params(model)
