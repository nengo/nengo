import logging

import numpy as np
import pytest

import nengo
from nengo.exceptions import MovedError, ValidationError, Unconvertible
from nengo.transforms import NoTransform
from nengo.utils.builder import (
    full_transform,
    generate_graphviz,
    objs_and_connections,
    remove_passthrough_nodes,
    _create_replacement_connection,
)


def test_create_replacement_exception():
    """ensures that _create_replacement_connection throws unconvertible"""

    with nengo.Network():

        class FakeObj:
            output = None

        class Test:
            post_obj = FakeObj
            pre_obj = FakeObj
            synapse = None
            function = "not none"
            transform = None

        c_in = Test
        c_out = Test
        with pytest.raises(Unconvertible):
            _create_replacement_connection(c_in, c_out)


def test_full_transform():
    """Tests full_transforms and its exceptions"""
    N = 30

    with nengo.Network():
        neurons3 = nengo.Ensemble(3, dimensions=1).neurons
        ens1 = nengo.Ensemble(N, dimensions=1)
        ens2 = nengo.Ensemble(N, dimensions=2)
        ens3 = nengo.Ensemble(N, dimensions=3)
        node1 = nengo.Node(output=[0])
        node2 = nengo.Node(output=[0, 0])
        node3 = nengo.Node(output=[0, 0, 0])

        class Test:
            transform = 1

        conn = Test  # nengo.Connection(node3[2], nengo.Ensemble(1, 1))
        with pytest.raises(ValidationError):
            full_transform(conn)

        class FakeObj:
            contents = [1, 2, 3]
            size_in = 1

        class FakeDense:
            def __init__(self):
                self.init = np.array([[[1], [2], [3]]])
                self.shape = (1, 1)

        class Test2:
            transform = FakeDense()
            function = 0
            post_slice = []
            pre_slice = []
            post_obj = FakeObj
            pre_obj = FakeObj
            function_info = np.array("nothing to see here")

        conn = Test2
        with pytest.raises(ValidationError):
            full_transform(conn)

        # Pre slice with default transform -> 1x3 transform
        conn = nengo.Connection(node3[2], ens1)
        assert isinstance(conn.transform, NoTransform)
        assert np.all(full_transform(conn) == np.array([[0, 0, 1]]))

        # Post slice with 1x1 transform -> 1x2 transform
        conn = nengo.Connection(node2[0], ens1, transform=-2)
        assert np.all(conn.transform.init == np.array(-2))
        assert np.all(full_transform(conn) == np.array([[-2, 0]]))

        # Post slice with 2x1 tranfsorm -> 3x1 transform
        conn = nengo.Connection(node1, ens3[::2], transform=[[1], [2]])
        assert np.all(conn.transform.init == np.array([[1], [2]]))
        assert np.all(full_transform(conn) == np.array([[1], [0], [2]]))

        # Both slices with 2x1 transform -> 3x2 transform
        conn = nengo.Connection(ens2[-1], neurons3[1:], transform=[[1], [2]])
        assert np.all(conn.transform.init == np.array([[1], [2]]))
        assert np.all(full_transform(conn) == np.array([[0, 0], [0, 1], [0, 2]]))

        # Full slices that can be optimized away
        conn = nengo.Connection(ens3[:], ens3, transform=2)
        assert np.all(conn.transform.init == np.array(2))
        assert np.all(full_transform(conn) == np.array(2))

        # Pre slice with 1x1 transform on 2x2 slices -> 2x3 transform
        conn = nengo.Connection(neurons3[:2], ens2, transform=-1)
        assert np.all(conn.transform.init == np.array(-1))
        assert np.all(full_transform(conn) == np.array([[-1, 0, 0], [0, -1, 0]]))

        # Both slices with 1x1 transform on 2x2 slices -> 3x3 transform
        conn = nengo.Connection(neurons3[1:], neurons3[::2], transform=-1)
        assert np.all(conn.transform.init == np.array(-1))
        assert np.all(
            full_transform(conn) == np.array([[0, -1, 0], [0, 0, 0], [0, 0, -1]])
        )

        # Both slices with 2x2 transform -> 3x3 transform
        conn = nengo.Connection(node3[[0, 2]], neurons3[1:], transform=[[1, 2], [3, 4]])
        assert np.all(conn.transform.init == np.array([[1, 2], [3, 4]]))
        assert np.all(
            full_transform(conn) == np.array([[0, 0, 0], [1, 0, 2], [3, 0, 4]])
        )

        # Both slices with 2x3 transform -> 3x3 transform... IN REVERSE!
        conn = nengo.Connection(
            neurons3[::-1], neurons3[[2, 0]], transform=[[1, 2, 3], [4, 5, 6]]
        )
        assert np.all(conn.transform.init == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(
            full_transform(conn) == np.array([[6, 5, 4], [0, 0, 0], [3, 2, 1]])
        )

        # Both slices using lists
        conn = nengo.Connection(
            neurons3[[1, 0, 2]], neurons3[[2, 1]], transform=[[1, 2, 3], [4, 5, 6]]
        )
        assert np.all(conn.transform.init == np.array([[1, 2, 3], [4, 5, 6]]))
        assert np.all(
            full_transform(conn) == np.array([[0, 0, 0], [5, 4, 6], [2, 1, 3]])
        )

        # using vector
        conn = nengo.Connection(ens3[[1, 0, 2]], ens3[[2, 0, 1]], transform=[1, 2, 3])
        assert np.all(conn.transform.init == np.array([1, 2, 3]))
        assert np.all(
            full_transform(conn) == np.array([[2, 0, 0], [0, 0, 3], [0, 1, 0]])
        )

        # using vector 1D
        conn = nengo.Connection(ens1, ens1, transform=[5])
        assert full_transform(conn).ndim != 1
        assert np.all(full_transform(conn) == 5)

        # using vector and lists
        conn = nengo.Connection(ens3[[1, 0, 2]], ens3[[2, 0, 1]], transform=[1, 2, 3])
        assert np.all(conn.transform.init == np.array([1, 2, 3]))
        assert np.all(
            full_transform(conn) == np.array([[2, 0, 0], [0, 0, 3], [0, 1, 0]])
        )

        # using multi-index lists
        conn = nengo.Connection(ens3, ens2[[0, 1, 0]])
        assert np.all(full_transform(conn) == np.array([[1, 0, 1], [0, 1, 0]]))


def test_graphviz_moved():
    with pytest.raises(MovedError):
        generate_graphviz()


def test_remove_passthrough():
    """Test scanning through a model and removing Nodes with output=None"""

    model = nengo.Network()
    with model:
        D = 3
        input = nengo.Node([1] * D, label="input")
        a = nengo.networks.EnsembleArray(50, D, label="a")
        b = nengo.networks.EnsembleArray(50, D, label="b")

        def printout(t, x):
            logging.info("%s, %s", t, x)

        output = nengo.Node(printout, size_in=D, label="output")

        nengo.Connection(input, a.input, synapse=0.01)
        nengo.Connection(a.output, b.input, synapse=0.01)
        nengo.Connection(b.output, b.input, synapse=0.01, transform=0.9)
        nengo.Connection(a.output, a.input, synapse=0.01, transform=np.ones((D, D)))
        nengo.Connection(b.output, output, synapse=0.01)

    objs, conns = remove_passthrough_nodes(*objs_and_connections(model))

    assert len(objs) == 8
    assert len(conns) == 21


def test_remove_passthrough_bg():
    """Test scanning through a model and removing Nodes with output=None"""

    model = nengo.Network()
    with model:
        D = 3
        input = nengo.Node([1] * D, label="input")

        def printout(t, x):
            logging.info("%s, %s", t, x)

        output = nengo.Node(printout, size_in=D, label="output")
        bg = nengo.networks.BasalGanglia(D, 20)
        nengo.Connection(input, bg.input, synapse=0.01)
        nengo.Connection(bg.output, output, synapse=0.01)

    objs, conns = remove_passthrough_nodes(*objs_and_connections(model))

    assert len(objs) == 17
    assert len(conns) == 42


def test_passthrough_errors():
    """Test errors removing Nodes with output=None"""

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        node = nengo.Node(None, size_in=1)
        nengo.Connection(a, node, synapse=0.01)
        nengo.Connection(node, b, synapse=0.01)
    with pytest.raises(Unconvertible):
        remove_passthrough_nodes(*objs_and_connections(model))

    model = nengo.Network()
    with model:
        node = nengo.Node(None, size_in=1)
        nengo.Connection(node, node, synapse=0.01)
    with pytest.raises(Unconvertible):
        remove_passthrough_nodes(*objs_and_connections(model))
