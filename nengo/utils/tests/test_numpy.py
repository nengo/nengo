import numpy as np

from nengo.utils.numpy import meshgrid_nd


def test_meshgrid_nd():
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
    assert np.allclose(expected, actual)
