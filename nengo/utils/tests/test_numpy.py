from __future__ import absolute_import

import numpy as np

from nengo.utils.numpy import array_like, meshgrid_nd


def test_array_like():
    # test complex function signature
    @array_like('a', 'b')
    def fn1(a, b=[1, 2], *args, **kwargs):
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert kwargs['c'] == 'hi'

    fn1([1, 3], c='hi')

    # test multiple decorators
    @array_like('a', ndmin=1)
    @array_like('b', ndmin=2)
    def fn2(a, b):
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.ndim >= 1
        assert b.ndim >= 2

    fn2(3, 5)


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
