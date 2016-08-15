from __future__ import absolute_import, division

import numpy as np

from nengo.utils.numpy import _expand_dims, norm, meshgrid_nd, rms


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


def test_expand_dims():
    a = np.array([0, 1])
    assert _expand_dims(a, 0).shape == (1, 2)
    assert _expand_dims(a, 1).shape == (2, 1)
    assert _expand_dims(a, -1).shape == (2, 1)
    assert _expand_dims(a, (0, 0, 1, -1)).shape == (1, 1, 1, 2, 1)


def test_norm():
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

    assert np.allclose(norm(a), np.sqrt(140))
    assert np.allclose(norm(a, axis=0), np.sqrt([[16, 26], [40, 58]]))
    assert np.allclose(norm(a, axis=-1), np.sqrt([[1, 13], [41, 85]]))
    assert np.allclose(norm(a, axis=(1, 2)), np.sqrt([14, 126]))
    assert np.allclose(norm(a, axis=(0, -1)), np.sqrt([42, 98]))

    assert norm(a, keepdims=True).shape == (1, 1, 1)
    assert norm(a, axis=0, keepdims=True).shape == (1, 2, 2)
    assert norm(a, axis=-1, keepdims=True).shape == (2, 2, 1)
    assert norm(a, axis=(0, 1), keepdims=True).shape == (1, 1, 2)
    assert norm(a, axis=(1, 2), keepdims=True).shape == (2, 1, 1)
    assert norm(a, axis=(0, -1), keepdims=True).shape == (1, 2, 1)


def test_rms():
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

    assert np.allclose(rms(a), np.sqrt(140. / 8.))
    assert np.allclose(
        rms(a, axis=0), np.sqrt(np.array([[16, 26], [40, 58]]) / 2.))
    assert np.allclose(
        rms(a, axis=-1), np.sqrt(np.array([[1, 13], [41, 85]]) / 2.))
    assert np.allclose(rms(a, axis=(1, 2)), np.sqrt(np.array([14, 126]) / 4.))
    assert np.allclose(rms(a, axis=(0, -1)), np.sqrt(np.array([42, 98]) / 4.))

    assert rms(a, keepdims=True).shape == (1, 1, 1)
    assert rms(a, axis=0, keepdims=True).shape == (1, 2, 2)
    assert rms(a, axis=-1, keepdims=True).shape == (2, 2, 1)
    assert rms(a, axis=(0, 1), keepdims=True).shape == (1, 1, 2)
    assert rms(a, axis=(1, 2), keepdims=True).shape == (2, 1, 1)
    assert rms(a, axis=(0, -1), keepdims=True).shape == (1, 2, 1)
