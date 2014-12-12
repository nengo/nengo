from __future__ import absolute_import
from __future__ import print_function

import logging

import numpy as np
import pytest

import nengo
from nengo.utils.numpy import meshgrid_nd

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
