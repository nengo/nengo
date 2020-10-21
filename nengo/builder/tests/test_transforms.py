import numpy as np
import pytest

from nengo.builder.transforms import multiply
from nengo.exceptions import BuildError


def test_multiply():
    sca = np.array(4)
    vec = np.array([2, 3])
    mat = np.array([[1, 2], [3, 4]])

    assert np.array_equal(multiply(sca, sca), sca * sca)
    assert np.array_equal(multiply(sca, vec), sca * vec)
    assert np.array_equal(multiply(vec, sca), sca * vec)
    assert np.array_equal(multiply(sca, mat), sca * mat)
    assert np.array_equal(multiply(mat, sca), sca * mat)

    assert np.array_equal(multiply(vec, vec), vec * vec)
    assert np.array_equal(multiply(vec, mat), np.diag(vec).dot(mat))
    assert np.array_equal(multiply(mat, vec), mat.dot(np.diag(vec)))
    assert np.array_equal(multiply(mat, mat), mat.dot(mat))

    with pytest.raises(BuildError):
        ary3 = np.ones((2, 2, 2))
        multiply(ary3, mat)
