import numpy as np
import pytest

import nengo
from nengo.helpers import piecewise


def test_basic():
    f = piecewise({0.5: 1, 1.0: 0})
    assert f(-10) == [0]
    assert f(0) == [0]
    assert f(0.25) == [0]
    assert f(0.5) == [1]
    assert f(0.75) == [1]
    assert f(1.0) == [0]
    assert f(1.5) == [0]
    assert f(100) == [0]


def test_lists():
    f = piecewise({0.5: [1, 0], 1.0: [0, 1]})
    assert f(-10) == [0, 0]
    assert f(0) == [0, 0]
    assert f(0.25) == [0, 0]
    assert f(0.5) == [1, 0]
    assert f(0.75) == [1, 0]
    assert f(1.0) == [0, 1]
    assert f(1.5) == [0, 1]
    assert f(100) == [0, 1]


def test_invalid_key():
    with pytest.raises(TypeError):
        f = piecewise({0.5: 1, 1: 0, 'a': 0.2})
        assert f


def test_invalid_length():
    with pytest.raises(Exception):
        f = piecewise({0.5: [1, 0], 1.0: [1, 0, 0]})
        assert f


def test_invalid_function_length():
    with pytest.raises(Exception):
        f = piecewise({0.5: 0, 1.0: lambda t: [t, t ** 2]})
        assert f


def test_function():
    f = piecewise({0: np.sin, 0.5: np.cos})
    assert f(0) == [np.sin(0)]
    assert f(0.25) == [np.sin(0.25)]
    assert f(0.4999) == [np.sin(0.4999)]
    assert f(0.5) == [np.cos(0.5)]
    assert f(0.75) == [np.cos(0.75)]
    assert f(1.0) == [np.cos(1.0)]


def test_function_list():

    def func1(t):
        return t, t**2, t**3

    def func2(t):
        return t**4, t**5, t**6

    f = piecewise({0: func1, 0.5: func2})
    assert f(0) == func1(0)
    assert f(0.25) == func1(0.25)
    assert f(0.4999) == func1(0.4999)
    assert f(0.5) == func2(0.5)
    assert f(0.75) == func2(0.75)
    assert f(1.0) == func2(1.0)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
