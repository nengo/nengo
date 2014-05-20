import logging

import numpy as np
import pytest

import nengo
from nengo.utils.inspect import checked_call

logger = logging.getLogger(__name__)


def test_checked_call():
    def func1(a):
        return a

    def func2(a, b=0, **kwargs):
        return a+b

    def func3(a, b=0, c=0, *args, **kwargs):
        return a+b+c+sum(args)

    func4 = lambda x=[0]: sum(x)

    class A(object):
        def __call__(self, a, b):
            return a + b

    assert checked_call(func1) == (None, False)
    assert checked_call(func1, 1) == (1, True)
    assert checked_call(func1, 1, 2) == (None, False)
    assert checked_call(func1, 1, two=2) == (None, False)

    assert checked_call(func2) == (None, False)
    assert checked_call(func2, 1) == (1, True)
    assert checked_call(func2, 1, 2) == (3, True)
    assert checked_call(func2, 1, 2, three=3) == (3, True)
    assert checked_call(func2, 1, 2, 3) == (None, False)

    assert checked_call(func3) == (None, False)
    assert checked_call(func3, 1) == (1, True)
    assert checked_call(func3, 1, 2) == (3, True)
    assert checked_call(func3, 1, 2, 3) == (6, True)
    assert checked_call(func3, 1, 2, 3, 4, 5, 6) == (21, True)

    assert checked_call(func4) == (0, True)
    assert checked_call(func4, [1, 2]) == (3, True)
    assert checked_call(func4, [1], 2) == (None, False)

    assert checked_call(A(), 1) == (None, False)
    assert checked_call(A(), 1, 2) == (3, True)
    assert checked_call(A(), 1, 2, 3) == (None, False)

    assert checked_call(np.sin) == (None, False)
    assert checked_call(np.sin, 0) == (0, True)
    assert checked_call(np.sin, 0, np.array([1])) == (np.array([0]), True)
    assert checked_call(np.sin, 0, np.array([1]), 1) == (None, False)


def test_checked_call_errors():
    class A(object):
        def __call__(self, a):
            raise NotImplementedError()

    assert checked_call(A()) == (None, False)
    with pytest.raises(NotImplementedError):
        checked_call(A(), 1)

    assert checked_call(np.sin, 1, 2, 3) == (None, False)
    with pytest.raises(ValueError):
        checked_call(lambda x: np.sin(1, 2, 3), 1)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
