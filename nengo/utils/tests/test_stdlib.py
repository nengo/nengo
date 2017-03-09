import sys
import weakref

import numpy as np
import pytest

from nengo.utils.compat import range
from nengo.utils.stdlib import (
    checked_call, groupby, Timer, WeakKeyIDDictionary)


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
    assert checked_call(np.sin, 0, np.array([1.])) == (np.array([0.]), True)
    assert checked_call(np.sin, 0, np.array([1.]), 1) == (None, False)


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


@pytest.mark.parametrize(
    "hashable, force_list",
    [(False, False), (False, True), (True, False), (True, True)])
def test_groupby(hashable, force_list, rng):
    if hashable:
        keys = list(range(1, 5))
    else:
        keys = [[0, 0], [0, 1], [1, 0], [1, 1]]

    keys = sorted(keys)

    # make groups and pairs
    groups = [rng.randn(rng.randint(5, 10)) for _ in keys]

    pairs = []
    for key, group in zip(keys, groups):
        pairs.extend((key, value) for value in group)

    # shuffle pairs
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]

    # call groupby
    keygroups = groupby(pairs, lambda p: p[0], force_list=force_list)

    keys2 = sorted(map(lambda x: x[0], keygroups))
    assert keys2 == keys

    for key2, keygroup2 in keygroups:
        group = groups[keys.index(key2)]
        group2 = map(lambda x: x[1], keygroup2)
        assert sorted(group2) == sorted(group)


def test_timer():
    with Timer() as timer:
        for _ in range(1000):
            2 + 2
    assert timer.duration > 0.0
    assert timer.duration < 1.0  # Pretty bad worst case


class C:
    def method(self):
        pass


def test_make_weakkeydict_from_dict():
    o = C()
    d = WeakKeyIDDictionary({o: 364})
    assert d[o] == 364


def test_make_weakkeydict_from_weakkeydict():
    o = C()
    d1 = WeakKeyIDDictionary({o: 364})
    d2 = WeakKeyIDDictionary(d1)
    assert d1[o] == 364
    assert d2[o] == 364


def test_weakkeydict_popitem(key1=C(), key2=C(), value1="v1", value2="v2"):
    d = WeakKeyIDDictionary()
    d[key1] = value1
    d[key2] = value2
    assert len(d) == 2
    k, v = d.popitem()
    assert len(d) == 1
    if k is key1:
        assert v is value1
    else:
        assert v is value2
    k, v = d.popitem()
    assert len(d) == 0
    if k is key1:
        assert v is value1
    else:
        assert v is value2


def test_weakkeydict_setdefault(key=C(), value1="v1", value2="v2"):
    d = WeakKeyIDDictionary()
    o = d.setdefault(key, value1)
    assert o is value1
    assert key in d
    assert d.get(key) is value1
    assert d[key] is value1

    o = d.setdefault(key, value2)
    assert o is value1
    assert key in d
    assert d.get(key) is value1
    assert d[key] is value1


def test_weakkeydict_update(in_d={C(): 1, C(): 2, C(): 3}):
    """This exercises d.update(), len(d), d.keys(), in d,  d.get(), d[]."""
    d = WeakKeyIDDictionary()
    d.update(in_d)
    assert len(d) == len(in_d)
    for k in d.keys():
        assert k in in_d, "mysterious new key appeared in weak dict"
        v = in_d.get(k)
        assert v is d[k]
        assert v is d.get(k)
    for k in in_d.keys():
        assert k in d, "original key disappeared in weak dict"
        v = in_d[k]
        assert v is d[k]
        assert v is d.get(k)


def test_weakkeydict_delitem():
    d = WeakKeyIDDictionary()
    o1 = C()
    o2 = C()
    d[o1] = 'something'
    d[o2] = 'something'
    assert len(d) == 2
    del d[o1]
    assert len(d) == 1
    assert list(d.keys()) == [o2]


def test_weakkeydict_bad_delitem():
    d = WeakKeyIDDictionary()
    o = C()
    # An attempt to delete an object that isn't there should raise KeyError.
    with pytest.raises(KeyError):
        del d[o]
    with pytest.raises(KeyError):
        d[o]

    # If a key isn't of a weakly referencable type, __getitem__ and
    # __setitem__ raise TypeError.  __delitem__ should too.
    with pytest.raises(TypeError):
        del d[13]
    with pytest.raises(TypeError):
        d[13]
    with pytest.raises(TypeError):
        d[13] = 13


def test_weakkeydict_frees_values():
    d = WeakKeyIDDictionary()
    k = C()
    v = C()
    d[k] = v
    weak_v = weakref.ref(v)
    del v
    assert sys.getrefcount(weak_v()) > 1  # function argument might make it > 1
    del k
    v = weak_v()
    assert v is None, "Value in WeakKeyIDDictionary not garbage collected."
